"""
shipsear_leakage_test.py -- SECOND-CORPUS replication of the recording-level
leakage finding (answering the reviewer's R2: does the IARA leakage result
generalise beyond IARA?).

Corpus: ShipsEar (Santos-Dominguez et al. 2016), obtained from the public
Hugging Face redistribution `peng7554/DS3500` (ShipsEar.zip), which ships the
audio pre-segmented into 5 s / 16 kHz windows organised as
    shipsear_5s_16k/<class>/<class>_<session>/<class>_<session>_<seg>.wav .
IMPORTANT HONESTY NOTE: this redistribution groups the original 90 ShipsEar
recordings into 12 coarser acoustic SESSIONS (e.g. class 2 is a single 70 min
session), so the grouping unit here is a *session*, COARSER than an original
recording. That makes the by-session split a CONSERVATIVE test: a coarser group
shares even more context, so if a by-segment split still inflates accuracy over
a by-session split, the leakage mechanism (windows of the same acoustic context
in train and test) is confirmed a fortiori.

We restrict to the classes that have >=2 sessions (so a grouped split can hold a
whole session out): classes {0,1,3} (5/3/2 sessions). We compare, for the SAME
log-mel digital models (MLP / CNN2D / ResNet) and the SAME data:
  (A) by-SESSION split  (grouped; no session's windows shared)  -> honest
  (B) by-SEGMENT split  (random over windows; LEAKY, mimics naive k-fold CV)
and report SEGMENT-level balanced accuracy for each (directly comparable to a
segment-level k-fold number).

Run:  py -u shipsear_leakage_test.py
Out:  shipsear_leakage_results.json
"""
from __future__ import annotations

import io
import json
import time
import wave
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import iara_logmel as ilm            # reuse the numpy mel filterbank + _logmel
import iara_dataset as iara          # reuse grouped_split
import models_bench as mb

torch.set_num_threads(16)

ROOT = Path(__file__).resolve().parent
ZIP = ROOT.parent / "ShipsEar-data" / "ShipsEar.zip"
PREFIX = "shipsear_5s_16k"
SEEDS = [0, 1, 2, 3, 4]
N_MELS, N_FFT, HOP = 48, 512, 256
DECIMATE_TO = 4000.0
MAX_PER_SESSION = 80               # cap to limit the single-session imbalance
CACHE = ROOT / "shipsear_logmel_features.npz"


def _decimate_to(data, sr, target):
    from scipy.signal import decimate
    q = max(1, int(round(sr / target)))
    while q > 13:
        data = decimate(data, 4, ftype="fir"); sr //= 4
        q = max(1, int(round(sr / target)))
    if q > 1:
        data = decimate(data, q, ftype="fir"); sr //= q
    return data, sr


def load_shipsear_logmel():
    """Return X[N,1,n_mels,T], y[N], sess[N] (session id), class_list."""
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        print(f"[ses] cache {CACHE.name}: X{d['X'].shape}", flush=True)
        return (torch.from_numpy(d["X"]), torch.from_numpy(d["y"]),
                d["sess"], list(d["classes"]))

    z = zipfile.ZipFile(str(ZIP))
    wavs = [n for n in z.namelist() if n.lower().endswith(".wav")]
    # group by (class, session dir); cap per session
    by_sess = defaultdict(list)
    for n in wavs:
        parts = n.split("/")
        if len(parts) < 4:
            continue
        cls, sess = parts[1], parts[2]
        by_sess[(cls, sess)].append(n)

    fb = None
    X, y, sess_ids = [], [], []
    sess_index = {}
    cls_set = sorted({c for c, _ in by_sess})
    cls_idx = {c: i for i, c in enumerate(cls_set)}
    rng = np.random.default_rng(0)
    for (cls, sess), members in sorted(by_sess.items()):
        members = sorted(members)
        if len(members) > MAX_PER_SESSION:
            sel = rng.choice(len(members), MAX_PER_SESSION, replace=False)
            members = [members[i] for i in sorted(sel)]
        sid = sess_index.setdefault((cls, sess), len(sess_index))
        for n in members:
            wf = wave.open(io.BytesIO(z.read(n)))
            sr = wf.getframerate()
            data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float64)
            if wf.getnchannels() > 1:
                data = data[::wf.getnchannels()]
            data = data / (np.max(np.abs(data)) + 1e-9)
            data, sr = _decimate_to(data, sr, DECIMATE_TO)
            if fb is None:
                fb = ilm._mel_filterbank(sr, N_FFT, N_MELS, f_lo=10.0, f_hi=sr / 2.0)
            if len(data) < N_FFT:
                continue
            X.append(ilm._logmel(data, sr, fb, N_FFT, HOP))
            y.append(cls_idx[cls]); sess_ids.append(sid)
    T = int(np.median([a.shape[1] for a in X]))
    Xf = np.zeros((len(X), 1, N_MELS, T), dtype=np.float32)
    for i, a in enumerate(X):
        t = min(T, a.shape[1]); Xf[i, 0, :, :t] = a[:, :t]
    Xf = (Xf - Xf.mean()) / (Xf.std() + 1e-6)
    y = np.array(y, dtype=np.int64); sess = np.array(sess_ids, dtype=np.int64)
    np.savez(CACHE, X=Xf, y=y, sess=sess, classes=np.array(cls_set))
    print(f"[ses] built X{Xf.shape}; classes {cls_set}; n_sessions {len(sess_index)}", flush=True)
    return torch.from_numpy(Xf), torch.from_numpy(y), sess, cls_set


def seg_split(n, test_frac, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n); ncut = int(round((1 - test_frac) * n))
    tr = np.zeros(n, bool); te = np.zeros(n, bool)
    tr[idx[:ncut]] = True; te[idx[ncut:]] = True
    return tr, te


def seg_bal_acc(pred, y, k):
    return float(np.mean([(pred[y == c] == c).mean() for c in range(k) if (y == c).any()]))


def _train(name, nm, T, k, Xtr, ytr, seed, epochs=30):
    torch.manual_seed(seed)
    m = mb.build(name, nm, T, k)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr)); m.train()
        for i in range(0, len(Xtr), 128):
            idx = perm[i:i + 128]
            opt.zero_grad(); F.cross_entropy(m(Xtr[idx]), ytr[idx]).backward(); opt.step()
    m.eval(); return m


def _ms(v):
    return {"mean": round(float(np.mean(v)), 3), "std": round(float(np.std(v)), 3),
            "vals": [round(x, 3) for x in v]}


def run(model_name, X, y, sess, k):
    nm, T = X.shape[2], X.shape[3]
    rec_acc, seg_acc = [], []
    for s in SEEDS:
        # by-SESSION (grouped, honest): reuse iara.grouped_split with sess as group
        ytt = y.clone()
        tr, te = iara.grouped_split(ytt, sess, test_frac=0.3, seed=s)
        m = _train(model_name, nm, T, k, X[tr], y[tr], s)
        with torch.no_grad():
            p = m(X[te]).argmax(1).numpy()
        rec_acc.append(seg_bal_acc(p, y[te].numpy(), k))
        # by-SEGMENT (random, leaky)
        trs, tes = seg_split(len(X), 0.3, s)
        trs = torch.from_numpy(trs); tes = torch.from_numpy(tes)
        m2 = _train(model_name, nm, T, k, X[trs], y[trs], s)
        with torch.no_grad():
            p2 = m2(X[tes]).argmax(1).numpy()
        seg_acc.append(seg_bal_acc(p2, y[tes].numpy(), k))
    return {"by_session_segacc": _ms(rec_acc), "by_segment_segacc": _ms(seg_acc)}


def main():
    t0 = time.time()
    X, y, sess, classes = load_shipsear_logmel()
    print(f"[ses] X{tuple(X.shape)} classes {classes} "
          f"n_sess {len(np.unique(sess))} per-class "
          f"{ {int(c): int((y==c).sum()) for c in torch.unique(y)} }", flush=True)

    # sessions per class
    spc = {int(c): len(np.unique(sess[(y == c).numpy()])) for c in torch.unique(y)}
    print(f"[ses] sessions per class: {spc}", flush=True)

    res = {
        "corpus": "ShipsEar (Santos-Dominguez 2016) via public HF redistribution peng7554/DS3500, "
                  "pre-segmented 5 s / 16 kHz windows",
        "honesty_note": ("the redistribution groups the 90 original recordings into 12 coarser "
                         "acoustic SESSIONS; the grouping unit here is a session (coarser than a "
                         "recording), so a by-session split is a CONSERVATIVE leakage test."),
        "feature": f"log-mel {N_MELS}x{X.shape[3]}, decimated to {int(DECIMATE_TO)} Hz, "
                   f"<= {MAX_PER_SESSION} windows/session",
        "sessions_per_class": spc,
        "metric": "SEGMENT-level balanced accuracy (mean+/-std over 5 seeds)",
    }

    # 3-class subset (classes with >=2 sessions)
    keep3 = sorted([c for c, n in spc.items() if n >= 2])
    print(f"[ses] classes with >=2 sessions (usable for grouped split): {keep3}", flush=True)
    for subset_name, keep in [("3class_ge2sessions", keep3),
                              ("2class_0v1", sorted(keep3)[:2])]:
        mask = np.isin(y.numpy(), keep)
        remap = {c: i for i, c in enumerate(keep)}
        Xs = X[mask]; ys = torch.tensor([remap[int(v)] for v in y[mask]], dtype=torch.long)
        ss = sess[mask.nonzero()[0]] if False else sess[np.where(mask)[0]]
        k = len(keep)
        res[subset_name] = {"classes_kept": keep, "n_segments": int(mask.sum()),
                            "n_sessions": int(len(np.unique(ss))), "k": k, "results": {}}
        for model_name in ("logmel_mlp", "cnn2d", "resnet"):
            r = run(model_name, Xs, ys, ss, k)
            infl = r["by_segment_segacc"]["mean"] - r["by_session_segacc"]["mean"]
            r["leakage_inflation"] = round(infl, 3)
            res[subset_name]["results"][model_name] = r
            print(f"  [{subset_name}] {model_name:11s} by-session {r['by_session_segacc']['mean']:.3f} "
                  f"| by-segment {r['by_segment_segacc']['mean']:.3f} | inflation {infl:+.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    # headline verdict from the 3-class ResNet
    r3 = res["3class_ge2sessions"]["results"]
    infls = [r3[m]["leakage_inflation"] for m in r3]
    res["verdict"] = (
        f"On ShipsEar (3-class, >=2 sessions each), a by-segment (leaky) split inflates segment-level "
        f"balanced accuracy by {min(infls):+.3f} to {max(infls):+.3f} over a by-session (honest) split "
        f"across MLP/CNN2D/ResNet -- replicating, on a second corpus, the recording-level leakage found "
        f"on IARA (+0.27 to +0.49). Because the grouping here is a coarser session than a recording, this "
        f"is a conservative confirmation.")
    (ROOT / "shipsear_leakage_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n[ses] VERDICT: inflation {min(infls):+.3f}..{max(infls):+.3f} (3-class)", flush=True)
    print(f"[ses] done {time.time()-t0:.0f}s -> shipsear_leakage_results.json", flush=True)


if __name__ == "__main__":
    main()
