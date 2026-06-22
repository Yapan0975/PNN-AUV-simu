"""
iara_drift_recal.py -- drift recalibration on REAL IARA features (answering the
re-review's request: the 0.578->0.843 recovery is a synthetic noisy-device proxy;
show a drift-recalibration result on real IARA recordings too).

We take the cached IARA log-mel features (real 128 kHz hydrophone recordings,
leakage-free by-recording split), train a digital classifier at "day 0", then
apply a PROGRESSIVE physical-drift model to the spectro-temporal features over a
30-day mission and compare two paths:
  (A) NO recalibration -- the day-0 model is frozen and sees drifting inputs;
  (B) periodic recalibration -- every 5 days a few labelled reference windows
      from the (drifted) training recordings are used for a short fine-tune,
      mimicking the surface-window PAT update.

Drift model on the log-mel feature (a proxy for biofouling + thermal detuning of
the resonator front end):
  * resonance DETUNING  -> roll the mel axis by s(day) bins;
  * biofouling DAMPING  -> subtract a high-band-weighted attenuation a(day)*w(mel)
    in the log domain (high mel bins damped more as fouling grows);
  * added structured NOISE -> N(0, sigma*day/30).

Metric: recording-level balanced accuracy (leakage-free), mean+/-std over seeds.

Run:  py -u iara_drift_recal.py
Out:  iara_drift_recal_results.json, iara_drift_recal_figure.png
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import iara_dataset as iara
import iara_logmel as ilm
import models_bench as mb

torch.set_num_threads(16)

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "IARA-data"
CLASSES = ("Background", "Cargo", "Tanker", "Tug", "Special Craft")
CACHE = ROOT / "iara_logmel_features.npz"
SEEDS = [0, 1, 2, 3]
DAYS = [0, 5, 10, 15, 20, 25, 30]
RECAL_EVERY = 5
RECAL_REFS = 160          # labelled reference windows used per recalibration
RECAL_STEPS = 40          # fine-tune steps per recalibration (surface-window budget)
# drift strength (tuned so day-30 no-recal degrades clearly but is recoverable)
SHIFT_MAX = 4             # mel bins of detuning by day 30
ATTEN_MAX = 2.5           # log-domain high-band attenuation by day 30
NOISE_MAX = 0.6           # std of added log-domain noise by day 30


def rec_bal_acc(prob, yte, rec_te, k):
    yn = yte.numpy(); preds, trues = [], []
    for r in np.unique(rec_te):
        m = rec_te == r
        preds.append(int(prob[m].mean(0).argmax())); trues.append(int(yn[m][0]))
    preds, trues = np.array(preds), np.array(trues)
    return float(np.mean([(preds[trues == c] == c).mean() for c in range(k) if (trues == c).any()]))


def drift(X, day, rng):
    """Apply the progressive physical-drift model to log-mel tensor X[N,1,M,T]."""
    if day == 0:
        return X.clone()
    f = day / 30.0
    Xd = X.clone()
    M = X.shape[2]
    # 1) resonance detuning: roll mel axis
    s = int(round(SHIFT_MAX * f))
    if s:
        Xd = torch.roll(Xd, shifts=s, dims=2)
        Xd[:, :, :s, :] = X[:, :, :1, :]      # clamp the wrapped low edge
    # 2) biofouling damping: high-mel-weighted attenuation (log domain)
    w = torch.linspace(0.2, 1.0, M).view(1, 1, M, 1)
    Xd = Xd - ATTEN_MAX * f * w
    # 3) structured noise
    Xd = Xd + NOISE_MAX * f * torch.from_numpy(rng.standard_normal(Xd.shape).astype(np.float32))
    return Xd


def _train(model, Xtr, ytr, steps, lr=2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(Xtr)
    for _ in range(steps):
        perm = torch.randperm(n); model.train()
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad(); F.cross_entropy(model(Xtr[idx]), ytr[idx]).backward(); opt.step()
    model.eval(); return model


def run_seed(X, y, rec, k, nm, T, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 7)
    tr, te = iara.grouped_split(y, rec, test_frac=0.3, seed=seed)
    Xtr, ytr, Xte, yte, rec_te = X[tr], y[tr], X[te], y[te], rec[te]

    base = _train(mb.build("cnn2d", nm, T, k), Xtr, ytr, steps=30)
    # recalibration model warm-starts from base and is fine-tuned over time
    recal = mb.build("cnn2d", nm, T, k)
    recal.load_state_dict(base.state_dict())
    recal.eval()   # so day-0 (pre-fine-tune) matches the frozen base exactly

    no_recal, with_recal = [], []
    for day in DAYS:
        Xte_d = drift(Xte, day, np.random.default_rng(1000 + day))
        with torch.no_grad():
            p = torch.softmax(base(Xte_d), 1).numpy()
        no_recal.append(rec_bal_acc(p, yte, rec_te, k))
        # recalibrate every RECAL_EVERY days on a few drifted TRAIN reference windows
        if day > 0 and day % RECAL_EVERY == 0:
            Xtr_d = drift(Xtr, day, rng)
            ridx = rng.choice(len(Xtr_d), min(RECAL_REFS, len(Xtr_d)), replace=False)
            recal = _train(recal, Xtr_d[ridx], ytr[ridx], steps=RECAL_STEPS, lr=1e-3)
        with torch.no_grad():
            pr = torch.softmax(recal(Xte_d), 1).numpy()
        with_recal.append(rec_bal_acc(pr, yte, rec_te, k))
    return no_recal, with_recal


def main():
    t0 = time.time()
    X, y, rec, _ = ilm.load_iara_logmel(DATA, str(DATA / "iara.xlsx"), CLASSES, cache=str(CACHE))
    nm, T = X.shape[2], X.shape[3]
    k = 5
    print(f"[drift] IARA log-mel X{tuple(X.shape)} 5-class, days {DAYS}", flush=True)

    NR, WR = [], []
    for s in SEEDS:
        nr, wr = run_seed(X, y, rec, k, nm, T, s)
        NR.append(nr); WR.append(wr)
        print(f"  seed {s}: day0 {nr[0]:.3f} | day30 no-recal {nr[-1]:.3f} | "
              f"day30 recal {wr[-1]:.3f} [{time.time()-t0:.0f}s]", flush=True)
    NR, WR = np.array(NR), np.array(WR)
    res = {
        "dataset": "IARA log-mel (real recordings), leakage-free by-recording split",
        "model": "log-mel CNN2D", "k": k, "n_seeds": len(SEEDS), "days": DAYS,
        "drift_model": {"detuning_melbins": SHIFT_MAX, "biofouling_atten_logdb": ATTEN_MAX,
                        "noise_std": NOISE_MAX, "note": "progressive over 30 days"},
        "recal": {"every_days": RECAL_EVERY, "ref_windows": RECAL_REFS, "steps": RECAL_STEPS},
        "no_recal_mean": [round(float(x), 3) for x in NR.mean(0)],
        "no_recal_std": [round(float(x), 3) for x in NR.std(0)],
        "with_recal_mean": [round(float(x), 3) for x in WR.mean(0)],
        "with_recal_std": [round(float(x), 3) for x in WR.std(0)],
    }
    res["headline"] = (
        f"On real IARA log-mel (5-class, leakage-free), 30-day progressive drift degrades the "
        f"day-0 accuracy {NR.mean(0)[0]:.3f} to {NR.mean(0)[-1]:.3f}+/-{NR.std(0)[-1]:.3f} without "
        f"recalibration; periodic surface-window recalibration (every {RECAL_EVERY} d, "
        f"{RECAL_REFS} refs, {RECAL_STEPS} steps) holds it at {WR.mean(0)[-1]:.3f}+/-{WR.std(0)[-1]:.3f} "
        f"on day 30, recovering {WR.mean(0)[-1]-NR.mean(0)[-1]:+.3f}.")
    (ROOT / "iara_drift_recal_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.errorbar(DAYS, NR.mean(0), yerr=NR.std(0), fmt="s--", color="#cc3333", capsize=3, label="no recalibration")
    ax.errorbar(DAYS, WR.mean(0), yerr=WR.std(0), fmt="o-", color="#2c6fbb", capsize=3, label="5-day recalibration")
    ax.axhline(0.2, ls=":", color="k", lw=0.8); ax.text(0, 0.205, "chance", fontsize=7)
    ax.set_xlabel("mission day (progressive drift)"); ax.set_ylabel("IARA 5-class balanced accuracy")
    ax.set_ylim(0.15, 0.55); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    ax.set_title("Drift recalibration on REAL IARA features", fontsize=9.5)
    fig.tight_layout(); fig.savefig(ROOT / "iara_drift_recal_figure.png", dpi=200)
    print("[drift] " + res["headline"], flush=True)
    print(f"[drift] done {time.time()-t0:.0f}s -> iara_drift_recal_results.json", flush=True)


if __name__ == "__main__":
    main()
