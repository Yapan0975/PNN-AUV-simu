"""
iara_volume_split.py -- volume-aware sensitivity for the background-volume confound
(PDF review r4, strongly-recommended #1). The IARA archive collected its no-ship
background as separate sessions (volumes E and H), so in the 225 subset all 45
Background recordings come from E/H while the four vessel classes come from A-D/F/G;
a reviewer may worry the low ceiling / the ship-vs-background detector is a
volume-tag artefact rather than acoustic.

Two tests on the SAME cached 225-subset log-mel features (no re-extraction):

  TEST 1 (leave-one-volume-out, 4-class SHIP-TYPE, no background).
    The hard part of the task -- vessel-TYPE discrimination, the source of the low
    5-class ceiling (confusion matrix) -- is tested across volumes: for each held-out
    volume v in {A,B,C,D} (each carries all four ship types), train a ResNet on ships
    from the OTHER volumes and test on ships in v. If 4-class accuracy holds across
    held-out volumes, type discrimination is acoustic, not a volume signature.

  TEST 2 (cross-volume background detection).
    Ship-vs-background detection with the background VOLUME held out: train the
    detector with background drawn only from volume E and test it on background drawn
    only from volume H (ships split by recording). If ROC-AUC stays near the random-
    split 0.79, the detector is not keying on a volume-E-specific channel signature.

Run:  py -u iara_volume_split.py  ->  iara_volume_split_results.json
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import iara_dataset as iara
import models_bench as mb
from phase0_iara import rec_balanced_acc

torch.set_num_threads(16)
ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "IARA-data"
XLSX = str(DATA / "iara.xlsx")
LM = ROOT / "iara_logmel_features.npz"
CLASSES5 = ("Background", "Cargo", "Tanker", "Tug", "Special Craft")
EPOCHS = 30
SEEDS = [0, 1, 2]


def train_resnet_split(X, y, k, tr_idx, te_idx, rec_te, seed):
    torch.manual_seed(seed)
    n_mels, T = X.shape[2], X.shape[3]
    m = mb.build("resnet", n_mels, T, k)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=1e-4)
    Xtr, ytr = X[tr_idx], y[tr_idx]
    n = len(Xtr)
    for _ in range(EPOCHS):
        perm = torch.randperm(n); m.train()
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad(); F.cross_entropy(m(Xtr[idx]), ytr[idx]).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        prob = torch.softmax(m(X[te_idx]), dim=1).numpy()
    return prob


def roc_auc(scores, labels):
    s = np.asarray(scores); yv = np.asarray(labels)
    order = np.argsort(-s); yv = yv[order]
    P = yv.sum(); N = len(yv) - P
    tp = np.cumsum(yv); fp = np.cumsum(1 - yv)
    tpr = tp / max(P, 1); fpr = fp / max(N, 1)
    fpr = np.concatenate([[0.], fpr]); tpr = np.concatenate([[0.], tpr])
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.))


def rec_pool_scores(prob_ship, yb, rec_te):
    out = []
    for r in np.unique(rec_te):
        msk = rec_te == r
        out.append((float(prob_ship[msk].mean()), int(yb[msk][0])))
    return out


def main():
    d = np.load(LM, allow_pickle=True)
    X = torch.from_numpy(d["X"]); y = torch.from_numpy(d["y"]).long(); rec = d["rec"]
    id2cls, id2vol = iara.build_iara_label_map(XLSX)
    vol = np.array([id2vol.get(int(r), "?") for r in rec])
    print("[vol] segment volume counts:", dict(Counter(vol)), flush=True)
    out = {"task": "volume-aware sensitivity for the background-volume confound (PDF review r4)",
           "reference_random_split": {"resnet_4cls_ship": 0.439, "detection_auc": 0.793}}

    # ---------- TEST 1: leave-one-volume-out, 4-class ship-type ----------
    ship = (y != 0).numpy()
    yship = (y[torch.from_numpy(ship)] - 1)           # remap 1..4 -> 0..3
    Xship = X[torch.from_numpy(ship)]
    recship = rec[ship]; volship = vol[ship]
    lovo = {}
    for v in ["A", "B", "C", "D"]:
        te = volship == v
        tr = ~te
        if te.sum() == 0 or len(np.unique(yship[torch.from_numpy(tr)])) < 4:
            continue
        accs = []
        tr_idx = torch.from_numpy(np.where(tr)[0]); te_idx = torch.from_numpy(np.where(te)[0])
        for s in SEEDS:
            prob = train_resnet_split(Xship, yship, 4, tr_idx, te_idx, recship[te], s)
            accs.append(rec_balanced_acc(prob, yship[te_idx], recship[te], 4))
        lovo[v] = {"n_test_rec": int(len(np.unique(recship[te]))),
                   "resnet_4cls_mean": round(float(np.mean(accs)), 3),
                   "resnet_4cls_std": round(float(np.std(accs)), 3),
                   "vals": [round(a, 3) for a in accs]}
        print(f"  LOVO hold-out {v}: 4-cls {lovo[v]['resnet_4cls_mean']}+/-{lovo[v]['resnet_4cls_std']}"
              f"  (n_test_rec {lovo[v]['n_test_rec']})", flush=True)
    macc = [v2["resnet_4cls_mean"] for v2 in lovo.values()]
    out["test1_leave_one_volume_out_4class_ship"] = {
        "per_heldout_volume": lovo,
        "across_volumes_mean": round(float(np.mean(macc)), 3),
        "across_volumes_std": round(float(np.std(macc)), 3),
        "note": "compare to random-split 4-class 0.439; stability across held-out volumes "
                "indicates ship-type discrimination is acoustic, not a volume signature"}

    # ---------- TEST 2: cross-volume background detection (train bg=E, test bg=H) ----------
    # ships split by recording 70/30; background by volume (E -> train, H -> test)
    bg = (y == 0).numpy()
    bgE = bg & (vol == "E"); bgH = bg & (vol == "H")
    aucs = []
    for s in SEEDS:
        tr_s, te_s = iara.grouped_split(y[torch.from_numpy(ship)], recship, test_frac=0.3, seed=s)
        ship_idx = np.where(ship)[0]
        tr_ship = ship_idx[tr_s.numpy()]; te_ship = ship_idx[te_s.numpy()]
        tr_idx = np.concatenate([tr_ship, np.where(bgE)[0]])
        te_idx = np.concatenate([te_ship, np.where(bgH)[0]])
        np.random.RandomState(s).shuffle(tr_idx)
        ybin = (y != 0).long()
        prob = train_resnet_split(X, ybin, 2, torch.from_numpy(tr_idx), torch.from_numpy(te_idx),
                                  rec[te_idx], s)
        pts = rec_pool_scores(prob[:, 1], ybin[torch.from_numpy(te_idx)].numpy(), rec[te_idx])
        sc = [p[0] for p in pts]; lb = [p[1] for p in pts]
        aucs.append(roc_auc(sc, lb))
        print(f"  cross-vol detection seed {s}: AUC {aucs[-1]:.3f}  (train bg=E, test bg=H)", flush=True)
    out["test2_cross_volume_background_detection"] = {
        "train_bg_volume": "E", "test_bg_volume": "H",
        "n_bg_E": int(bgE.sum() and len(np.unique(rec[bgE]))), "n_bg_H": int(len(np.unique(rec[bgH]))),
        "detection_auc_mean": round(float(np.mean(aucs)), 3),
        "detection_auc_std": round(float(np.std(aucs)), 3),
        "vals": [round(a, 3) for a in aucs],
        "note": "background volume held out (train on E, test on H); AUC near the random-split 0.79 "
                "would indicate the detector is acoustic, not keyed to a volume-specific signature"}

    (ROOT / "iara_volume_split_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n[vol] TEST1 LOVO 4-class ship across volumes:",
          out["test1_leave_one_volume_out_4class_ship"]["across_volumes_mean"], "+/-",
          out["test1_leave_one_volume_out_4class_ship"]["across_volumes_std"])
    print("[vol] TEST2 cross-volume detection AUC:",
          out["test2_cross_volume_background_detection"]["detection_auc_mean"])
    print("  -> iara_volume_split_results.json")


if __name__ == "__main__":
    main()
