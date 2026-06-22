"""Quick task-difficulty sweep on the CACHED real IARA features.
Reuses the extracted 5-class features and evaluates, recording-level over 5
seeds, how digital baseline vs PNN-PAT do on progressively coarser tasks:
  * binary  : ship-present vs background
  * 3-class : Background / Cargo / Tanker
  * 5-class : Background / Cargo / Tanker / Tug / Special Craft
to find the task granularity at which the 10-1000 Hz front end gives a
credible, honest real-data number. No re-extraction (cache only).
"""
from __future__ import annotations
import numpy as np, torch
import core, iara_dataset as iara
from phase0_iara import train_digital, train_insilico, rec_balanced_acc, _probs

d = np.load("phase0_iara_features.npz", allow_pickle=True)
X = torch.from_numpy(d["X"]); y0 = torch.from_numpy(d["y"]); rec0 = d["rec"]
classes5 = list(d["classes"])
print("cache:", X.shape, "classes:", classes5)
SEEDS = [0, 1, 2, 3, 4]


def run_task(name, Xs, ys, recs, k, n_modes=32):
    rec_pat, rec_dig = [], []
    for s in SEEDS:
        tr, te = iara.grouped_split(ys, recs, 0.3, seed=s)
        Xtr, ytr, Xte, yte = Xs[tr], ys[tr], Xs[te], ys[te]
        rec_te = recs[te]
        cfg = core.Config(seed=s, n_classes=k, n_modes=n_modes, epochs=60, batch=128, lr=3e-3)
        nrng = torch.Generator().manual_seed(s + 100)
        dig = train_digital(cfg, Xtr, ytr)
        mp, _ = core.train_pnn_pat(cfg, Xtr, ytr, Xte, yte, nrng, return_model=True)
        rec_dig.append(rec_balanced_acc(_probs(lambda S: dig(S), Xte), yte, rec_te, k))
        rec_pat.append(rec_balanced_acc(_probs(lambda S: mp.forward_eval(S, noise_rng=nrng), Xte), yte, rec_te, k))
    print(f"{name:28s} modes={n_modes:>3} chance {1/k:.2f} | digital {np.mean(rec_dig):.3f} | "
          f"PNN-PAT {np.mean(rec_pat):.3f}+/-{np.std(rec_pat):.3f} | gap {np.mean(rec_dig)-np.mean(rec_pat):+.3f}")
    return np.mean(rec_pat), np.mean(rec_dig)


y_bin = (y0 != 0).long()
keep3 = (y0 <= 2)
print("=== how many resonators does real data need to match digital? ===")
for nm in [32, 64, 96]:
    run_task("binary ship-vs-bg", X, y_bin, rec0, 2, n_modes=nm)
for nm in [32, 64, 96]:
    run_task("3-class bg/cargo/tanker", X[keep3], y0[keep3], rec0[keep3.numpy()], 3, n_modes=nm)
