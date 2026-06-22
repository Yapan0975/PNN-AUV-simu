"""
phase0_ensemble.py -- device-instance x seed ensemble for the Phase-0 acoustic
headline claims (addresses methodology-review W6: the headline 0.990 / +0.265 /
frozen-ablation were single-realization point estimates).

Each seed draws a FRESH device instance (fabrication dispersion + thermal-noise
realization) AND a fresh training seed, then runs the full Phase-0 comparison.
Reports mean +/- std over N seeds. Energy is deterministic (device model), not
re-ensembled here.

Run:  py phase0_ensemble.py
Outputs: phase0_ensemble_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import phase0_acoustic as p0

N_SEEDS = 8
ROOT = Path(__file__).resolve().parent


def main():
    rows = {k: [] for k in ["digital", "pnn_pat", "frozen", "insilico_deploy",
                            "pat_minus_insilico", "physics_contribution", "grad_cosine"]}
    for s in range(N_SEEDS):
        cfg = p0.Config(seed=s)
        np_rng = np.random.default_rng(s)
        noise_rng = torch.Generator().manual_seed(1000 + s)   # device instance + noise
        Xtr, ytr = p0.make_dataset(cfg, cfg.n_train, np_rng)
        Xte, yte = p0.make_dataset(cfg, cfg.n_test, np_rng)

        acc_dig = p0.train_digital(cfg, Xtr, ytr, Xte, yte)
        model_pat, acc_pat = p0.train_pnn_pat(cfg, Xtr, ytr, Xte, yte, noise_rng)
        _, acc_frozen = p0.train_pnn_pat(cfg, Xtr, ytr, Xte, yte, noise_rng, freeze=True)
        _, acc_is_deploy = p0.train_pnn_insilico(cfg, Xtr, ytr, Xte, yte, noise_rng)
        cos = p0.pat_gradient_cosine(cfg, model_pat, Xte[:256], yte[:256], noise_rng)

        rows["digital"].append(acc_dig)
        rows["pnn_pat"].append(acc_pat)
        rows["frozen"].append(acc_frozen)
        rows["insilico_deploy"].append(acc_is_deploy)
        rows["pat_minus_insilico"].append(acc_pat - acc_is_deploy)
        rows["physics_contribution"].append(acc_pat - acc_frozen)
        rows["grad_cosine"].append(cos)
        print(f"  seed {s}: dig {acc_dig:.3f} pat {acc_pat:.3f} frozen {acc_frozen:.3f} "
              f"insilico {acc_is_deploy:.3f} cos {cos:.3f}")

    stat = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                "vals": [round(x, 4) for x in v]} for k, v in rows.items()}
    res = {"n_seeds": N_SEEDS, "note": "each seed = fresh device instance (fab dispersion) "
           "+ noise realization + training seed", "statistics": stat}
    (ROOT / "phase0_ensemble_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(f"\n===== PHASE 0 ENSEMBLE ({N_SEEDS} device instances x seeds) =====")
    lbl = {"digital": "digital baseline", "pnn_pat": "PNN-PAT (noisy dev.)",
           "frozen": "frozen rand. physics", "insilico_deploy": "in-silico deploy",
           "pat_minus_insilico": "PAT - in-silico (reality gap)",
           "physics_contribution": "trained-physics gain (PAT - frozen)",
           "grad_cosine": "PAT gradient cosine"}
    for k in ["digital", "pnn_pat", "frozen", "insilico_deploy",
              "pat_minus_insilico", "physics_contribution", "grad_cosine"]:
        print(f"{lbl[k]:34s} {stat[k]['mean']:+.3f} +/- {stat[k]['std']:.3f}")


if __name__ == "__main__":
    main()
