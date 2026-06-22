"""
phase2_ensemble.py -- device-instance x seed ensemble for the Phase-2 controller
headline claims (addresses methodology-review W6: 0.996x / rho=0.962 / dK_max
were single-realization point estimates).

Each seed draws a FRESH device instance (fab_seed) AND a fresh training seed,
trains the analog PNN controller, then reports the day-0 tracking ratio, the
independent computed-torque-PD reference, the closed-loop spectral radius, and
the Lyapunov drift bound. Reports mean +/- std over N seeds.

Run:  py phase2_ensemble.py
Outputs: phase2_ensemble_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import phase2_nonlinear as p2

N_SEEDS = 6
ROOT = Path(__file__).resolve().parent


def main():
    tcfg = p2.TaskCfg()
    plant = p2.FossenAUV(p2.AUVParams())
    rows = {k: [] for k in ["pnn_err0", "dig_err", "pd_err", "tracking_ratio",
                            "rho", "drift_bound"]}
    for s in range(N_SEEDS):
        ctrl = p2.train_pnn(plant, tcfg, seed=s, fab_seed=123 + s)
        e_eval = p2.sample_e0(256, tcfg, gen=torch.Generator().manual_seed(100 + s))
        pnn_err0 = p2.eval_tracking(lambda eta, nu, e: ctrl.forward_truth(e), plant, tcfg, e_eval)
        dig_err = p2.eval_tracking(lambda eta, nu, e: ctrl.forward_clean(e), plant, tcfg, e_eval)
        pd_err = p2.eval_tracking(lambda eta, nu, e: p2.digital_baseline_tau(plant, eta, nu, tcfg),
                                  plant, tcfg, e_eval)
        A, B_u, Acl0, K0 = p2.transition_jacobians(plant, ctrl, tcfg, N=40, truth=True)
        rho0 = p2.spectral_radius(Acl0)
        Q0 = 0.02 * torch.eye(10)
        if rho0 < 0.9999:
            P = p2.discrete_lyapunov(Acl0, Q0)
            lam = torch.linalg.eigvalsh(Q0).min()
            nA = torch.linalg.matrix_norm(Acl0, 2); nB = torch.linalg.matrix_norm(B_u, 2)
            nP = torch.linalg.matrix_norm(P, 2)
            bound = float(torch.clamp((-nA + torch.sqrt(nA ** 2 + lam / nP)) / nB, min=0.0))
        else:
            bound = 0.0
        rows["pnn_err0"].append(pnn_err0); rows["dig_err"].append(dig_err)
        rows["pd_err"].append(pd_err); rows["tracking_ratio"].append(pnn_err0 / dig_err)
        rows["rho"].append(rho0); rows["drift_bound"].append(bound)
        print(f"  seed {s}: ratio {pnn_err0/dig_err:.3f}  pnn {pnn_err0:.3f}  dig {dig_err:.3f}  "
              f"pd {pd_err:.3f}  rho {rho0:.3f}  dKmax {bound:.4f}")

    stat = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                "vals": [round(x, 4) for x in v]} for k, v in rows.items()}
    res = {"n_seeds": N_SEEDS, "note": "each seed = fresh device instance (fab_seed) + training seed",
           "statistics": stat}
    (ROOT / "phase2_ensemble_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(f"\n===== PHASE 2 ENSEMBLE ({N_SEEDS} device instances x seeds) =====")
    lbl = {"tracking_ratio": "PNN / digital-twin tracking ratio", "pnn_err0": "PNN tracking error",
           "dig_err": "digital-twin tracking error", "pd_err": "computed-torque PD (independent)",
           "rho": "spectral radius rho(Acl, 2.0s)", "drift_bound": "Lyapunov drift bound dK_max"}
    for k in ["tracking_ratio", "pnn_err0", "dig_err", "pd_err", "rho", "drift_bound"]:
        print(f"{lbl[k]:36s} {stat[k]['mean']:.4f} +/- {stat[k]['std']:.4f}")


if __name__ == "__main__":
    main()
