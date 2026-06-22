"""
phase2_stability_cont.py -- continuous-time local stability of the analog PNN
control loop, as an honest salvage of the discrete-window certificate.

The discrete 2.0 s transition-matrix spectral radius hovers at ~1.00 +/- 0.04
because depth/heading are quasi-integrator error modes whose slow poles sit on
the unit circle; the discrete radius is sign-sensitive near that margin. The
principled test is the CONTINUOUS closed-loop linearization: linearize the
reduced error vector field de/dt = f(e) at trim and check the spectral abscissa
alpha = max Re(eig(A_c)). alpha < 0  <=>  locally asymptotically stable.

Run:  py phase2_stability_cont.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import phase2_nonlinear as p2

N_SEEDS = 6
ROOT = Path(__file__).resolve().parent


def cont_abscissa(plant, ctrl, t, truth=True):
    """max Re eigenvalue of the continuous reduced closed-loop Jacobian at trim."""
    e0 = torch.zeros(10)

    def f_cont(e):
        eta, nu = p2.embed_state(e, t)
        uc = ctrl.forward_truth(e, smooth=True) if truth else ctrl.forward_clean(e)
        tau = plant.B_act(uc)
        etadot = plant.J(eta, nu)              # [xdot,ydot,zdot,phidot,thetadot,psidot]
        nudot = plant.nu_dot(eta, nu, tau)     # [udot..rdot]
        return torch.cat([etadot[2:6], nudot], dim=-1)   # de/dt, 10-dim

    Ac = torch.autograd.functional.jacobian(f_cont, e0, vectorize=True)
    ev = torch.linalg.eigvals(Ac)
    return float(ev.real.max())


def main():
    t = p2.TaskCfg(); plant = p2.FossenAUV(p2.AUVParams())
    alphas = []
    for s in range(N_SEEDS):
        ctrl = p2.train_pnn(plant, t, seed=s, fab_seed=123 + s)
        a = cont_abscissa(plant, ctrl, t, truth=True)
        alphas.append(a)
        print(f"  seed {s}: continuous spectral abscissa alpha = {a:+.4f}  "
              f"({'STABLE' if a < 0 else 'unstable/marginal'})")
    a = np.array(alphas)
    res = {"n_seeds": N_SEEDS,
           "metric": "continuous closed-loop spectral abscissa alpha=max Re eig(A_c)",
           "alpha_mean": float(a.mean()), "alpha_std": float(a.std()),
           "alpha_max": float(a.max()), "frac_stable": float((a < 0).mean()),
           "vals": [round(x, 5) for x in alphas]}
    (ROOT / "phase2_stability_cont_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n===== CONTINUOUS-TIME LOCAL STABILITY ({N_SEEDS} instances) =====")
    print(f"spectral abscissa alpha = {a.mean():+.4f} +/- {a.std():.4f}  (max {a.max():+.4f})")
    print(f"fraction with alpha < 0 (asymptotically stable): {100*(a<0).mean():.0f}%")
    print("VERDICT:", "robustly locally asymptotically stable" if a.max() < 0
          else "marginal -- not robustly certifiable")


if __name__ == "__main__":
    main()
