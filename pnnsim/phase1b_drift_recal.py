"""
phase1b_drift_recal.py — Phase 1B: online PAT recalibration under environmental
drift (validates paper §8; answers reviewer point 6).

A PNN-PAT acoustic classifier is trained at mission start. Over a simulated
multi-day mission, progressive biofouling drift detunes the resonators
(omega_m, Q_m shift). We compare:
  (i)  NO recalibration  -> accuracy decays as the device drifts away from the
       trained operating point;
  (ii) ONLINE recalibration -> at periodic surface windows, a few PAT steps on a
       small labelled batch (the reference-signal data of §8.2) re-tune the
       physical parameters to the CURRENT drifted device, recovering accuracy.

This tests the OPERATIONAL question §8 raises: can a brief surface-window PAT
update hold task performance across weeks of drift? (Function only; the 60-s
window budget is a separate energy/timing argument.)

Outputs: phase1b_results.json, phase1b_figure.png
Run:  py phase1b_drift_recal.py
"""
from __future__ import annotations
import json, time
from dataclasses import replace
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import Config, make_dataset, PNNClassifier, accuracy


def recalibrate(model, Xcal, ycal, noise_rng, steps=40, lr=3e-3):
    """A short PAT recalibration on the CURRENT (drifted) device: re-tune the
    physical params + readout to the present hardware state using a small batch."""
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(model.forward_pat(Xcal, noise_rng=noise_rng), ycal).backward()
        opt.step()


def run():
    t0 = time.time()
    # moderate task so baseline accuracy is high and degradation is clearly visible
    cfg = Config(n_classes=4, snr_db=10.0, class_overlap=0.0, n_modes=32,
                 n_train=1500, n_test=600, epochs=50,
                 drift_omega=0.006, drift_Q=0.004)   # per-day biofouling drift
    days = 30
    recal_every = 5          # surface window cadence (days)
    n_cal = 120              # labelled tuples available in a surface window
    seeds = list(range(8))   # 8 device instances x seeds (was 3; reviewer W6)

    no_recal_curves, recal_curves, recal_days = [], [], []
    for s in seeds:
        cfg_s = replace(cfg, seed=s)
        np_rng = np.random.default_rng(s)
        Xtr, ytr = make_dataset(cfg_s, cfg_s.n_train, np_rng)
        Xte, yte = make_dataset(cfg_s, cfg_s.n_test, np_rng)
        Xcal, ycal = make_dataset(cfg_s, n_cal, np_rng)   # surface-window reference data

        # ---- train at mission start (day 0) ----
        torch.manual_seed(s)
        nrng = torch.Generator().manual_seed(s + 100)
        model = PNNClassifier(cfg_s, freeze_physics=False)
        model.array.set_device_instance(nrng)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg_s.lr)
        for _ in range(cfg_s.epochs):
            perm = torch.randperm(cfg_s.n_train)
            for i in range(0, cfg_s.n_train, cfg_s.batch):
                idx = perm[i:i + cfg_s.batch]
                opt.zero_grad()
                F.cross_entropy(model.forward_pat(Xtr[idx], noise_rng=nrng), ytr[idx]).backward()
                opt.step()

        # snapshot the trained state to run two missions (no-recal vs recal) on
        # identical drift trajectories
        import copy
        model_norecal = copy.deepcopy(model)
        model_recal = copy.deepcopy(model)
        nrng_a = torch.Generator().manual_seed(s + 200)
        nrng_b = torch.Generator().manual_seed(s + 200)

        acc_nr, acc_rc = [], []
        for d in range(days + 1):
            if d > 0:
                model_norecal.array.apply_drift(1.0)   # same drift each day
                model_recal.array.apply_drift(1.0)
            a_nr = accuracy(model_norecal.forward_eval(Xte, noise_rng=nrng_a), yte)
            # recalibration at surface windows
            if d > 0 and d % recal_every == 0:
                recalibrate(model_recal, Xcal, ycal, nrng_b, steps=40, lr=cfg_s.lr)
                if s == seeds[0]:
                    recal_days.append(d)
            a_rc = accuracy(model_recal.forward_eval(Xte, noise_rng=nrng_b), yte)
            acc_nr.append(a_nr); acc_rc.append(a_rc)
        no_recal_curves.append(acc_nr); recal_curves.append(acc_rc)
        print(f"  seed={s}: day0={acc_nr[0]:.3f}  "
              f"day30 no-recal={acc_nr[-1]:.3f}  day30 recal={acc_rc[-1]:.3f}")

    nr = np.array(no_recal_curves); rc = np.array(recal_curves)
    nr_m, nr_s = nr.mean(0), nr.std(0)
    rc_m, rc_s = rc.mean(0), rc.std(0)

    res = {
        "config": cfg.__dict__, "days": days, "recal_every": recal_every,
        "n_cal_tuples": n_cal, "recal_days": recal_days,
        "no_recal_mean": nr_m.tolist(), "no_recal_std": nr_s.tolist(),
        "recal_mean": rc_m.tolist(), "recal_std": rc_s.tolist(),
        "day0_acc": float(nr_m[0]),
        "day30_no_recal": float(nr_m[-1]), "day30_recal": float(rc_m[-1]),
        "recal_recovery": float(rc_m[-1] - nr_m[-1]),
        "wall_clock_s": time.time() - t0,
    }
    with open("phase1b_results.json", "w") as f:
        json.dump(res, f, indent=2)

    # figure
    x = np.arange(days + 1)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(x, nr_m, "-o", color="#cc3333", ms=3, label="No recalibration")
    ax.fill_between(x, nr_m - nr_s, nr_m + nr_s, color="#cc3333", alpha=0.15)
    ax.plot(x, rc_m, "-o", color="#2c6fbb", ms=3, label="Online PAT recalibration (every 5 d)")
    ax.fill_between(x, rc_m - rc_s, rc_m + rc_s, color="#2c6fbb", alpha=0.15)
    for d in recal_days:
        ax.axvline(d, color="#2c6fbb", ls=":", lw=0.6, alpha=0.5)
    ax.set_xlabel("Mission day (progressive biofouling drift)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Drift vs online PAT recalibration (Subsystem A)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.2); ax.set_ylim(0, 1.0)
    fig.tight_layout(); fig.savefig("phase1b_figure.png", dpi=200)

    print("\n===== PHASE 1B SUMMARY =====")
    print(f"day 0 accuracy ............ {nr_m[0]:.3f}")
    print(f"day 30 no recalibration ... {nr_m[-1]:.3f}  (drop {nr_m[0]-nr_m[-1]:+.3f})")
    print(f"day 30 with recalibration . {rc_m[-1]:.3f}  (recovery {rc_m[-1]-nr_m[-1]:+.3f})")
    print(f"Wall clock: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    run()
