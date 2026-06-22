"""
phase1a_difficulty_sweep.py — Phase 1A: when does TRAINING the physics beat a
fixed random analog projection (reservoir / PRC regime)?

This is the experiment that turns the He & Musgrave differentiation from a
DESIGN-layer claim into a RESULT-layer claim. He & Musgrave use a fixed
reservoir (readout-only training); we use full PAT (train the physics). Phase 0
found that on an EASY task the two are indistinguishable (reservoir regime). Here
we sweep task difficulty and resonator count to locate where training the
physics actually earns its keep.

Hypothesis (falsifiable): with FEW resonators (small n_modes), where each
resonator must be placed on a discriminative band, PAT (trainable physics)
beats frozen-random physics; with MANY resonators, random placement covers the
band by luck and the gap closes (reservoir regime).

Primary axis : n_modes in {4, 8, 16, 32, 64}
Task         : hard variant (more classes, lower SNR, less data, class overlap)
Methods      : PAT (train physics) vs FROZEN random physics vs digital baseline
Seeds        : 4 (error bars)

Outputs: phase1a_results.json, phase1a_figure.png
Run:  py phase1a_difficulty_sweep.py
"""
from __future__ import annotations
import json, time
from dataclasses import replace
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import Config, make_dataset, train_pnn_pat, train_digital


def run():
    t0 = time.time()
    n_modes_list = [4, 8, 16, 32, 64]
    seeds = [0, 1, 2, 3]
    # HARD task: 6 classes, low SNR, little data, partial class overlap
    base = Config(n_classes=6, snr_db=3.0, class_overlap=0.35,
                  n_train=400, n_test=600, epochs=50)

    res = {"n_modes": n_modes_list, "config_base": base.__dict__,
           "pat": {}, "frozen": {}, "digital": {}}
    pat_mean, pat_std, frz_mean, frz_std = [], [], [], []
    dig_mean, dig_std = [], []

    for M in n_modes_list:
        pat_runs, frz_runs, dig_runs = [], [], []
        for s in seeds:
            cfg = replace(base, n_modes=M, seed=s)
            np_rng = np.random.default_rng(s)
            Xtr, ytr = make_dataset(cfg, cfg.n_train, np_rng)
            Xte, yte = make_dataset(cfg, cfg.n_test, np_rng)
            nrng = torch.Generator().manual_seed(s + 100)
            a_pat = train_pnn_pat(cfg, Xtr, ytr, Xte, yte, nrng, freeze=False)
            nrng = torch.Generator().manual_seed(s + 100)
            a_frz = train_pnn_pat(cfg, Xtr, ytr, Xte, yte, nrng, freeze=True)
            a_dig = train_digital(cfg, Xtr, ytr, Xte, yte)
            pat_runs.append(a_pat); frz_runs.append(a_frz); dig_runs.append(a_dig)
            print(f"  n_modes={M:3d} seed={s}: PAT={a_pat:.3f} frozen={a_frz:.3f} "
                  f"digital={a_dig:.3f}  gap(PAT-frozen)={a_pat-a_frz:+.3f}")
        res["pat"][M] = pat_runs; res["frozen"][M] = frz_runs; res["digital"][M] = dig_runs
        pat_mean.append(float(np.mean(pat_runs))); pat_std.append(float(np.std(pat_runs)))
        frz_mean.append(float(np.mean(frz_runs))); frz_std.append(float(np.std(frz_runs)))
        dig_mean.append(float(np.mean(dig_runs))); dig_std.append(float(np.std(dig_runs)))
        print(f"  >> n_modes={M:3d}: PAT={pat_mean[-1]:.3f}±{pat_std[-1]:.3f} "
              f"frozen={frz_mean[-1]:.3f}±{frz_std[-1]:.3f} "
              f"GAP={pat_mean[-1]-frz_mean[-1]:+.3f}")

    gap = [p - f for p, f in zip(pat_mean, frz_mean)]
    res["summary"] = {
        "pat_mean": pat_mean, "pat_std": pat_std,
        "frozen_mean": frz_mean, "frozen_std": frz_std,
        "digital_mean": dig_mean, "digital_std": dig_std,
        "gap_pat_minus_frozen": gap,
        "max_gap": float(max(gap)), "max_gap_at_n_modes": n_modes_list[int(np.argmax(gap))],
        "wall_clock_s": time.time() - t0,
    }
    with open("phase1a_results.json", "w") as f:
        json.dump(res, f, indent=2)

    # figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    x = np.array(n_modes_list)
    ax1.errorbar(x, pat_mean, yerr=pat_std, marker="o", color="#2c6fbb",
                 label="PAT (train physics)", capsize=3)
    ax1.errorbar(x, frz_mean, yerr=frz_std, marker="s", color="#cc5500",
                 label="Frozen random physics (PRC)", capsize=3, ls="--")
    ax1.errorbar(x, dig_mean, yerr=dig_std, marker="^", color="#888888",
                 label="Digital baseline", capsize=3, ls=":")
    ax1.set_xscale("log", base=2); ax1.set_xticks(x); ax1.set_xticklabels(x)
    ax1.set_xlabel("# resonators (n_modes)"); ax1.set_ylabel("Test accuracy")
    ax1.set_title(f"Hard task ({base.n_classes}-class, SNR={base.snr_db}dB, "
                  f"N={base.n_train})", fontsize=9)
    ax1.legend(fontsize=7); ax1.grid(alpha=0.2)

    ax2.bar([str(m) for m in n_modes_list], gap,
            color=["#2c6fbb" if g > 0.02 else "#bbbbbb" for g in gap])
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xlabel("# resonators (n_modes)")
    ax2.set_ylabel("Accuracy gap: PAT − frozen")
    ax2.set_title("When does training the physics help?", fontsize=9)
    for i, g in enumerate(gap):
        ax2.text(i, g + (0.005 if g >= 0 else -0.015), f"{g:+.3f}",
                 ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig("phase1a_figure.png", dpi=200)

    print("\n===== PHASE 1A SUMMARY =====")
    print(f"Max PAT-minus-frozen gap = {max(gap):+.3f} at n_modes={n_modes_list[int(np.argmax(gap))]}")
    print(f"Gap at largest n_modes ({n_modes_list[-1]}) = {gap[-1]:+.3f}")
    print(f"PNN-PAT vs digital at n_modes=64: {pat_mean[-1]:.3f} vs {dig_mean[-1]:.3f} "
          f"({pat_mean[-1]-dig_mean[-1]:+.3f})")
    verdict = ("training-the-physics helps in the few-resonator regime"
               if max(gap) > 0.03 else
               "reservoir regime persists across all tested settings")
    print(f"Verdict: {verdict}")
    print(f"Wall clock: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    run()
