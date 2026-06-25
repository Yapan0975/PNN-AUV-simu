"""
spice_iara_eval.py -- plug the CIRCUIT-LEVEL (ngspice) reservoir transfer matrix
into the real IARA leakage-free classifier and compare it head-to-head with the
abstract closed-form reservoir the paper used. Same recording-level splits, same
15 seeds, same LayerNorm+Linear readout (frozen physics / reservoir computing).

Three front ends, identical everywhere else:
  ideal_shape   -- closed-form |H|^2 (the paper's abstract reservoir), no noise
  spice_shape   -- ngspice-measured |H|^2 of the Tow-Thomas bank,        no noise
  spice_noisy   -- ngspice bank + the model's thermal floor + 8-bit ADC quantise

If ideal ~= spice the abstract reservoir is a FAITHFUL stand-in for the real
filter SHAPES (validates the classification claim); the correction the circuit
forces is then on ENERGY (246 mW budgeted vs 13 mW guessed), not accuracy.

Outputs: spice_iara_eval_results.json, spice_iara_eval_figure.png
Run:  py spice_iara_eval.py    (~3-6 min CPU)
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PN = ROOT.parent
sys.path.insert(0, str(PN))
import core
import iara_dataset as iara
from phase0_iara_stats import (rec_predict, balanced_acc, macro_f1,
                               cluster_bootstrap_ci)
from scipy import stats as sstats

DATA = PN.parent / "IARA-data"
CLASSES = ("Background", "Cargo", "Tanker", "Tug", "Special Craft")
SEEDS = list(range(15))
PSD_CACHE = PN / "phase0_iara_features.npz"

F_LO, F_HI, N_FREQ, N_MODES, Q_NOM = 10.0, 1000.0, 256, 32, 40.0
GRID = np.linspace(F_LO, F_HI, N_FREQ)
MODE_F0 = np.logspace(np.log10(F_LO * 1.2), np.log10(F_HI * 0.9), N_MODES)


def ideal_Hsq():
    """Closed-form |H|^2 used by core.PiezoResonatorArray._modal_energy, nominal
    (no fabrication tolerance), row-normalised -- the paper's abstract front end."""
    w = 2 * np.pi * GRID[None, :]
    wm = 2 * np.pi * MODE_F0[:, None]
    denom = (wm**2 - w**2)**2 + (w * wm / Q_NOM)**2
    Hsq = 1.0 / (denom + 1e-12)
    return Hsq / (Hsq.max(axis=1, keepdims=True) + 1e-30)


class Readout(nn.Module):
    def __init__(self, n_modes, k):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(n_modes), nn.Linear(n_modes, k))

    def forward(self, e):
        return self.net(e)


def features(X, Hsq_t, add_noise=False, adc_bits=8, thermal_floor=0.02,
             Qvec=None, rng=None):
    """E = X @ Hsq^T  (+ optional thermal noise scaled 1/sqrt(Q) + ADC), mirroring
    core.forward_features but with a swappable transfer matrix Hsq_t [modes,freq]."""
    E = X @ Hsq_t.t()
    if add_noise:
        nstd = thermal_floor * (E.mean() + 1e-6) * (1.0 / torch.sqrt(Qvec)).unsqueeze(0)
        E = E + nstd * (torch.randn(E.shape, generator=rng) if rng is not None
                        else torch.randn_like(E))
        levels = 2 ** adc_bits
        emax = E.amax(dim=1, keepdim=True) + 1e-9
        E = torch.round(E / emax * levels) / levels * emax
    return E


def train_eval(Xtr, ytr, Xte, yte, rec_te, Hsq_t, k, seed,
               add_noise=False, Qvec=None):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed + 100) if add_noise else None
    Etr = features(Xtr, Hsq_t, add_noise, Qvec=Qvec, rng=rng)
    m = Readout(Hsq_t.shape[0], k)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    n = len(Etr)
    for _ in range(60):
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            F.cross_entropy(m(Etr[idx]), ytr[idx]).backward()
            opt.step()
    with torch.no_grad():
        Ete = features(Xte, Hsq_t, add_noise, Qvec=Qvec, rng=rng)
        prob = torch.softmax(m(Ete), 1).numpy()
    return rec_predict(prob, yte, rec_te)


def subset(X, y, rec, kk):
    if kk == 2:
        return X, (y != 0).long(), rec
    keep = (y <= (kk - 1))
    return X[keep], y[keep], rec[keep.numpy()]


def main():
    t0 = time.time()
    d = np.load(ROOT / "spice_reservoir.npz")
    Hsq_spice = torch.from_numpy(d["Hsq_spice"].astype(np.float32))
    Qmeas = torch.from_numpy(np.nan_to_num(d["Q_meas"].astype(np.float32), nan=Q_NOM))
    Hsq_ideal = torch.from_numpy(ideal_Hsq().astype(np.float32))
    Qnom = torch.full((N_MODES,), Q_NOM)

    cfg0 = core.Config(seed=0, n_classes=5, n_modes=32)
    X, y, rec, classes = iara.load_iara(cfg0, DATA, str(DATA / "iara.xlsx"), CLASSES,
                                        seg_seconds=4.0, max_per_rec=12, recs_per_class=45,
                                        seed=0, cache=str(PSD_CACHE))
    print(f"[spice-eval] X{tuple(X.shape)} n_rec {len(np.unique(rec))}", flush=True)

    fronts = [("ideal_shape", Hsq_ideal, False, Qnom),
              ("spice_shape", Hsq_spice, False, Qnom),
              ("spice_noisy", Hsq_spice, True, Qmeas)]
    sweep = {}
    perseed5 = {}
    for kk in (2, 3, 4, 5):
        Xs, ys, rs = subset(X, y, rec, kk)
        sweep[kk] = {}
        for name, Hsq_t, noise, Qv in fronts:
            bals, f1s, ppt = [], [], []
            for s in SEEDS:
                tr, te = iara.grouped_split(ys, rs, test_frac=0.3, seed=s)
                p, t = train_eval(Xs[tr], ys[tr], Xs[te], ys[te], rs[te],
                                  Hsq_t, kk, s, add_noise=noise, Qvec=Qv)
                bals.append(balanced_acc(p, t, kk)); f1s.append(macro_f1(p, t, kk))
                ppt.append((p, t))
            if kk == 5:
                perseed5[name] = list(bals)
            lo, hi = cluster_bootstrap_ci(ppt, kk)
            sweep[kk][name] = {"bal_mean": round(float(np.mean(bals)), 3),
                               "bal_std": round(float(np.std(bals)), 3),
                               "macro_f1_mean": round(float(np.mean(f1s)), 3),
                               "bal_ci95": [lo, hi]}
        print(f"  {kk}-class: " + " | ".join(
            f"{n} {sweep[kk][n]['bal_mean']:.3f}[{sweep[kk][n]['bal_ci95'][0]:.2f},{sweep[kk][n]['bal_ci95'][1]:.2f}]"
            for n, *_ in fronts), flush=True)

    # head-to-head delta at 5-class
    i5, s5, sn5 = (sweep[5]["ideal_shape"]["bal_mean"], sweep[5]["spice_shape"]["bal_mean"],
                  sweep[5]["spice_noisy"]["bal_mean"])

    # PAIRED equivalence test: circuit (spice_shape) vs abstract (ideal_shape), 5-class.
    # Same seeds/splits/readout init -> the per-seed accuracies are paired; only Hsq differs.
    di = np.array(perseed5["ideal_shape"]); ds = np.array(perseed5["spice_shape"])
    diff = ds - di
    nseed = len(diff); sd = float(diff.std(ddof=1))
    mean_d = float(diff.mean()); sem = sd / np.sqrt(nseed) if sd > 0 else 0.0
    tci = sstats.t.interval(0.95, nseed - 1, loc=mean_d, scale=sem) if sem > 0 else (mean_d, mean_d)
    mde = float(sstats.t.ppf(0.975, nseed - 1) * sem) if sem > 0 else 0.0
    margin = 0.02   # +-0.02 balanced-accuracy equivalence margin (pre-stated)
    if sem > 0:
        p_lo = 1 - sstats.t.cdf((mean_d + margin) / sem, nseed - 1)   # H0: diff <= -margin
        p_hi = sstats.t.cdf((mean_d - margin) / sem, nseed - 1)       # H0: diff >= +margin
        tost_p = float(max(p_lo, p_hi)); equiv = bool(tost_p < 0.05)
    else:
        tost_p = 0.0; equiv = True
    equivalence = {
        "n_seeds": nseed, "mean_diff_spice_minus_ideal": round(mean_d, 4),
        "paired_ci95": [round(float(tci[0]), 4), round(float(tci[1]), 4)],
        "min_detectable_diff": round(mde, 4),
        "tost_margin": margin, "tost_p": round(tost_p, 4),
        "equivalent_within_margin": equiv,
        "verdict": (f"Circuit and abstract front ends are statistically EQUIVALENT within +-{margin} "
                    f"balanced accuracy (TOST p={tost_p:.3f}); paired diff {mean_d:+.4f} "
                    f"(95% CI [{tci[0]:.4f},{tci[1]:.4f}], min detectable {mde:.4f})."
                    if equiv else
                    f"No difference detected (paired diff {mean_d:+.4f}, CI [{tci[0]:.4f},{tci[1]:.4f}]) "
                    f"but TOST does NOT establish equivalence within +-{margin} (p={tost_p:.3f}); "
                    f"the test resolves differences of {mde:.4f}.")}
    print(f"[spice-eval] equivalence: diff {mean_d:+.4f} CI{equivalence['paired_ci95']} "
          f"MDE {mde:.4f} TOST_p {tost_p:.3f} equiv={equiv}", flush=True)
    res = {
        "what": "Circuit-level (ngspice) reservoir vs abstract closed-form reservoir on real IARA",
        "dataset": "IARA (Zenodo 10.5281/zenodo.15758636), recording-level 30% test, 15 seeds",
        "front_ends": {
            "ideal_shape": "paper's closed-form |H|^2, no noise",
            "spice_shape": "ngspice Tow-Thomas bank |H|^2, no noise",
            "spice_noisy": "ngspice bank + thermal floor + 8-bit ADC"},
        "granularity_sweep": {str(k): v for k, v in sweep.items()},
        "delta_5class_spice_minus_ideal": round(s5 - i5, 3),
        "delta_5class_spicenoisy_minus_ideal": round(sn5 - i5, 3),
        "equivalence_5class": equivalence,
        "analog_frontend_mW_budgeted": float(d["analog_frontend_mW"]),
        "verdict": (f"At 5-class the ngspice-circuit reservoir scores {s5:.3f} (noiseless) / "
                    f"{sn5:.3f} (with device noise+ADC) vs the abstract reservoir's {i5:.3f}: "
                    f"a {s5-i5:+.3f}/{sn5-i5:+.3f} change. The abstract model is "
                    f"{'a faithful stand-in for' if abs(s5-i5)<0.03 else 'optimistic about'} "
                    f"the real filter shapes; the circuit forces its correction mainly on ENERGY "
                    f"({float(d['analog_frontend_mW']):.0f} mW budgeted vs 13 mW guessed)."),
        "wall_clock_s": round(time.time() - t0, 1),
    }
    (ROOT / "spice_iara_eval_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    _figure(sweep, fronts)
    print("\n[spice-eval] " + res["verdict"])
    print(f"[spice-eval] done {time.time()-t0:.0f}s")


def _figure(sweep, fronts):
    ks = [2, 3, 4, 5]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    colors = {"ideal_shape": "#2c6fbb", "spice_shape": "#cc3333", "spice_noisy": "#e08a1e"}
    labels = {"ideal_shape": "abstract |H|² (paper)", "spice_shape": "ngspice circuit |H|²",
              "spice_noisy": "ngspice + noise + ADC"}
    for name, *_ in fronts:
        ys = [sweep[k][name]["bal_mean"] for k in ks]
        lo = [sweep[k][name]["bal_ci95"][0] for k in ks]
        hi = [sweep[k][name]["bal_ci95"][1] for k in ks]
        ax.plot(ks, ys, "-o", color=colors[name], label=labels[name], ms=5)
        ax.fill_between(ks, lo, hi, color=colors[name], alpha=0.12)
    for k in ks:
        ax.axhline(1.0 / k, ls=":", color="grey", lw=0.7)
    ax.set_xticks(ks); ax.set_xlabel("number of classes")
    ax.set_ylabel("recording-level balanced accuracy")
    ax.set_title("Circuit-level vs abstract reservoir on IARA\n(dotted = chance)", fontsize=9)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(ROOT / "spice_iara_eval_figure.png", dpi=200)


if __name__ == "__main__":
    main()
