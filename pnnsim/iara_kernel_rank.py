"""
iara_kernel_rank.py -- reservoir-computing diagnostic (round-5 review R3-W1):
substantiate the word "reservoir" with a kernel-rank / memory characterisation of
the piezoelectric-resonator forward model, using REAL IARA band spectra as drive.

Two numbers the reservoir-computing literature uses:
  * KERNEL RANK (Legendre/Legenstein-Maass; Dambre et al. 2012): effective rank of
    the device's feature matrix over DIVERSE inputs -- how rich its instantaneous
    nonlinear mixing is (a good static kernel uses many of its modes).
  * GENERALISATION / noise rank: effective rank of the feature response to
    near-DUPLICATE (noise-perturbed) inputs -- low is good (the device does not
    amplify input noise into spurious dimensions). Kernel >> noise rank => useful
    static kernel.
And the structural fact about TEMPORAL MEMORY: the readout is the time-AVERAGED
modal ENERGY of a power spectrum, so phase/within-window timing is discarded =>
fading-memory / linear memory capacity ~ 0 by construction (the mechanistic reason
training the physics adds nothing and the device is modulation-blind).

Run:  py iara_kernel_rank.py  ->  iara_kernel_rank_results.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import core

ROOT = Path(__file__).resolve().parent
FEAT = ROOT / "phase0_iara_features.npz"          # 256-d band spectra (resonator input S)


def eff_rank(M):
    """Effective rank via participation ratio of singular values: (sum s)^2 / sum s^2."""
    s = np.linalg.svd(np.asarray(M, dtype=np.float64), compute_uv=False)
    s = s[s > 1e-12]
    pr = (s.sum() ** 2) / (np.square(s).sum() + 1e-12)
    nrank = int((s > 0.01 * s.max()).sum())          # count > 1% of top singular value
    return round(float(pr), 2), nrank


def main():
    d = np.load(FEAT, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)          # (N, 256) real IARA band powers
    cfg = core.Config()                               # n_freq=256, n_modes=32, f_lo=10, f_hi=1000
    arr = core.PiezoResonatorArray(cfg)
    arr.set_device_instance(torch.Generator().manual_seed(0), drift_step=0.0)   # fix a device instance

    # take a diverse subset of real spectra (deduplicate-ish by sampling)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(600, len(X)), replace=False)
    S = torch.from_numpy(X[idx])                      # (M, 256)
    with torch.no_grad():
        F = arr.forward_features(S, truth=False).numpy()     # (M, 32) reservoir features
    # centre before rank (remove the trivial mean dimension)
    Fc = F - F.mean(0, keepdims=True)
    pr_kernel, nr_kernel = eff_rank(Fc)

    # generalisation/noise rank: ONE input repeated with small input noise -> feature spread
    base = S[:1].repeat(F.shape[0], 1)
    noise = 0.02 * base.std() * torch.randn(base.shape, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        Fn = arr.forward_features((base + noise).clamp(min=0), truth=False).numpy()
    pr_noise, nr_noise = eff_rank(Fn - Fn.mean(0, keepdims=True))

    # raw-input effective rank for contrast (the 256-d drive)
    pr_in, nr_in = eff_rank(X[idx] - X[idx].mean(0, keepdims=True))

    out = {
        "device": {"n_freq": cfg.n_freq, "n_modes": cfg.n_modes,
                   "f_lo_hz": cfg.f_lo, "f_hi_hz": cfg.f_hi, "Q_nominal": cfg.Q_nominal},
        "n_inputs": int(F.shape[0]),
        "kernel_rank_participation": pr_kernel, "kernel_rank_count_1pct": nr_kernel,
        "noise_rank_participation": pr_noise, "noise_rank_count_1pct": nr_noise,
        "input_rank_participation": pr_in,
        "interpretation": (
            f"The {cfg.n_modes}-mode resonator bank keeps {nr_kernel} of {cfg.n_modes} modes "
            f"active over real IARA spectra (effective rank by 1%-of-top-singular-value = "
            f"{nr_kernel}; participation-ratio rank approximately {pr_kernel}, comparable to the "
            f"256-d input's approximately {pr_in}) -- a band-limited multi-mode nonlinear "
            f"projection rather than a rank-EXPANDING kernel. The decisive structural fact is "
            f"analytic: the readout is the time-averaged modal ENERGY of a power spectrum, so "
            f"within-window phase and timing are discarded and the linear/fading MEMORY CAPACITY "
            f"is ~0 by construction. This zero-memory, fixed-projection character is the "
            f"mechanistic root of two reported facts -- training the physics adds nothing over a "
            f"frozen reservoir (no recurrent state to tune; p=0.45), and the device is "
            f"modulation-blind (it cannot recover the envelope/DEMON shaft-rate lines that "
            f"discriminate vessel type). The noise-rank field is reported for completeness only; "
            f"with the deterministic surrogate map it reflects the projection-matrix rank, not a "
            f"generalisation property, so we do not interpret it."),
    }
    (ROOT / "iara_kernel_rank_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "interpretation"}, indent=2))
    print("\n" + out["interpretation"])
    print("  -> iara_kernel_rank_results.json")


if __name__ == "__main__":
    main()
