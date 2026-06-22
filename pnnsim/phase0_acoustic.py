"""
phase0_acoustic.py  —  PNN-AUV simulation, Phase 0
====================================================
Minimal but FAITHFUL closed loop for Subsystem A (acoustic target classifier),
implemented as a Physical Neural Network (PNN) trained by Physics-Aware Training
(PAT) under a *noise-realistic* piezoelectric-resonator device model.

This file validates the FUNCTION/LEARNING side of the architecture only. The
ENERGY claim is handled separately by a bottom-up device power model (see
EnergyModel) and is NEVER measured from GPU watts.

Two methodological principles are enforced (see paper §4):

  1. DUAL-MODEL PAT.  The physical layer has TWO models:
       - f_truth: the noisy device (thermal noise + fabrication variance +
                  drift + ADC quantization). Stands in for real hardware.
       - f_surrogate (f_m): a clean, differentiable nominal model used for the
                  PAT backward pass.
     PAT forward uses f_truth; PAT backward uses f_surrogate. The deliberate
     mismatch between them is exactly what PAT must tolerate. Implemented with
     the straight-through estimator:  y = y_sur + (y_truth - y_sur).detach()
     so forward value == truth, gradient flows through the surrogate.

  2. FUNCTION vs ENERGY are separate.  Classification accuracy comes from the
     sim; power comes from a device-anchored bottom-up model.

Honesty guardrails:
  - identity-physics ablation (replace resonator physics with identity) shows
    whether the physics is doing real computational work (cf. Wright 2022).
  - PAT vs in-silico-on-clean-model comparison shows PAT's value under mismatch.
  - everything seeded; results reported as measured, with no massaging.

Run:  py phase0_acoustic.py
Outputs: phase0_results.json, phase0_figure.png
Synthetic ship-like acoustic data is used for Phase 0 so the script is
self-contained; the DataConfig interface is built so DeepShip/ShipsEar drop in
at Phase 1 (replace make_dataset()).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===========================================================================
# Configuration
# ===========================================================================
@dataclass
class Config:
    seed: int = 0
    # acoustic band (Hz) — low-frequency ship tonals, DeepShip-like
    f_lo: float = 10.0
    f_hi: float = 1000.0
    n_freq: int = 256          # frequency-grid resolution
    # device: piezoelectric resonator array
    n_modes: int = 32          # number of resonators (= analog feature dim)
    Q_nominal: float = 40.0    # nominal quality factor
    # noise-realistic device imperfections
    sigma_fab_omega: float = 0.03   # fabrication variance on resonant freq (3%)
    sigma_fab_Q: float = 0.08       # fabrication variance on Q (8%)
    thermal_floor: float = 0.02     # thermal-mechanical noise floor (rel.)
    adc_bits: int = 8               # low-rate readout ADC bit depth
    drift_omega: float = 0.0        # biofouling drift on omega (set >0 to inject)
    drift_Q: float = 0.0            # biofouling drift on Q
    # task
    n_classes: int = 4         # cargo / tanker / tug / passenger (DeepShip-like)
    n_train: int = 2000
    n_test: int = 600
    # training
    epochs: int = 60
    batch: int = 128
    lr: float = 3e-3
    device: str = "cpu"


# ===========================================================================
# Synthetic ship-acoustic dataset (Phase-0 stand-in for DeepShip/ShipsEar)
# Each class has a characteristic fundamental (shaft/blade rate) + harmonics
# + broadband cavitation shape, with per-sample randomization.  Returns POWER
# SPECTRA on the frequency grid (what a hydrophone front end would feed in).
# ===========================================================================
def make_dataset(cfg: Config, n: int, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = np.linspace(cfg.f_lo, cfg.f_hi, cfg.n_freq)
    # class-characteristic fundamentals (Hz) and broadband tilt
    class_f0 = [7.5, 12.0, 22.0, 16.0]          # blade/shaft rates per class
    class_tilt = [-1.0, -0.6, -1.4, -0.9]       # broadband spectral slope
    class_nharm = [8, 6, 4, 10]                 # harmonic richness
    X = np.zeros((n, cfg.n_freq), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        c = int(rng.integers(cfg.n_classes))
        y[i] = c
        f0 = class_f0[c] * (1.0 + 0.10 * rng.standard_normal())  # speed jitter
        spec = np.zeros(cfg.n_freq)
        # harmonic tonals
        for h in range(1, class_nharm[c] + 1):
            fh = f0 * h
            if fh < cfg.f_hi:
                amp = (1.0 / h) * (1.0 + 0.3 * rng.standard_normal())
                width = 2.0 + 0.5 * h
                spec += np.abs(amp) * np.exp(-0.5 * ((freqs - fh) / width) ** 2)
        # broadband cavitation: power-law tilt
        broadband = (freqs / cfg.f_lo) ** class_tilt[c]
        spec += 0.4 * broadband / broadband.max()
        # propagation + sensor noise
        spec *= (1.0 + 0.15 * rng.standard_normal(cfg.n_freq)).clip(0.1, None)
        spec += 0.05 * rng.random(cfg.n_freq)   # noise floor
        # normalize to unit total power (per-sample gain invariance)
        spec = spec / (spec.sum() + 1e-9)
        X[i] = spec
    return torch.from_numpy(X), torch.from_numpy(y)


# ===========================================================================
# PNN device: piezoelectric resonator array (frequency-domain modal model)
#   Each resonator m: H_m(w) = 1 / (w_m^2 - w^2 + i*w*w_m/Q_m)
#   Readout feature  E_m = sum_w |H_m(w)|^2 * S(w)   (energy in mode m)
#   Trainable physical params theta: d_omega (bias shift of w_m), d_lnQ.
#   The readout matrix W is a small DIGITAL low-rate layer ("digitize only the
#   decision") — matches the architecture; included in both PNN paths.
# ===========================================================================
class PiezoResonatorArray(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        freqs = torch.linspace(cfg.f_lo, cfg.f_hi, cfg.n_freq)
        self.register_buffer("w", 2 * math.pi * freqs)                  # angular grid
        # nominal resonant freqs: log-spaced across band
        wm = 2 * math.pi * torch.logspace(
            math.log10(cfg.f_lo * 1.2), math.log10(cfg.f_hi * 0.9), cfg.n_modes)
        self.register_buffer("wm_nominal", wm)
        self.register_buffer("Q_nominal", torch.full((cfg.n_modes,), cfg.Q_nominal))
        # trainable PHYSICAL parameters (bias-induced shifts), bounded later
        self.d_omega = nn.Parameter(torch.zeros(cfg.n_modes))    # fractional shift of w_m
        self.d_lnQ = nn.Parameter(torch.zeros(cfg.n_modes))      # log-shift of Q
        # per-instance fabrication variance + drift (set by set_device_instance)
        self.register_buffer("fab_omega", torch.ones(cfg.n_modes))
        self.register_buffer("fab_Q", torch.ones(cfg.n_modes))

    def set_device_instance(self, rng: torch.Generator, drift_step: float = 0.0):
        """Draw a fresh physical device instance (fabrication variance) and apply
        optional biofouling drift. Called to instantiate 'the hardware'."""
        cfg = self.cfg
        self.fab_omega = (1.0 + cfg.sigma_fab_omega * torch.randn(
            cfg.n_modes, generator=rng)) * (1.0 - cfg.drift_omega * drift_step)
        self.fab_Q = (1.0 + cfg.sigma_fab_Q * torch.randn(
            cfg.n_modes, generator=rng)) * (1.0 - cfg.drift_Q * drift_step)

    def _modal_energy(self, S: torch.Tensor, wm: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        # |H_m(w)|^2 = 1 / ((w_m^2 - w^2)^2 + (w*w_m/Q)^2)
        w = self.w                                  # (F,)
        wm = wm.unsqueeze(1)                         # (M,1)
        Q = Q.unsqueeze(1)                          # (M,1)
        denom = (wm**2 - w**2)**2 + (w * wm / Q)**2  # (M,F)
        Hsq = 1.0 / (denom + 1e-12)                  # (M,F)
        Hsq = Hsq / Hsq.amax(dim=1, keepdim=True)    # normalize per mode (gain-invariant readout)
        # E_m = sum_w |H_m|^2 S(w);  S: (B,F) -> E: (B,M)
        E = S @ Hsq.t()
        return E

    def _params(self, truth: bool):
        cfg = self.cfg
        # bias shifts (bounded to +/-15% omega, +/-1 in lnQ for physical realism)
        wm = self.wm_nominal * (1.0 + 0.15 * torch.tanh(self.d_omega))
        Q = self.Q_nominal * torch.exp(0.5 * torch.tanh(self.d_lnQ))
        if truth:
            wm = wm * self.fab_omega
            Q = Q * self.fab_Q
        return wm, Q

    def forward_features(self, S: torch.Tensor, truth: bool,
                         noise_rng: torch.Generator = None) -> torch.Tensor:
        wm, Q = self._params(truth=truth)
        E = self._modal_energy(S, wm, Q)            # (B,M)
        if truth:
            cfg = self.cfg
            # thermal-mechanical noise: std grows as 1/sqrt(Q) + floor
            nstd = cfg.thermal_floor * (E.mean() + 1e-6) * (1.0 / torch.sqrt(Q)).unsqueeze(0)
            if noise_rng is not None:
                E = E + nstd * torch.randn(E.shape, generator=noise_rng)
            else:
                E = E + nstd * torch.randn_like(E)
            # low-rate ADC quantization on readout
            levels = 2 ** cfg.adc_bits
            emax = E.amax(dim=1, keepdim=True) + 1e-9
            E = torch.round(E / emax * levels) / levels * emax
        return E


class PNNClassifier(nn.Module):
    """Piezo resonator array (physical) + small digital readout.

    freeze_physics=True is the FAIR ablation (cf. Wright 2022 control): the
    resonator parameters are randomized and FROZEN (not trained), only the
    digital readout trains. Same feature dim (M) and same readout capacity as
    the full model, so the comparison isolates the value of *training the
    physics* vs using it as a fixed random analog projection (reservoir regime).
    """
    def __init__(self, cfg: Config, freeze_physics: bool = False):
        super().__init__()
        self.cfg = cfg
        self.freeze = freeze_physics
        self.array = PiezoResonatorArray(cfg)
        if freeze_physics:
            # randomize the physical params and freeze them (untrained analog projection)
            g = torch.Generator().manual_seed(cfg.seed + 7)
            with torch.no_grad():
                self.array.d_omega.copy_(0.8 * torch.randn(cfg.n_modes, generator=g))
                self.array.d_lnQ.copy_(0.8 * torch.randn(cfg.n_modes, generator=g))
            self.array.d_omega.requires_grad_(False)
            self.array.d_lnQ.requires_grad_(False)
        self.readout = nn.Sequential(
            nn.LayerNorm(cfg.n_modes),
            nn.Linear(cfg.n_modes, cfg.n_classes),
        )

    def features(self, S, truth, noise_rng=None):
        return self.array.forward_features(S, truth=truth, noise_rng=noise_rng)

    def forward_pat(self, S, noise_rng=None):
        """Dual-model PAT forward: value from truth device, gradient via surrogate."""
        f_truth = self.array.forward_features(S, truth=True, noise_rng=noise_rng)
        f_sur = self.array.forward_features(S, truth=False)
        feat = f_sur + (f_truth - f_sur).detach()   # PAT straight-through
        return self.readout(feat)

    def forward_clean(self, S):
        """In-silico forward: surrogate model only (no truth/noise)."""
        feat = self.array.forward_features(S, truth=False)
        return self.readout(feat)

    def forward_eval(self, S, noise_rng=None):
        """Deployment-time forward: real (truth) device, no grad."""
        with torch.no_grad():
            feat = self.array.forward_features(S, truth=True, noise_rng=noise_rng)
            return self.readout(feat)


# ===========================================================================
# Digital baseline: small MLP on the raw spectrum (fair param budget)
# ===========================================================================
class DigitalBaseline(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(cfg.n_freq),
            nn.Linear(cfg.n_freq, 64), nn.ReLU(),
            nn.Linear(64, cfg.n_classes),
        )

    def forward(self, S):
        return self.net(S)


# ===========================================================================
# Bottom-up, device-anchored ENERGY model (NOT measured from GPU)
# ===========================================================================
@dataclass
class EnergyModel:
    """Bottom-up, device-anchored power model. All terms are explicit
    assumptions; the dominant term (SoC platform overhead) drives the result
    and is swept in sensitivity().  This is NOT measured from GPU watts.

    Key architectural point (paper §3): the PNN front end runs continuously in
    the analog domain WITHOUT waking the power-hungry SoC; the digital baseline
    must run a wide-band ADC + continuous inference on an edge SoC whose
    idle+active platform power (1-3 W for Jetson-class) dominates everything.
    The PNN's win comes from AVOIDING that platform overhead, not from cheaper
    MACs (at the MAC level neither path is large).
    """
    cfg: Config
    walden_fom_j_per_convstep: float = 1e-12     # ~1 pJ/conv-step (Walden/Murmann floor)
    digital_mac_j: float = 1e-12                  # ~1 pJ/MAC on edge accelerator
    bias_static_w_per_mode: float = 3e-5          # 30 uW static bias per resonator (MEMS-class)
    sonar_adc_hz: float = 32000.0                # wide-band hydrophone sample rate (DeepShip = 32 kHz)
    frame_rate_hz: float = 100.0                 # CNN inference frame rate (continuous)
    cnn_macs_per_frame: float = 2.0e6            # small spectrogram CNN
    soc_overhead_w: float = 1.5                  # Jetson-class continuous platform power (idle+active)
    decision_rate_hz: float = 10.0              # low-rate readout for PNN path
    pnn_soc_duty: float = 0.01                   # PNN wakes SoC only 1% of the time (decisions)

    def pnn_power_w(self) -> float:
        c = self.cfg
        adc = c.n_modes * self.decision_rate_hz * (2 ** c.adc_bits) * self.walden_fom_j_per_convstep
        readout_macs = c.n_modes * c.n_classes * self.decision_rate_hz * self.digital_mac_j
        bias = c.n_modes * self.bias_static_w_per_mode
        soc = self.pnn_soc_duty * self.soc_overhead_w   # only occasional SoC wake
        return adc + readout_macs + bias + soc

    def digital_power_w(self) -> float:
        adc = self.sonar_adc_hz * (2 ** 16) * self.walden_fom_j_per_convstep
        macs = self.cnn_macs_per_frame * self.frame_rate_hz * self.digital_mac_j
        soc = self.soc_overhead_w                        # continuous SoC platform power (dominant)
        return adc + macs + soc

    def sensitivity(self) -> Dict[str, float]:
        """Power ratio under different SoC-overhead assumptions (the load-bearing one)."""
        out = {}
        base = self.soc_overhead_w
        for w in [0.5, 1.5, 3.0]:
            self.soc_overhead_w = w
            out[f"soc_{w}W"] = self.digital_power_w() / self.pnn_power_w()
        self.soc_overhead_w = base
        return out


# ===========================================================================
# Training / evaluation
# ===========================================================================
def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


def train_pnn_pat(cfg, Xtr, ytr, Xte, yte, noise_rng, freeze=False):
    torch.manual_seed(cfg.seed)
    model = PNNClassifier(cfg, freeze_physics=freeze)
    model.array.set_device_instance(noise_rng)       # instantiate "the hardware"
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    n = Xtr.shape[0]
    for ep in range(cfg.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, cfg.batch):
            idx = perm[i:i + cfg.batch]
            opt.zero_grad()
            logits = model.forward_pat(Xtr[idx], noise_rng=noise_rng)
            loss = F.cross_entropy(logits, ytr[idx])
            loss.backward()
            opt.step()
    # deployment-time eval uses the TRUTH device
    acc = accuracy(model.forward_eval(Xte, noise_rng=noise_rng), yte)
    return model, acc


def train_pnn_insilico(cfg, Xtr, ytr, Xte, yte, noise_rng):
    """Train on the CLEAN surrogate only (in-silico), deploy on noisy truth.
    Shows the model-reality gap PAT is designed to close."""
    torch.manual_seed(cfg.seed)
    model = PNNClassifier(cfg, freeze_physics=False)
    model.array.set_device_instance(noise_rng)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    n = Xtr.shape[0]
    for ep in range(cfg.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, cfg.batch):
            idx = perm[i:i + cfg.batch]
            opt.zero_grad()
            logits = model.forward_clean(Xtr[idx])   # clean surrogate forward
            loss = F.cross_entropy(logits, ytr[idx])
            loss.backward()
            opt.step()
    acc_clean = accuracy(model.forward_clean(Xte), yte)
    acc_deploy = accuracy(model.forward_eval(Xte, noise_rng=noise_rng), yte)
    return acc_clean, acc_deploy


def train_digital(cfg, Xtr, ytr, Xte, yte):
    torch.manual_seed(cfg.seed)
    model = DigitalBaseline(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n = Xtr.shape[0]
    for ep in range(cfg.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, cfg.batch):
            idx = perm[i:i + cfg.batch]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
    return accuracy(model(Xte), yte)


def pat_gradient_cosine(cfg, model, X, y, noise_rng):
    """Measure cos(PAT surrogate gradient, true device gradient) on theta.
    True gradient = autograd through the (noiseless) truth-param device."""
    model.zero_grad()
    logits = model.forward_pat(X, noise_rng=noise_rng)
    F.cross_entropy(logits, y).backward()
    g_pat = torch.cat([model.array.d_omega.grad.flatten(),
                       model.array.d_lnQ.grad.flatten()]).clone()
    # true gradient: forward through truth params (no noise), differentiable
    model.zero_grad()
    feat_true = model.array.forward_features(X, truth=False)  # nominal=surrogate is clean
    # build a 'truth-param' clean forward by temporarily using fab-varied params w/o noise
    wm, Q = model.array._params(truth=True)
    E = model.array._modal_energy(X, wm, Q)
    logits_true = model.readout(E)
    F.cross_entropy(logits_true, y).backward()
    g_true = torch.cat([model.array.d_omega.grad.flatten(),
                        model.array.d_lnQ.grad.flatten()]).clone()
    cos = F.cosine_similarity(g_pat.unsqueeze(0), g_true.unsqueeze(0)).item()
    return cos


# ===========================================================================
# Main experiment
# ===========================================================================
def main(real_root=None):
    t0 = time.time()
    cfg = Config()
    np_rng = np.random.default_rng(cfg.seed)
    noise_rng = torch.Generator().manual_seed(cfg.seed + 100)

    if real_root is not None:
        # REAL corpus path: ingest <root>/<class>/*.wav via the shared loader
        # (identical feature format), then stratified 70/30 train/test split.
        import core
        X, y = core.load_real_dataset(cfg, real_root)
        rng_split = np.random.default_rng(cfg.seed)
        tr_idx, te_idx = [], []
        yn = y.numpy()
        for c in range(int(yn.max()) + 1):
            ci = np.where(yn == c)[0]
            rng_split.shuffle(ci)
            k = max(1, int(0.7 * len(ci)))
            tr_idx += ci[:k].tolist(); te_idx += ci[k:].tolist()
        Xtr, ytr, Xte, yte = X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]
        print(f"[data] REAL corpus '{real_root}': {len(Xtr)} train / {len(Xte)} test segments")
    else:
        print("[data] generating synthetic ship-acoustic spectra ...")
        Xtr, ytr = make_dataset(cfg, cfg.n_train, np_rng)
        Xte, yte = make_dataset(cfg, cfg.n_test, np_rng)

    prefix = "phase0_real" if real_root is not None else "phase0"
    results: Dict[str, object] = {"config": asdict(cfg), "data_source": real_root or "synthetic"}

    print("[1/5] digital baseline (MLP on spectrum) ...")
    acc_digital = train_digital(cfg, Xtr, ytr, Xte, yte)
    print(f"      digital baseline acc = {acc_digital:.3f}")

    print("[2/5] PNN trained by PAT (noisy device fwd, surrogate bwd) ...")
    model_pat, acc_pat = train_pnn_pat(cfg, Xtr, ytr, Xte, yte, noise_rng)
    print(f"      PNN-PAT acc (truth device) = {acc_pat:.3f}")

    print("[3/5] FAIR ABLATION: frozen random physics (untrained analog projection) ...")
    _, acc_frozen = train_pnn_pat(cfg, Xtr, ytr, Xte, yte, noise_rng, freeze=True)
    print(f"      frozen-random-physics acc = {acc_frozen:.3f}")

    print("[4/5] in-silico training (clean fwd) then deploy on noisy device ...")
    acc_insilico_clean, acc_insilico_deploy = train_pnn_insilico(
        cfg, Xtr, ytr, Xte, yte, noise_rng)
    print(f"      in-silico acc (clean)  = {acc_insilico_clean:.3f}")
    print(f"      in-silico acc (deploy) = {acc_insilico_deploy:.3f}")

    print("[5/5] PAT gradient cosine + energy model ...")
    cos = pat_gradient_cosine(cfg, model_pat, Xte[:256], yte[:256], noise_rng)
    print(f"      cos(PAT grad, true device grad) = {cos:.3f}")

    em = EnergyModel(cfg)
    p_pnn = em.pnn_power_w()
    p_dig = em.digital_power_w()
    sens = em.sensitivity()
    print(f"      PNN power = {p_pnn*1e3:.2f} mW ; digital = {p_dig*1e3:.1f} mW ; "
          f"ratio = {p_dig/p_pnn:.0f}x ; sensitivity(SoC) = {sens}")

    results.update({
        "acc_digital_baseline": acc_digital,
        "acc_pnn_pat": acc_pat,
        "acc_frozen_random_physics_ablation": acc_frozen,
        "acc_insilico_clean": acc_insilico_clean,
        "acc_insilico_deploy_on_noisy": acc_insilico_deploy,
        "pat_minus_insilico_deploy": acc_pat - acc_insilico_deploy,
        "pnn_over_baseline_ratio": acc_pat / acc_digital,
        "physics_training_contribution_acc": acc_pat - acc_frozen,
        "pat_gradient_cosine": cos,
        "energy": {
            "pnn_power_mw": p_pnn * 1e3,
            "digital_power_mw": p_dig * 1e3,
            "power_ratio_digital_over_pnn": p_dig / p_pnn,
            "sensitivity_vs_soc_overhead": sens,
            "note": "device-anchored bottom-up model; dominant term is SoC platform "
                    "overhead the architecture avoids; NOT measured from GPU",
        },
        "wall_clock_s": time.time() - t0,
    })

    with open(f"{prefix}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] wrote {prefix}_results.json ({results['wall_clock_s']:.1f}s)")

    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4))
    names = ["Digital\nbaseline", "PNN-PAT\n(noisy dev.)", "Frozen rand.\nphysics\n(ablation)",
             "In-silico\n(deploy on\nnoisy dev.)"]
    accs = [acc_digital, acc_pat, acc_frozen, acc_insilico_deploy]
    colors = ["#888888", "#2c6fbb", "#cc5500", "#bbbbbb"]
    bars = ax1.bar(names, accs, color=colors)
    ax1.set_ylabel("Test accuracy")
    ax1.set_ylim(0, 1.0)
    chance = 1.0 / cfg.n_classes
    ax1.axhline(chance, ls=":", color="k", lw=0.8)
    ax1.text(0, chance + 0.02, f"chance ({cfg.n_classes}-class)", fontsize=7)
    for b, a in zip(bars, accs):
        ax1.text(b.get_x() + b.get_width()/2, a + 0.02, f"{a:.2f}",
                 ha="center", fontsize=8)
    ax1.set_title("Function (simulation)", fontsize=10)
    ax1.tick_params(axis="x", labelsize=7)

    # energy (log scale)
    ax2.bar(["PNN\n(device model)", "Digital\nbaseline"],
            [p_pnn*1e3, p_dig*1e3], color=["#2c6fbb", "#888888"])
    ax2.set_yscale("log")
    ax2.set_ylabel("Power (mW, bottom-up model)")
    ax2.set_title(f"Energy (device-anchored, {p_dig/p_pnn:.0f}x)", fontsize=10)
    for i, p in enumerate([p_pnn*1e3, p_dig*1e3]):
        ax2.text(i, p*1.2, f"{p:.2f} mW", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{prefix}_figure.png", dpi=200)
    print(f"[done] wrote {prefix}_figure.png")

    # ---- honest summary to stdout ----
    print("\n================ HONEST SUMMARY ================")
    print(f"Digital baseline accuracy ............. {acc_digital:.3f}")
    print(f"PNN-PAT accuracy (noisy device) ...... {acc_pat:.3f}  ({acc_pat/acc_digital*100:.0f}% of baseline)")
    print(f"Physics-TRAINING contribution ........ {acc_pat-acc_frozen:+.3f}  "
          f"(vs frozen random physics; {'training helps' if acc_pat-acc_frozen>0.02 else 'reservoir regime: training marginal'})")
    print(f"PAT vs in-silico-deploy gap .......... {acc_pat-acc_insilico_deploy:+.3f}  "
          f"({'PAT closes the reality gap' if acc_pat-acc_insilico_deploy>0.02 else 'gap small'})")
    print(f"PAT gradient cosine .................. {cos:.3f}  ({'>0: PAT viable' if cos>0 else 'NEGATIVE: PAT broken'})")
    print(f"Power ratio (digital/PNN) ............ {p_dig/p_pnn:.0f}x  [device model, not measured; SoC-overhead-dominated]")
    print("================================================")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", metavar="ROOT", default=None,
                    help="path to a real WAV corpus laid out as <ROOT>/<class>/*.wav "
                         "(e.g. an extracted ShipsEar/DeepShip). If omitted, synthetic data is used.")
    args = ap.parse_args()
    main(real_root=args.real)
