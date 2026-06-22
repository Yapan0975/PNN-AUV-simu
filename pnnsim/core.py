"""
core.py — shared device model, dataset, and training utilities for PNN-AUV sim.
Extracted from phase0_acoustic.py so Phase-1 experiments reuse the same faithful
noise-realistic piezoelectric-resonator PNN + dual-model PAT machinery.

Dataset now exposes DIFFICULTY knobs (n_classes, snr_db, class_overlap, n_train)
so phase1a can sweep task difficulty.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    seed: int = 0
    f_lo: float = 10.0
    f_hi: float = 1000.0
    n_freq: int = 256
    n_modes: int = 32
    Q_nominal: float = 40.0
    sigma_fab_omega: float = 0.03
    sigma_fab_Q: float = 0.08
    thermal_floor: float = 0.02
    adc_bits: int = 8
    drift_omega: float = 0.0
    drift_Q: float = 0.0
    # task difficulty
    n_classes: int = 4
    snr_db: float = 12.0          # lower = harder
    class_overlap: float = 0.0    # 0=separated f0s, 1=heavy overlap
    n_train: int = 2000
    n_test: int = 600
    # training
    epochs: int = 50
    batch: int = 128
    lr: float = 3e-3


# up to 8 ship-like classes (blade/shaft fundamentals, tilt, harmonic richness)
_CLASS_F0 = [7.5, 12.0, 22.0, 16.0, 9.5, 14.0, 19.0, 11.0]
_CLASS_TILT = [-1.0, -0.6, -1.4, -0.9, -1.1, -0.7, -1.3, -0.8]
_CLASS_NH = [8, 6, 4, 10, 7, 5, 9, 6]


def make_dataset(cfg: Config, n: int, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = np.linspace(cfg.f_lo, cfg.f_hi, cfg.n_freq)
    K = cfg.n_classes
    # class_overlap pulls the per-class fundamentals toward their mean
    f0_arr = np.array(_CLASS_F0[:K])
    f0_arr = f0_arr * (1 - cfg.class_overlap) + f0_arr.mean() * cfg.class_overlap
    sig_amp = 1.0
    noise_amp = sig_amp / (10 ** (cfg.snr_db / 20.0))   # SNR -> noise scale
    X = np.zeros((n, cfg.n_freq), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        c = int(rng.integers(K))
        y[i] = c
        f0 = f0_arr[c] * (1.0 + 0.10 * rng.standard_normal())
        spec = np.zeros(cfg.n_freq)
        for h in range(1, _CLASS_NH[c % len(_CLASS_NH)] + 1):
            fh = f0 * h
            if fh < cfg.f_hi:
                amp = (1.0 / h) * (1.0 + 0.3 * rng.standard_normal())
                width = 2.0 + 0.5 * h
                spec += np.abs(amp) * np.exp(-0.5 * ((freqs - fh) / width) ** 2)
        broadband = (freqs / cfg.f_lo) ** _CLASS_TILT[c % len(_CLASS_TILT)]
        spec += 0.4 * broadband / broadband.max()
        spec *= sig_amp
        # additive noise scaled by SNR + multiplicative propagation jitter
        spec *= (1.0 + 0.15 * rng.standard_normal(cfg.n_freq)).clip(0.1, None)
        spec += noise_amp * rng.random(cfg.n_freq)
        spec = spec / (spec.sum() + 1e-9)
        X[i] = spec
    return torch.from_numpy(X), torch.from_numpy(y)


class PiezoResonatorArray(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        freqs = torch.linspace(cfg.f_lo, cfg.f_hi, cfg.n_freq)
        self.register_buffer("w", 2 * math.pi * freqs)
        wm = 2 * math.pi * torch.logspace(
            math.log10(cfg.f_lo * 1.2), math.log10(cfg.f_hi * 0.9), cfg.n_modes)
        self.register_buffer("wm_nominal", wm)
        self.register_buffer("Q_nominal_b", torch.full((cfg.n_modes,), cfg.Q_nominal))
        self.d_omega = nn.Parameter(torch.zeros(cfg.n_modes))
        self.d_lnQ = nn.Parameter(torch.zeros(cfg.n_modes))
        self.register_buffer("fab_omega", torch.ones(cfg.n_modes))
        self.register_buffer("fab_Q", torch.ones(cfg.n_modes))

    def set_device_instance(self, rng: torch.Generator, drift_step: float = 0.0):
        cfg = self.cfg
        self.fab_omega = (1.0 + cfg.sigma_fab_omega * torch.randn(cfg.n_modes, generator=rng)) \
            * (1.0 - cfg.drift_omega * drift_step)
        self.fab_Q = (1.0 + cfg.sigma_fab_Q * torch.randn(cfg.n_modes, generator=rng)) \
            * (1.0 - cfg.drift_Q * drift_step)

    def apply_drift(self, drift_step: float):
        """Progressive biofouling drift on the CURRENT instance (Phase 1b)."""
        self.fab_omega = self.fab_omega * (1.0 - self.cfg.drift_omega * drift_step)
        self.fab_Q = self.fab_Q * (1.0 - self.cfg.drift_Q * drift_step)

    def _modal_energy(self, S, wm, Q):
        w = self.w
        wm = wm.unsqueeze(1); Q = Q.unsqueeze(1)
        denom = (wm**2 - w**2)**2 + (w * wm / Q)**2
        Hsq = 1.0 / (denom + 1e-12)
        Hsq = Hsq / Hsq.amax(dim=1, keepdim=True)
        return S @ Hsq.t()

    def _params(self, truth):
        wm = self.wm_nominal * (1.0 + 0.15 * torch.tanh(self.d_omega))
        Q = self.Q_nominal_b * torch.exp(0.5 * torch.tanh(self.d_lnQ))
        if truth:
            wm = wm * self.fab_omega; Q = Q * self.fab_Q
        return wm, Q

    def forward_features(self, S, truth, noise_rng=None):
        wm, Q = self._params(truth=truth)
        E = self._modal_energy(S, wm, Q)
        if truth:
            cfg = self.cfg
            nstd = cfg.thermal_floor * (E.mean() + 1e-6) * (1.0 / torch.sqrt(Q)).unsqueeze(0)
            E = E + nstd * (torch.randn(E.shape, generator=noise_rng) if noise_rng is not None
                            else torch.randn_like(E))
            levels = 2 ** cfg.adc_bits
            emax = E.amax(dim=1, keepdim=True) + 1e-9
            E = torch.round(E / emax * levels) / levels * emax
        return E


class PNNClassifier(nn.Module):
    def __init__(self, cfg: Config, freeze_physics: bool = False):
        super().__init__()
        self.cfg = cfg
        self.freeze = freeze_physics
        self.array = PiezoResonatorArray(cfg)
        if freeze_physics:
            g = torch.Generator().manual_seed(cfg.seed + 7)
            with torch.no_grad():
                self.array.d_omega.copy_(0.8 * torch.randn(cfg.n_modes, generator=g))
                self.array.d_lnQ.copy_(0.8 * torch.randn(cfg.n_modes, generator=g))
            self.array.d_omega.requires_grad_(False)
            self.array.d_lnQ.requires_grad_(False)
        self.readout = nn.Sequential(nn.LayerNorm(cfg.n_modes),
                                     nn.Linear(cfg.n_modes, cfg.n_classes))

    def forward_pat(self, S, noise_rng=None):
        f_truth = self.array.forward_features(S, truth=True, noise_rng=noise_rng)
        f_sur = self.array.forward_features(S, truth=False)
        feat = f_sur + (f_truth - f_sur).detach()
        return self.readout(feat)

    def forward_clean(self, S):
        return self.readout(self.array.forward_features(S, truth=False))

    def forward_eval(self, S, noise_rng=None):
        with torch.no_grad():
            return self.readout(self.array.forward_features(S, truth=True, noise_rng=noise_rng))


class DigitalBaseline(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(cfg.n_freq), nn.Linear(cfg.n_freq, 64),
                                 nn.ReLU(), nn.Linear(64, cfg.n_classes))

    def forward(self, S):
        return self.net(S)


def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


def train_pnn_pat(cfg, Xtr, ytr, Xte, yte, noise_rng, freeze=False, return_model=False):
    torch.manual_seed(cfg.seed)
    model = PNNClassifier(cfg, freeze_physics=freeze)
    model.array.set_device_instance(noise_rng)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    n = Xtr.shape[0]
    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, cfg.batch):
            idx = perm[i:i + cfg.batch]
            opt.zero_grad()
            F.cross_entropy(model.forward_pat(Xtr[idx], noise_rng=noise_rng), ytr[idx]).backward()
            opt.step()
    acc = accuracy(model.forward_eval(Xte, noise_rng=noise_rng), yte)
    return (model, acc) if return_model else acc


def train_digital(cfg, Xtr, ytr, Xte, yte):
    torch.manual_seed(cfg.seed)
    model = DigitalBaseline(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n = Xtr.shape[0]
    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, cfg.batch):
            idx = perm[i:i + cfg.batch]
            opt.zero_grad()
            F.cross_entropy(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
    return accuracy(model(Xte), yte)


# ============================================================================
# REAL-DATA INGESTION (DeepShip / ShipsEar and any WAV corpus)
# ----------------------------------------------------------------------------
# Reads real .wav recordings laid out as  <root>/<class_name>/*.wav  (the layout
# both DeepShip and ShipsEar use after extraction), computes a per-segment power
# spectrum over the SAME resonator band [f_lo, f_hi] the PNN sees, and returns
# (X, y) tensors in EXACTLY the format make_dataset produces -- so the Phase 0/1
# PNN / PAT / digital code runs unchanged. Synthetic data is only a fallback.
# Depends on scipy (welch + wavfile); no soundfile/librosa needed.
# ============================================================================
def _segment_spectrum(seg: np.ndarray, sr: float, cfg: Config) -> np.ndarray:
    """Welch PSD of one segment, interpolated onto the resonator grid and
    sum-normalized -- identical output format to make_dataset rows."""
    from scipy.signal import welch
    nps = int(2 ** round(math.log2(max(256.0, sr / 5.0))))   # ~5 Hz resolution
    nps = max(256, min(nps, len(seg)))
    f, pxx = welch(seg, fs=sr, nperseg=nps)
    grid = np.linspace(cfg.f_lo, cfg.f_hi, cfg.n_freq)
    p = np.interp(grid, f, pxx, left=pxx[0], right=pxx[-1])
    p = np.clip(p, 0.0, None)
    return (p / (p.sum() + 1e-9)).astype(np.float32)


def load_real_dataset(cfg: Config, root, seg_seconds: float = 2.0,
                      max_per_file: int = 12, n: int = None,
                      rng: np.random.Generator = None, verbose: bool = True
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ingest a real WAV corpus <root>/<class>/*.wav into (X, y) tensors that
    match make_dataset's format. Classes are the sorted subdirectory names
    (first cfg.n_classes used). Each recording is split into seg_seconds windows.
    """
    from pathlib import Path
    from scipy.io import wavfile
    root = Path(root)
    classdirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not classdirs:
        raise FileNotFoundError(f"no class subdirectories under {root}")
    classdirs = classdirs[:cfg.n_classes]
    X, y = [], []
    per_class = {}
    for ci, cd in enumerate(classdirs):
        cnt = 0
        for wf in sorted(cd.glob("*.wav")):
            try:
                sr, data = wavfile.read(str(wf))
            except Exception:
                continue
            data = np.asarray(data)
            if data.ndim > 1:
                data = data[:, 0]
            data = data.astype(np.float64)
            data = data / (np.max(np.abs(data)) + 1e-9)     # amplitude-normalize
            seglen = int(seg_seconds * sr)
            if seglen < 256 or len(data) < seglen:
                continue
            taken = 0
            for k in range(len(data) // seglen):
                if taken >= max_per_file:
                    break
                seg = data[k * seglen:(k + 1) * seglen]
                if np.max(np.abs(seg)) < 1e-4:
                    continue
                X.append(_segment_spectrum(seg, sr, cfg))
                y.append(ci)
                taken += 1
                cnt += 1
        per_class[cd.name] = cnt
    if not X:
        raise RuntimeError(f"no usable WAV segments found under {root}")
    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    if n is not None and n < len(X):
        rng = rng or np.random.default_rng(cfg.seed)
        idx = rng.choice(len(X), size=n, replace=False)
        X, y = X[idx], y[idx]
    if verbose:
        print(f"[load_real_dataset] {root}: {len(X)} segments, "
              f"{len(classdirs)} classes -> {per_class}")
    return torch.from_numpy(X), torch.from_numpy(y)


def make_dataset_auto(cfg: Config, n: int, rng: np.random.Generator,
                      real_root=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Use the real WAV corpus at real_root if it exists, else synthetic.
    This is the single switch that turns Phase 0/1 from synthetic to real."""
    if real_root is not None:
        from pathlib import Path
        if Path(real_root).exists():
            return load_real_dataset(cfg, real_root, n=n, rng=rng)
        print(f"[make_dataset_auto] real_root '{real_root}' not found -> synthetic fallback")
    return make_dataset(cfg, n, rng)
