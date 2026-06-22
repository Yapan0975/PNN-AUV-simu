"""
test_real_pipeline.py -- end-to-end validation of the REAL-WAV ingestion path.

There is no licensing-free way to pull DeepShip/ShipsEar (gated) or the 31 GB
IARA corpus inside this sandbox, so this test proves the *pipeline* the real
data will flow through: it synthesizes physically grounded ship-radiated-noise
TIME SERIES, writes them as real int16 PCM .wav files, then ingests them with
the SAME core.load_real_dataset a user points at ShipsEar, and trains the PNN
on the result. If accuracy beats chance, the full real-WAV path
(file write -> wavfile.read -> Welch PSD -> resonator-band features -> PNN/PAT)
is verified. Pointing core.load_real_dataset at a real corpus is then a one-line
path change (see phase0_acoustic.py --real).

Run:  py test_real_pipeline.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter

import core

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "_realtest_wav"

SR = 8000                      # Hz, a real low-rate hydrophone-like sample rate
# four ship-like classes: (shaft/blade fundamental Hz, # harmonics, spectral tilt)
CLASSES = [
    ("classA_tug",     7.5, 8, -0.8),
    ("classB_cargo",  12.0, 6, -1.1),
    ("classC_tanker", 22.0, 4, -1.4),
    ("classD_passen", 16.0, 10, -0.9),
]


def synth_ship(path: Path, dur: float, f0: float, nh: int, tilt: float, seed: int):
    """Synthesize one ship-radiated-noise clip and write a real int16 WAV."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * dur)) / SR
    sig = np.zeros_like(t)
    for h in range(1, nh + 1):
        fh = f0 * h * (1.0 + 0.01 * rng.standard_normal())
        if fh < SR / 2:
            ph = rng.uniform(0, 2 * np.pi)
            am = 1.0 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.1, 0.5) * t + ph)
            sig += (1.0 / h) * am * np.sin(2 * np.pi * fh * t + ph)
    # broadband flow/cavitation noise, low-pass shaped, scaled by spectral tilt
    noise = rng.standard_normal(len(t))
    b, a = butter(2, 1500 / (SR / 2))
    noise = lfilter(b, a, noise)
    sig = sig + (10.0 ** tilt) * 3.0 * noise
    sig = sig / (np.max(np.abs(sig)) + 1e-9)
    pcm = (0.9 * sig * 32767).astype(np.int16)
    wavfile.write(str(path), SR, pcm)


def build_corpus(files_per_class: int = 12, dur: float = 6.0):
    if DATA.exists():
        shutil.rmtree(DATA)
    for ci, (name, f0, nh, tilt) in enumerate(CLASSES):
        d = DATA / name
        d.mkdir(parents=True)
        for k in range(files_per_class):
            synth_ship(d / f"rec_{k:02d}.wav", dur, f0, nh, tilt, seed=1000 * ci + k)
    print(f"[corpus] wrote {len(CLASSES) * files_per_class} real WAV files under {DATA}")


def main():
    build_corpus()

    cfg = core.Config(seed=0, n_classes=4, n_modes=16, epochs=40, lr=3e-3)

    # ingest the REAL WAV files through the production loader
    X, y = core.load_real_dataset(cfg, DATA, seg_seconds=1.5, max_per_file=4)
    n = X.shape[0]
    assert X.shape[1] == cfg.n_freq, "feature width must match resonator grid"
    assert set(y.tolist()) == set(range(cfg.n_classes)), "all classes present"

    # train/test split
    g = np.random.default_rng(0)
    perm = g.permutation(n)
    cut = int(0.7 * n)
    tr, te = perm[:cut], perm[cut:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    noise_rng = __import__("torch").Generator().manual_seed(0)
    acc_pat = core.train_pnn_pat(cfg, Xtr, ytr, Xte, yte, noise_rng)
    acc_dig = core.train_digital(cfg, Xtr, ytr, Xte, yte)
    chance = 1.0 / cfg.n_classes

    print("\n===== REAL-WAV PIPELINE TEST =====")
    print(f"segments ingested ......... {n}  ({cut} train / {n - cut} test)")
    print(f"feature width ............. {X.shape[1]} (resonator grid, sum-normalized)")
    print(f"chance level .............. {chance:.3f}")
    print(f"digital baseline acc ...... {acc_dig:.3f}")
    print(f"PNN-PAT (noisy device) acc  {acc_pat:.3f}")
    ok = acc_pat > chance + 0.15 and acc_dig > chance + 0.15
    print(f"PIPELINE VALID (> chance) . {ok}")
    print("\nThe full real-WAV path (write -> wavfile.read -> Welch PSD ->")
    print("resonator-band features -> PNN/PAT) is exercised. To run Phase 0 on a")
    print("real corpus:  py phase0_acoustic.py --real <path-to>/ShipsEar")


if __name__ == "__main__":
    main()
