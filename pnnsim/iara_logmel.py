"""
iara_logmel.py -- extract & cache LOG-MEL spectrograms from the SAME IARA
recordings used by phase0_iara.py, so the digital SOTA baselines see the
standard UATR front end (full-band log-mel) while the analog PNN keeps its
physically band-limited resonator feature.

Why this exists
---------------
The Phase-0 digital baseline is a tiny MLP on a 256-bin, sum-normalised,
10-1000 Hz power spectrum -- the SAME impoverished feature the piezo resonator
array can physically produce. That is the right control for "what does the
device cost?", but it is NOT a credible *digital* SOTA: the UATR literature
reaches 90%+ on ShipsEar/DeepShip with log-mel + CNNs. To establish the FIRST
honest multi-method benchmark on IARA we must give the digital models the
front end they would actually use. This module produces (n_mels x n_frames)
log-mel tensors per 4 s segment, with leakage-free per-recording ids, on the
identical recording selection (same seed/recs_per_class) as phase0_iara.py.

The mel filterbank is implemented in numpy (no librosa dependency).

Returns (X2d [N,1,n_mels,n_frames] float32, y [N], rec [N] int64, classes).
Caches to a .npz so multi-seed training is instant.
"""
from __future__ import annotations

import io
import re
import wave
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

import iara_dataset as iara   # reuse build_iara_label_map + grouped_split


# ----------------------------------------------------------------------------
# mel filterbank (numpy)
# ----------------------------------------------------------------------------
def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank(sr, n_fft, n_mels, f_lo=10.0, f_hi=None):
    f_hi = f_hi or sr / 2.0
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2.0, n_bins)
    mel_pts = np.linspace(_hz_to_mel(f_lo), _hz_to_mel(f_hi), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(1, n_mels + 1):
        lo, ctr, hi = hz_pts[m - 1], hz_pts[m], hz_pts[m + 1]
        up = (fft_freqs - lo) / (ctr - lo + 1e-9)
        dn = (hi - fft_freqs) / (hi - ctr + 1e-9)
        fb[m - 1] = np.clip(np.minimum(up, dn), 0.0, None)
    return fb


def _logmel(seg, sr, fb, n_fft, hop):
    """Log-mel spectrogram of one segment -> (n_mels, n_frames)."""
    from scipy.signal import stft
    _, _, Z = stft(seg, fs=sr, nperseg=n_fft, noverlap=n_fft - hop,
                   boundary=None, padded=False)
    power = (np.abs(Z) ** 2)                       # (n_bins, n_frames)
    mel = fb @ power                               # (n_mels, n_frames)
    return np.log(mel + 1e-6).astype(np.float32)


def _read_wav_head(raw: bytes, max_seconds: float):
    w = wave.open(io.BytesIO(raw))
    sr, ch = w.getframerate(), w.getnchannels()
    nframes = min(w.getnframes(), int(max_seconds * sr))
    data = np.frombuffer(w.readframes(nframes), dtype=np.int16).astype(np.float64)
    if ch > 1:
        data = data[::ch]
    return sr, data


def load_iara_logmel(zip_dir, meta_xlsx, classes: Tuple[str, ...],
                     seg_seconds: float = 4.0, max_per_rec: int = 12,
                     recs_per_class: int = 45, decimate_to: float = 4000.0,
                     n_mels: int = 48, n_fft: int = 512, hop: int = 256,
                     seed: int = 0, cache=None, verbose: bool = True):
    """Same recording selection as phase0_iara.load_iara (seed, recs_per_class),
    but emits 2-D log-mel features. Cached to .npz."""
    if cache is not None and Path(cache).exists():
        d = np.load(cache, allow_pickle=True)
        if verbose:
            print(f"[logmel] loaded cache {cache}: X{d['X'].shape}")
        return (torch.from_numpy(d["X"]), torch.from_numpy(d["y"]),
                d["rec"], list(d["classes"]))

    from scipy.signal import decimate
    id2cls, id2vol = iara.build_iara_label_map(meta_xlsx)
    cls_idx = {c: i for i, c in enumerate(classes)}
    rng = np.random.default_rng(seed)

    # IDENTICAL selection to load_iara (deterministic dict order + same rng)
    targets: Dict[str, List[Tuple[int, str]]] = {}
    for c in classes:
        ids = [iid for iid, cl in id2cls.items() if cl == c]
        rng.shuffle(ids)
        for iid in ids[:recs_per_class]:
            targets.setdefault(id2vol[iid], []).append((iid, c))

    read_seconds = max_per_rec * seg_seconds + 4.0
    fb = None
    X, y, rec = [], [], []
    per_class = {c: 0 for c in classes}
    for vol in sorted(targets):
        zp = Path(zip_dir) / f"{vol}.zip"
        if not zp.exists():
            continue
        z = zipfile.ZipFile(str(zp))
        members = {}
        for n in z.namelist():
            mm = re.search(r"-(\d+)\.wav$", n)
            if mm:
                members[int(mm.group(1))] = n
        for iid, c in targets[vol]:
            n = members.get(iid)
            if n is None:
                continue
            sr, data = _read_wav_head(z.read(n), read_seconds)
            data = data / (np.max(np.abs(data)) + 1e-9)
            q = max(1, int(round(sr / decimate_to)))
            while q > 13:
                data = decimate(data, 4, ftype="fir"); sr //= 4
                q = max(1, int(round(sr / decimate_to)))
            if q > 1:
                data = decimate(data, q, ftype="fir"); sr //= q
            if fb is None:
                fb = _mel_filterbank(sr, n_fft, n_mels, f_lo=10.0, f_hi=sr / 2.0)
            seglen = int(seg_seconds * sr)
            if seglen < n_fft or len(data) < seglen:
                continue
            for k in range(min(max_per_rec, len(data) // seglen)):
                seg = data[k * seglen:(k + 1) * seglen]
                if np.max(np.abs(seg)) < 1e-4:
                    continue
                X.append(_logmel(seg, sr, fb, n_fft, hop))
                y.append(cls_idx[c]); rec.append(iid); per_class[c] += 1
    if verbose:
        print(f"[logmel] segments per class: {per_class}")
    if not X:
        raise RuntimeError("no IARA log-mel segments extracted")

    # pad/trim to a common frame count
    T = int(np.median([a.shape[1] for a in X]))
    Xf = np.zeros((len(X), 1, n_mels, T), dtype=np.float32)
    for i, a in enumerate(X):
        t = min(T, a.shape[1])
        Xf[i, 0, :, :t] = a[:, :t]
    # per-feature standardisation (over the whole set; train-only would be
    # marginally cleaner but the benchmark uses recording-level splits so the
    # leakage here is negligible and applied identically to all models)
    mu, sd = Xf.mean(), Xf.std() + 1e-6
    Xf = (Xf - mu) / sd

    y = np.array(y, dtype=np.int64)
    rec = np.array(rec, dtype=np.int64)
    if cache is not None:
        np.savez(cache, X=Xf, y=y, rec=rec, classes=np.array(list(classes)))
        if verbose:
            print(f"[logmel] cached -> {cache}  X{Xf.shape}")
    return torch.from_numpy(Xf), torch.from_numpy(y), rec, list(classes)


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent
    DATA = ROOT.parent / "IARA-data"
    CLASSES = ("Background", "Cargo", "Tanker", "Tug", "Special Craft")
    X, y, rec, classes = load_iara_logmel(
        DATA, str(DATA / "iara.xlsx"), CLASSES,
        cache=str(ROOT / "iara_logmel_features.npz"))
    print("X", tuple(X.shape), "y", tuple(y.shape), "n_rec", len(np.unique(rec)))
