"""
iara_dataset.py -- adapter feeding the IARA acoustical archive
(Zenodo 10.5281/zenodo.15758636) into the Phase-0 PNN pipeline.

Verified IARA structure (iara.xlsx):
  * sheet 'dataset_info' (1825 rows): IARA ID, Dataset(volume A..H), Ship ID, ...
  * sheet 'ship_info'    ( 645 rows): Ship ID -> 'AIS TYPE SUMMARY' (vessel class)
  * audio: <VOL>/<VOL>-<IARA ID>.wav, 128 kHz mono 16-bit, 300 s each.
  * volume H (IARA ID 1779-1825) has NO ship -> BACKGROUND noise.

This adapter:
  1. maps filename IARA ID -> Ship ID -> vessel class (or 'Background'),
  2. SELECTS a sample of recordings per requested class up front (so we read
     only the needed files, not all 1825, and only the first ~50 s of each),
  3. decimates 128 kHz -> ~4 kHz (band of interest) and computes the SAME
     resonator-band power-spectrum feature as core.load_real_dataset,
  4. returns per-segment RECORDING id for leakage-free (by-recording) splits,
  5. caches the extracted (X, y, rec) so multi-seed training is instant.
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

import core


def build_iara_label_map(meta_xlsx):
    """Return (id2cls, id2vol) dicts keyed by IARA ID (recording number)."""
    import pandas as pd
    di = pd.read_excel(meta_xlsx, sheet_name="dataset_info")
    si = pd.read_excel(meta_xlsx, sheet_name="ship_info")
    di.columns = [str(c).strip() for c in di.columns]
    si.columns = [str(c).strip() for c in si.columns]
    ship2cls = {}
    for _, r in si.iterrows():
        try:
            ship2cls[int(r["Ship ID"])] = str(r["AIS TYPE SUMMARY"]).strip()
        except (ValueError, TypeError):
            pass
    id2cls, id2vol = {}, {}
    for _, r in di.iterrows():
        iid = int(r["IARA ID"])
        id2vol[iid] = str(r["Dataset"]).strip()
        try:
            id2cls[iid] = ship2cls.get(int(r["Ship ID"]), "Background")
        except (ValueError, TypeError):
            id2cls[iid] = "Background"
    return id2cls, id2vol


def _read_wav_head(raw: bytes, max_seconds: float):
    """Read only the first max_seconds of a WAV (mono float)."""
    w = wave.open(io.BytesIO(raw))
    sr, ch = w.getframerate(), w.getnchannels()
    nframes = min(w.getnframes(), int(max_seconds * sr))
    data = np.frombuffer(w.readframes(nframes), dtype=np.int16).astype(np.float64)
    if ch > 1:
        data = data[::ch]
    return sr, data


def load_iara(cfg: core.Config, zip_dir, meta_xlsx, classes: Tuple[str, ...],
              seg_seconds: float = 4.0, max_per_rec: int = 12,
              recs_per_class: int = 50, decimate_to: float = 4000.0,
              seed: int = 0, cache=None, verbose: bool = True):
    """Returns (X, y, rec, classes). Uses cache (.npz) if present."""
    if cache is not None and Path(cache).exists():
        d = np.load(cache, allow_pickle=True)
        if verbose:
            print(f"[iara] loaded cache {cache}: X{d['X'].shape}")
        return (torch.from_numpy(d["X"]), torch.from_numpy(d["y"]),
                d["rec"], list(d["classes"]))

    from scipy.signal import decimate
    id2cls, id2vol = build_iara_label_map(meta_xlsx)
    cls_idx = {c: i for i, c in enumerate(classes)}
    rng = np.random.default_rng(seed)

    # select recordings per class -> group targets by volume
    targets: Dict[str, List[Tuple[int, str]]] = {}
    for c in classes:
        ids = [iid for iid, cl in id2cls.items() if cl == c]
        rng.shuffle(ids)
        for iid in ids[:recs_per_class]:
            targets.setdefault(id2vol[iid], []).append((iid, c))

    read_seconds = max_per_rec * seg_seconds + 4.0
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
            seglen = int(seg_seconds * sr)
            if seglen < 256 or len(data) < seglen:
                continue
            for k in range(min(max_per_rec, len(data) // seglen)):
                seg = data[k * seglen:(k + 1) * seglen]
                if np.max(np.abs(seg)) < 1e-4:
                    continue
                X.append(core._segment_spectrum(seg, sr, cfg))
                y.append(cls_idx[c]); rec.append(iid); per_class[c] += 1
    if verbose:
        print(f"[iara] segments per class: {per_class}")
    if not X:
        raise RuntimeError("no IARA segments extracted")
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    rec = np.array(rec, dtype=np.int64)
    if cache is not None:
        np.savez(cache, X=X, y=y, rec=rec, classes=np.array(list(classes)))
        if verbose:
            print(f"[iara] cached -> {cache}")
    return torch.from_numpy(X), torch.from_numpy(y), rec, list(classes)


def grouped_split(y: torch.Tensor, rec: np.ndarray, test_frac: float = 0.3, seed: int = 0):
    """Train/test split BY RECORDING id (no segment leakage), stratified by class."""
    rng = np.random.default_rng(seed)
    yn = y.numpy()
    tr_mask = np.zeros(len(yn), dtype=bool)
    te_mask = np.zeros(len(yn), dtype=bool)
    for c in np.unique(yn):
        recs = np.unique(rec[yn == c])
        rng.shuffle(recs)
        n_te = max(1, int(round(test_frac * len(recs))))
        te = set(recs[:n_te].tolist())
        for i in np.where(yn == c)[0]:
            (te_mask if rec[i] in te else tr_mask)[i] = True
    return tr_mask, te_mask
