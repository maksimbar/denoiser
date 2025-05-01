# src/dsp.py
"""
Data-set builder for R-CED style speech-denoising.

* Builds (X, Y) pairs where
    X  =  129×N  window of **noisy** magnitudes
    Y  =  129     “phase-aware” clean magnitude
          |S_clean| · cos(Δφ)  with Δφ clamped to ±max_phase_diff

* Saves        train/valid/test   →  inputs.npy / targets.npy
* Computes     μ, σ   (per-bin)   →  mean_std.npz
"""

from __future__ import annotations
import logging, random, math
from pathlib import Path

import numpy as np
import torch, torchaudio
from omegaconf import DictConfig

log = logging.getLogger(__name__)


# ───────────────────────────────── helpers ───────────────────────────────── #


def stft_mag_phase(wav: torch.Tensor, n_fft: int, hop: int, win: int):
    """Return magnitude and phase (only positive-freq bins)."""
    spec = torch.stft(
        wav,
        n_fft,
        hop_length=hop,
        win_length=win,
        window=torch.hamming_window(win, device=wav.device),
        return_complex=True,
        center=True,
        pad_mode="constant",
    )
    mag, ph = torch.abs(spec), torch.angle(spec)
    return mag[: n_fft // 2 + 1], ph[: n_fft // 2 + 1]


def slice_examples(
    mn: torch.Tensor,
    mc: torch.Tensor,
    pn: torch.Tensor,
    pc: torch.Tensor,
    input_frames: int,
    max_phase_diff: float,
):
    """
    Build one training tensor from a clean/noisy pair.

    • mn / pn : noisy magnitude / phase
    • mc / pc : clean magnitude / phase
    • returns  X  (num_slices, 129, input_frames)
               Y  (num_slices, 129)
    """
    T = min(mn.shape[1], mc.shape[1])  # align lengths
    mn, mc, pn, pc = (x[:, :T] for x in (mn, mc, pn, pc))

    # ----- phase-aware target ---------------------------------------------
    delta = (pc - pn + torch.pi) % (2 * torch.pi) - torch.pi  # wrap to (−π,π]
    delta = torch.clamp(delta, -max_phase_diff, max_phase_diff)
    tgt = mc * torch.cos(delta)  # shape: 129 × T
    # ----------------------------------------------------------------------

    xs, ys = [], []
    for t in range(input_frames - 1, T):
        xs.append(mn[:, t - input_frames + 1 : t + 1])  # 129 × input_frames
        ys.append(tgt[:, t])  # 129
    return torch.stack(xs), torch.stack(ys)  # (N, 129, F) , (N, 129)


# ───────────────────────────── split processor ───────────────────────────── #


def _process_split(
    name: str,
    clean_files: list[Path],
    noisy_dir: Path,
    cache_root: Path,
    sr: int,
    n_fft: int,
    hop: int,
    win: int,
    input_frames: int,
    max_phase_diff: float,
):
    xs, ys = [], []
    log.info(f"{name}: {len(clean_files)} clean wavs")

    for cp in clean_files:
        stem = cp.stem
        wc, r = torchaudio.load(cp)
        if r != sr:
            wc = torchaudio.functional.resample(wc, r, sr)
        wc = wc.mean(0) if wc.shape[0] > 1 else wc.squeeze(0)
        mc, pc = stft_mag_phase(wc, n_fft, hop, win)

        for npf in noisy_dir.glob(f"{stem}_w??.wav"):
            wn, r = torchaudio.load(npf)
            if r != sr:
                wn = torchaudio.functional.resample(wn, r, sr)
            wn = wn.mean(0) if wn.shape[0] > 1 else wn.squeeze(0)
            mn, pn = stft_mag_phase(wn, n_fft, hop, win)

            x, y = slice_examples(mn, mc, pn, pc, input_frames, max_phase_diff)
            if x.numel():  # guard against empty slice
                xs.append(x)
                ys.append(y)

    X = torch.cat(xs).numpy().astype("float32")
    Y = torch.cat(ys).numpy().astype("float32")
    log.info(f"{name}: X{X.shape}  Y{Y.shape}")

    (cache_root / name).mkdir(parents=True, exist_ok=True)
    np.save(cache_root / name / "inputs.npy", X)
    np.save(cache_root / name / "targets.npy", Y)
    return X, Y


# ───────────────────────────── public entry ─────────────────────────────── #


def preprocess_dataset(cfg: DictConfig):
    clean_dir = Path(cfg.data.samples.clean).expanduser()
    noisy_dir = Path(cfg.data.samples.noisy).expanduser()
    cache_dir = Path(cfg.data.cache).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(clean_dir.glob("*.wav"))
    random.seed(cfg.data.seed)
    random.shuffle(files)

    n = len(files)
    n_train, n_valid = math.floor(0.70 * n), math.floor(0.15 * n)
    splits = {
        "train": files[:n_train],
        "valid": files[n_train : n_train + n_valid],
        "test": files[n_train + n_valid :],
    }

    max_diff = math.radians(cfg.stft.max_phase_diff)  # 45° default

    trX = trY = None
    for name, flist in splits.items():
        X, Y = _process_split(
            name,
            flist,
            noisy_dir,
            cache_dir,
            cfg.data.sample_rate,
            cfg.stft.n_fft,
            cfg.stft.hop_length,
            cfg.stft.win_length,
            cfg.stft.input_frames,
            max_diff,
        )
        if name == "train":
            trX, trY = X, Y

    # ---------- per-frequency mean & std (paper uses z-score) -------------
    mu_in = trX.mean((0, 2), keepdims=True)
    std_in = trX.std((0, 2), keepdims=True)
    mu_tg = trY.mean(0, keepdims=True)
    std_tg = trY.std(0, keepdims=True)

    eps = 1e-8
    std_in[std_in < eps] = eps
    std_tg[std_tg < eps] = eps

    np.savez(
        cache_dir / "mean_std.npz",
        mu_in=mu_in.squeeze(),
        std_in=std_in.squeeze(),
        mu_tg=mu_tg.squeeze(),
        std_tg=std_tg.squeeze(),
    )
    log.info("Saved mean_std.npz")
