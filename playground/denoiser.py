#!/usr/bin/env python3


from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
import hydra
from omegaconf import DictConfig

sys.path.append(str(Path(__file__).parent.parent))

# ─── locate files relative to script ─────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
NOISY_WAV = SCRIPT_DIR / "noisy.wav"
MODEL_PT = SCRIPT_DIR / "best_model.pt"
STATS_NPZ = SCRIPT_DIR / "mean_std.npz"
OUT_WAV = SCRIPT_DIR / "enhanced.wav"
OUT_IMG = SCRIPT_DIR / "compare.png"

# ─── import RCED robustly ────────────────────────────────────
# 1) try local "model" module (playground/model.py)
# 2) try original package "src.model"
# 3) if both fail, but a file named model.py exists, load it via importlib
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from model import RCED  # type: ignore
except ModuleNotFoundError:
    try:
        from src.model import RCED  # type: ignore
    except ModuleNotFoundError:
        import importlib.machinery, importlib.util

        model_path = SCRIPT_DIR / "model.py"
        if not model_path.exists():
            raise  # no fallbacks left
        loader = importlib.machinery.SourceFileLoader("rced_dynamic", str(model_path))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        loader.exec_module(mod)  # type: ignore[arg-type]
        RCED = mod.RCED  # type: ignore[attr-defined]

# ─── basic DSP helper ──────────────────────────────────────── ────────────────────────────────────────


def stft_mag_phase(wav: torch.Tensor, n_fft: int, hop: int, win: int):
    s = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=torch.hamming_window(win, device=wav.device),
        return_complex=True,
        center=True,
        pad_mode="constant",
    )
    m, p = torch.abs(s), torch.angle(s)
    return m[: n_fft // 2 + 1, :], p[: n_fft // 2 + 1, :]


# ─── Hydra entry‑point (to reuse training config) ────────────


@hydra.main(config_path="..", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # STFT & model hyper‑parameters from config
    sr = cfg.data.sample_rate
    n_fft, hop, win, input_frames = (
        cfg.stft.n_fft,
        cfg.stft.hop_length,
        cfg.stft.win_length,
        cfg.stft.input_frames,
    )

    # normalisation stats
    stats = np.load(STATS_NPZ)
    mu_in, std_in = [
        torch.from_numpy(stats[k]).float().unsqueeze(1).to(device)
        for k in ("mu_in", "std_in")
    ]
    mu_tg, std_tg = [
        torch.from_numpy(stats[k]).float().to(device) for k in ("mu_tg", "std_tg")
    ]
    eps = 1e-8

    # model
    model = RCED(
        filters=list(cfg.model.filters),
        kernels=list(cfg.model.kernels),
        in_ch=input_frames,
    ).to(device)
    missing = model.load_state_dict(
        torch.load(MODEL_PT, map_location=device), strict=False
    )
    if missing.missing_keys:
        print(
            f"⚠️  Ignored {len(missing.missing_keys)} missing keys (biases?) when loading weights."
        )
    model.eval()

    # noisy wav
    wav, r = torchaudio.load(NOISY_WAV)
    if r != sr:
        wav = torchaudio.functional.resample(wav, r, sr)
    wav = wav.mean(0) if wav.shape[0] > 1 else wav.squeeze(0)
    wav = wav.to(device)

    mag, phase = stft_mag_phase(wav, n_fft, hop, win)

    # inference
    frames = mag.unfold(1, input_frames, 1).permute(1, 0, 2)
    frames = ((frames - mu_in) / (std_in + eps)).permute(0, 2, 1)
    with torch.no_grad():
        pred_norm = torch.cat(
            [model(frames[i : i + 10000]) for i in range(0, frames.size(0), 10000)],
            dim=0,
        )

    # ---------- probe & safety clamp ---------------- #
    pred_mag = pred_norm * (std_tg + eps) + mu_tg  #   ← already linear magnitude

    neg_pct = (pred_mag < 0).float().mean().item() * 100
    print(
        f"[probe] pred_mag   min={pred_mag.min():.4f}   "
        f"max={pred_mag.max():.4f}   neg%={neg_pct:.2f}"
    )

    pred_mag = torch.clamp(pred_mag, min=0.0)  # keep ISTFT & log10 happy
    # ------------------------------------------------ #

    enhanced_mag = pred_mag.T

    pad = input_frames - 1
    phase_frames = torch.nn.functional.pad(phase, (pad, 0))[
        :, pad : pad + enhanced_mag.shape[1]
    ]
    enh_complex = enhanced_mag * torch.exp(1j * phase_frames)
    enh_wav = torch.istft(
        enh_complex,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=torch.hamming_window(win, device=device),
        center=True,
    ).cpu()

    torchaudio.save(OUT_WAV, enh_wav.unsqueeze(0), sr)

    # comparison plot
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 2, figsize=(12, 6))
        t_noisy = np.linspace(0, wav.numel() / sr, wav.numel())
        t_enh = np.linspace(0, enh_wav.numel() / sr, enh_wav.numel())
        ax[0, 0].plot(t_noisy, wav.cpu().numpy())
        ax[0, 0].set_title("Noisy waveform")
        ax[0, 1].plot(t_enh, enh_wav.numpy())
        ax[0, 1].set_title("Denoised waveform")

        noisy_spec = 20 * torch.log10(mag + 1e-6).cpu().numpy()
        clean_spec = 20 * torch.log10(enhanced_mag + 1e-6).cpu().numpy()
        im0 = ax[1, 0].imshow(noisy_spec, origin="lower", aspect="auto")
        ax[1, 0].set_title("Noisy log-mag")
        im1 = ax[1, 1].imshow(clean_spec, origin="lower", aspect="auto")
        ax[1, 1].set_title("Denoised log-mag")
        fig.colorbar(im0, ax=ax[1, 0])
        fig.colorbar(im1, ax=ax[1, 1])
        fig.tight_layout()
        fig.savefig(OUT_IMG)
        plt.close(fig)
        print(f"✔ saved plot  → {OUT_IMG}")
    except ModuleNotFoundError:
        print("matplotlib not installed – skipping plot.")

    print(f"✔ saved wav   → {OUT_WAV}")


if __name__ == "__main__":
    main()
