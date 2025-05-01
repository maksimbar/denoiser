import math
import random
import re
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio
from omegaconf import DictConfig

from pesq import pesq
from pystoi import stoi
from src.dsp import stft_mag_phase


def calculate_mag_sdr(
    estimation_mag: torch.Tensor, origin_mag: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    if origin_mag.shape != estimation_mag.shape:
        min_len = min(origin_mag.shape[-1], estimation_mag.shape[-1])
        origin_mag = origin_mag[..., :min_len]
        estimation_mag = estimation_mag[..., :min_len]

    origin_mag = origin_mag.clamp_min(0).float()
    estimation_mag = estimation_mag.clamp_min(0).float()

    if origin_mag.dim() == 2:
        origin_mag = origin_mag.unsqueeze(0)
        estimation_mag = estimation_mag.unsqueeze(0)

    dims = tuple(range(1, origin_mag.dim()))
    origin_power = torch.sum(origin_mag**2, dim=dims, keepdim=True)
    error_power = (
        torch.sum((origin_mag - estimation_mag) ** 2, dim=dims, keepdim=True) + epsilon
    )
    return (10 * torch.log10(origin_power / error_power)).mean()


def evaluate(model, cfg: DictConfig, device: torch.device, logger: logging.Logger):
    logger.info("Evaluation starts")

    model.eval()

    clean_dir = Path(cfg.data.samples.clean).expanduser().resolve()
    noisy_dir = Path(cfg.data.samples.noisy).expanduser().resolve()
    cache_dir = Path(cfg.data.cache).expanduser().resolve()

    sr = cfg.data.sample_rate
    n_fft, hop, win, input_frames = (
        cfg.stft.n_fft,
        cfg.stft.hop_length,
        cfg.stft.win_length,
        cfg.stft.input_frames,
    )

    all_clean_files = sorted(clean_dir.glob("*.wav"))
    random.seed(cfg.data.seed)
    random.shuffle(all_clean_files)

    n_files = len(all_clean_files)
    n_train = math.floor(0.70 * n_files)
    n_valid = math.floor(0.15 * n_files)
    test_clean_files = all_clean_files[n_train + n_valid :]

    stats = np.load(cache_dir / "mean_std.npz")
    mu_in, std_in = [
        torch.from_numpy(stats[k]).float().unsqueeze(1).to(device)
        for k in ("mu_in", "std_in")
    ]
    mu_tg, std_tg = [
        torch.from_numpy(stats[k]).float().to(device) for k in ("mu_tg", "std_tg")
    ]
    eps = 1e-8

    pesq_by_snr, stoi_by_snr, sdr_mag_by_snr = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    snr_pattern = re.compile(r"_w(\d+)")

    for cp in test_clean_files:
        wc_orig, r = torchaudio.load(cp)
        if r != sr:
            wc_orig = torchaudio.functional.resample(wc_orig, r, sr)
        wc_wav = wc_orig.mean(dim=0) if wc_orig.shape[0] > 1 else wc_orig.squeeze(0)
        wc_wav = wc_wav.to(device)
        mc, _ = stft_mag_phase(wc_wav, n_fft, hop, win)

        for npf in sorted(noisy_dir.glob(f"{cp.stem}_w??.wav")):
            match = snr_pattern.search(npf.name)
            snr_level = int(match.group(1)) if match else None

            wn_orig, r = torchaudio.load(npf)
            if r != sr:
                wn_orig = torchaudio.functional.resample(wn_orig, r, sr)
            wn_wav = wn_orig.mean(dim=0) if wn_orig.shape[0] > 1 else wn_orig.squeeze(0)
            wn_wav = wn_wav.to(device)

            mn, pn = stft_mag_phase(wn_wav, n_fft, hop, win)
            if mn.shape[1] < input_frames:
                continue

            noisy_frames = mn.unfold(1, input_frames, 1)
            num_frames = noisy_frames.shape[1]
            pn_frames = torch.nn.functional.pad(pn, (input_frames - 1, 0))[
                :, input_frames - 1 : num_frames + input_frames - 1
            ]
            noisy_frames = noisy_frames.permute(1, 0, 2)
            noisy_frames = ((noisy_frames - mu_in) / (std_in + eps)).permute(0, 2, 1)

            with torch.no_grad():
                pred_mag_norm = torch.cat(
                    [
                        model(noisy_frames[i : i + 10000])
                        for i in range(0, num_frames, 10000)
                    ],
                    dim=0,
                )

            pred_mag = (pred_mag_norm * (std_tg + eps)) + mu_tg
            enhanced_mag = pred_mag.T

            start_idx = input_frames - 1
            eff_frames = min(num_frames, mc.shape[1] - start_idx)
            sdr_val = calculate_mag_sdr(
                enhanced_mag[:, :eff_frames], mc[:, start_idx : start_idx + eff_frames]
            )
            if snr_level is not None:
                sdr_mag_by_snr[snr_level].append(sdr_val.item())

            mag_istft = enhanced_mag[
                :, : min(enhanced_mag.shape[1], pn_frames.shape[1])
            ]
            enh_complex = mag_istft * torch.exp(1j * pn_frames[:, : mag_istft.shape[1]])
            enh_wav = torch.istft(
                enh_complex,
                n_fft=n_fft,
                hop_length=hop,
                win_length=win,
                window=torch.hamming_window(win, device=device),
                center=True,
            ).cpu()

            min_len = min(wc_wav.shape[0], enh_wav.shape[0])
            wc_np, enh_np = wc_wav[:min_len].cpu().numpy(), enh_wav[:min_len].numpy()

            if snr_level is not None:
                mode = "nb" if sr == 8000 else "wb"
                pesq_by_snr[snr_level].append(pesq(sr, wc_np, enh_np, mode))
                stoi_by_snr[snr_level].append(stoi(wc_np, enh_np, sr, extended=False))

    results_overall = {}
    for name, data in (
        ("pesq", pesq_by_snr),
        ("stoi", stoi_by_snr),
        ("sdr_mag", sdr_mag_by_snr),
    ):
        scores = [s for lst in data.values() for s in lst]
        results_overall[name] = {
            "mean": np.mean(scores) if scores else 0.0,
            "count": len(scores),
        }

    results_per_snr = [
        {
            "snr_level": snr,
            "avg_pesq": np.mean(pesq_by_snr[snr]) if pesq_by_snr[snr] else 0.0,
            "avg_stoi": np.mean(stoi_by_snr[snr]) if stoi_by_snr[snr] else 0.0,
            "avg_sdr_mag": np.mean(sdr_mag_by_snr[snr]) if sdr_mag_by_snr[snr] else 0.0,
            "num_files": len(pesq_by_snr[snr]),
        }
        for snr in sorted(pesq_by_snr.keys())
    ]

    logger.info("Evaluation complete")
    return results_overall, results_per_snr
