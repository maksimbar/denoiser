import math, random, re, logging
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, torchaudio
from omegaconf import DictConfig
from pesq import pesq
from pystoi import stoi
from src.dsp import stft_mag_phase


def calculate_mag_sdr(est, ref, eps: float = 1e-8):
    if est.shape != ref.shape:
        m = min(est.shape[-1], ref.shape[-1])
        est, ref = est[..., :m], ref[..., :m]
    est, ref = est.clamp_min(0).float(), ref.clamp_min(0).float()
    if est.dim() == 2:
        est, ref = est.unsqueeze(0), ref.unsqueeze(0)
    dims = tuple(range(1, est.dim()))
    return (
        10
        * torch.log10(
            torch.sum(ref**2, dims, True)
            / (torch.sum((ref - est) ** 2, dims, True) + eps)
        )
    ).mean()


def evaluate(model, cfg: DictConfig, device: torch.device, logger: logging.Logger):
    logger.info("Evaluation starts")
    model.eval()

    split_root = Path(cfg.data.samples.test).expanduser().resolve()
    clean_dir, noisy_dir = split_root / "clean", split_root / "noisy"
    cache_dir = Path(cfg.data.cache).expanduser().resolve()

    sr = cfg.data.sample_rate
    n_fft, hop, win, input_frames = (
        cfg.stft.n_fft,
        cfg.stft.hop_length,
        cfg.stft.win_length,
        cfg.stft.input_frames,
    )

    test_clean_files = sorted(clean_dir.glob("*.wav"))

    stats = np.load(cache_dir / "mean_std.npz")
    mu_in, std_in = [
        torch.from_numpy(stats[k]).float().unsqueeze(1).to(device)
        for k in ("mu_in", "std_in")
    ]
    mu_tg, std_tg = [
        torch.from_numpy(stats[k]).float().to(device) for k in ("mu_tg", "std_tg")
    ]
    eps = 1e-8

    pesq_d, stoi_d, sdr_d = defaultdict(list), defaultdict(list), defaultdict(list)
    snr_pat = re.compile(r"_w(\d+)")

    for cp in test_clean_files:
        w, r = torchaudio.load(cp)
        if r != sr:
            w = torchaudio.functional.resample(w, r, sr)
        w = w.mean(0) if w.shape[0] > 1 else w.squeeze(0)
        w = w.to(device)
        mc, _ = stft_mag_phase(w, n_fft, hop, win)

        for npf in sorted(noisy_dir.glob(f"{cp.stem}_w??.wav")):
            m = snr_pat.search(npf.name)
            snr = int(m.group(1)) if m else None

            wn, r = torchaudio.load(npf)
            if r != sr:
                wn = torchaudio.functional.resample(wn, r, sr)
            wn = wn.mean(0) if wn.shape[0] > 1 else wn.squeeze(0)
            wn = wn.to(device)

            mn, pn = stft_mag_phase(wn, n_fft, hop, win)
            if mn.shape[1] < input_frames:
                continue

            nf = mn.unfold(1, input_frames, 1)
            frames = nf.shape[1]
            pn_f = torch.nn.functional.pad(pn, (input_frames - 1, 0))[
                :, input_frames - 1 : frames + input_frames - 1
            ]
            nf = ((nf.permute(1, 0, 2) - mu_in) / (std_in + eps)).permute(0, 2, 1)

            with torch.no_grad():
                pred = torch.cat(
                    [model(nf[i : i + 10000]) for i in range(0, frames, 10000)], 0
                )
            pred = pred * (std_tg + eps) + mu_tg
            enh_mag = pred.T

            s_idx = input_frames - 1
            eff = min(frames, mc.shape[1] - s_idx)
            sdr_val = calculate_mag_sdr(enh_mag[:, :eff], mc[:, s_idx : s_idx + eff])
            if snr is not None:
                sdr_d[snr].append(sdr_val.item())

            m_istft = enh_mag[:, : min(enh_mag.shape[1], pn_f.shape[1])]
            c = m_istft * torch.exp(1j * pn_f[:, : m_istft.shape[1]])
            e = torch.istft(
                c,
                n_fft,
                hop,
                win,
                window=torch.hamming_window(win, device=device),
                center=True,
            ).cpu()

            l = min(w.shape[0], e.shape[0])
            ref_np, enh_np = w[:l].cpu().numpy(), e[:l].numpy()
            if snr is not None:
                mode = "nb" if sr == 8000 else "wb"
                pesq_d[snr].append(pesq(sr, ref_np, enh_np, mode))
                stoi_d[snr].append(stoi(ref_np, enh_np, sr, extended=False))

    overall = {}
    for k, d in (("pesq", pesq_d), ("stoi", stoi_d), ("sdr_mag", sdr_d)):
        vals = [v for lst in d.values() for v in lst]
        overall[k] = {"mean": np.mean(vals) if vals else 0.0, "count": len(vals)}

    expected = list(range(0, 34, 3))
    per_snr = [
        {
            "snr_level": s,
            "avg_pesq": np.mean(pesq_d[s]) if pesq_d[s] else 0.0,
            "avg_stoi": np.mean(stoi_d[s]) if stoi_d[s] else 0.0,
            "avg_sdr_mag": np.mean(sdr_d[s]) if sdr_d[s] else 0.0,
            "num_files": len(pesq_d[s]),
        }
        for s in expected
    ]

    logger.info("Evaluation complete")
    return overall, per_snr
