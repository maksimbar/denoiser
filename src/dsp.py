import logging
import math
from pathlib import Path
import numpy as np
import torch
import torchaudio
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def stft_mag_phase(wav: torch.Tensor, n_fft: int, hop: int, win: int):
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
    # pn: torch.Tensor, # REMOVED
    # pc: torch.Tensor, # REMOVED
    input_frames: int,
    # max_phase_diff: float, # REMOVED
):
    T = min(mn.shape[1], mc.shape[1])
    # mn, mc, pn, pc = (x[:, :T] for x in (mn, mc, pn, pc)) # MODIFIED
    mn, mc = mn[:, :T], mc[:, :T]
    # delta = (pc - pn + torch.pi) % (2 * torch.pi) - torch.pi # REMOVED
    # delta = torch.clamp(delta, -max_phase_diff, max_phase_diff) # REMOVED
    # tgt = mc * torch.cos(delta) # REMOVED
    xs, ys = [], []
    for t in range(input_frames - 1, T):
        xs.append(mn[:, t - input_frames + 1 : t + 1])
        # ys.append(tgt[:, t]) # MODIFIED
        ys.append(mc[:, t])  # USE CLEAN MAGNITUDE DIRECTLY
    if xs:
        return torch.stack(xs), torch.stack(ys)
    # Return empty tensors matching expected dimensions if no examples
    num_freq_bins = mn.shape[0]
    return torch.empty(
        (0, num_freq_bins, input_frames), dtype=mn.dtype, device=mn.device
    ), torch.empty((0, num_freq_bins), dtype=mc.dtype, device=mc.device)


def _count_samples(
    clean_files: list[Path],
    noisy_dir: Path,
    sr: int,
    n_fft: int,
    hop: int,
    win: int,
    input_frames: int,
):
    total = 0
    for cp in clean_files:
        stem = cp.stem
        wc, r = torchaudio.load(cp)
        if r != sr:
            wc = torchaudio.functional.resample(wc, r, sr)
        wc = wc.mean(0) if wc.shape[0] > 1 else wc.squeeze(0)
        mc, _ = stft_mag_phase(wc, n_fft, hop, win)
        # MODIFIED: Use broader glob pattern, adapt if needed
        for npf in noisy_dir.glob(f"{stem}_*.wav"):
            wn, r = torchaudio.load(npf)
            if r != sr:
                wn = torchaudio.functional.resample(wn, r, sr)
            wn = wn.mean(0) if wn.shape[0] > 1 else wn.squeeze(0)
            mn, _ = stft_mag_phase(wn, n_fft, hop, win)
            T = min(mn.shape[1], mc.shape[1])
            total += max(0, T - input_frames + 1)
    return total


def _write_split(
    name: str,
    clean_files: list[Path],
    noisy_dir: Path,
    cache_root: Path,
    sr: int,
    n_fft: int,
    hop: int,
    win: int,
    input_frames: int,
    # max_phase_diff: float, # REMOVED
    X_mem: np.memmap,
    Y_mem: np.memmap,
):
    idx = 0
    num_freq_bins = n_fft // 2 + 1
    sum_in = np.zeros(num_freq_bins, dtype="float64")
    sumsq_in = np.zeros(num_freq_bins, dtype="float64")
    total_in = 0
    sum_tg = np.zeros(num_freq_bins, dtype="float64")
    sumsq_tg = np.zeros(num_freq_bins, dtype="float64")
    total_tg = 0
    for cp in clean_files:
        stem = cp.stem
        wc, r = torchaudio.load(cp)
        if r != sr:
            wc = torchaudio.functional.resample(wc, r, sr)
        wc = wc.mean(0) if wc.shape[0] > 1 else wc.squeeze(0)
        # mc, pc = stft_mag_phase(wc, n_fft, hop, win) # MODIFIED: Don't need pc
        mc, _ = stft_mag_phase(wc, n_fft, hop, win)
        # MODIFIED: Use broader glob pattern, adapt if needed
        for npf in noisy_dir.glob(f"{stem}_*.wav"):
            wn, r = torchaudio.load(npf)
            if r != sr:
                wn = torchaudio.functional.resample(wn, r, sr)
            wn = wn.mean(0) if wn.shape[0] > 1 else wn.squeeze(0)
            # mn, pn = stft_mag_phase(wn, n_fft, hop, win) # MODIFIED: Don't need pn
            mn, _ = stft_mag_phase(wn, n_fft, hop, win)
            # MODIFIED: Updated call to slice_examples
            x, y = slice_examples(mn, mc, input_frames)
            if x.numel():
                n = x.shape[0]
                # Basic bounds check before writing
                if idx + n <= X_mem.shape[0]:
                    X_mem[idx : idx + n] = x.numpy()
                    Y_mem[idx : idx + n] = y.numpy()
                    s = x.numpy().sum(axis=(0, 2))
                    ss = (x.numpy() ** 2).sum(axis=(0, 2))
                    sum_in += s
                    sumsq_in += ss
                    total_in += n * input_frames
                    s2 = y.numpy().sum(axis=0)
                    ss2 = (y.numpy() ** 2).sum(axis=0)
                    sum_tg += s2
                    sumsq_tg += ss2
                    total_tg += n
                    idx += n
                else:
                    # Simple log if overflow would occur, prevents crash but data is lost
                    log.warning(
                        f"Skipping write for {npf} due to potential memmap overflow (index {idx + n} > size {X_mem.shape[0]})"
                    )

    # Ensure final data is written
    X_mem.flush()
    Y_mem.flush()

    stats = {
        "sum_in": sum_in,
        "sumsq_in": sumsq_in,
        "total_in": total_in,
        "sum_tg": sum_tg,
        "sumsq_tg": sumsq_tg,
        "total_tg": total_tg,
    }
    return stats


def preprocess_dataset(cfg: DictConfig):
    cache_dir = Path(cfg.data.cache).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    split_roots = {
        "train": Path(cfg.data.samples.train).expanduser(),
        "valid": Path(cfg.data.samples.valid).expanduser(),
        "test": Path(cfg.data.samples.test).expanduser(),
    }
    # max_diff = math.radians(cfg.stft.max_phase_diff) # REMOVED
    stats_acc = None
    num_freq_bins = cfg.stft.n_fft // 2 + 1

    for name, root in split_roots.items():
        clean_dir = root / "clean"
        noisy_dir = root / "noisy"
        clean_files = sorted(clean_dir.glob("*.wav"))
        total = _count_samples(
            clean_files,
            noisy_dir,
            cfg.data.sample_rate,
            cfg.stft.n_fft,
            cfg.stft.hop_length,
            cfg.stft.win_length,
            cfg.stft.input_frames,
        )

        # Avoid creating empty memmaps if count is zero
        if total == 0:
            log.warning(
                f"Calculated 0 total examples for split '{name}'. Skipping memmap creation."
            )
            continue

        split_cache = cache_dir / name
        split_cache.mkdir(parents=True, exist_ok=True)

        X_mem = np.lib.format.open_memmap(
            split_cache / "inputs.npy",
            mode="w+",
            dtype="float32",
            # Use calculated num_freq_bins here
            shape=(total, num_freq_bins, cfg.stft.input_frames),
        )
        Y_mem = np.lib.format.open_memmap(
            split_cache / "targets.npy",
            mode="w+",
            dtype="float32",
            # Use calculated num_freq_bins here
            shape=(total, num_freq_bins),
        )
        stats = _write_split(
            name,
            clean_files,
            noisy_dir,
            cache_root=cache_dir,  # Pass cache_root explicitly
            sr=cfg.data.sample_rate,
            n_fft=cfg.stft.n_fft,
            hop=cfg.stft.hop_length,
            win=cfg.stft.win_length,
            input_frames=cfg.stft.input_frames,
            # max_diff=max_diff, # REMOVED
            X_mem=X_mem,
            Y_mem=Y_mem,
        )
        if name == "train":
            stats_acc = stats

        # Explicitly delete memmap objects to release file handles
        del X_mem
        del Y_mem

    # Check if stats were accumulated before proceeding
    if stats_acc is None:
        log.error(
            "No statistics accumulated (training set might be empty or failed processing). Cannot save mean/std."
        )
        raise RuntimeError("Failed to compute normalization statistics.")

    sum_in = stats_acc["sum_in"]
    sumsq_in = stats_acc["sumsq_in"]
    total_in = stats_acc["total_in"]
    sum_tg = stats_acc["sum_tg"]
    sumsq_tg = stats_acc["sumsq_tg"]
    total_tg = stats_acc["total_tg"]

    # Avoid division by zero if no data was processed
    if total_in == 0 or total_tg == 0:
        log.error(f"Cannot compute mean/std: total_in={total_in}, total_tg={total_tg}")
        raise ValueError(
            "Zero samples processed, cannot compute normalization statistics."
        )

    mu_in = sum_in / total_in
    # Add small epsilon inside sqrt for numerical stability
    std_in = np.sqrt(np.maximum(0, sumsq_in / total_in - mu_in**2))
    mu_tg = sum_tg / total_tg
    std_tg = np.sqrt(np.maximum(0, sumsq_tg / total_tg - mu_tg**2))

    eps = 1e-8
    std_in[std_in < eps] = eps
    std_tg[std_tg < eps] = eps

    np.savez(
        cache_dir / "mean_std.npz",
        mu_in=mu_in,
        std_in=std_in,
        mu_tg=mu_tg,
        std_tg=std_tg,
    )
    log.info("Saved mean_std.npz")
