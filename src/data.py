import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import logging

log = logging.getLogger(__name__)


class SpeechDataset(Dataset):
    def __init__(self, split: str, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        split_dir = self.cache_dir / split

        self.inputs = np.load(split_dir / "inputs.npy")
        self.targets = np.load(split_dir / "targets.npy")

        stats = np.load(self.cache_dir / "mean_std.npz")
        self.eps = 1e-8
        self.mu_in = torch.from_numpy(stats["mu_in"]).float().unsqueeze(1)
        self.std_in = torch.from_numpy(stats["std_in"]).float().unsqueeze(1)
        self.mu_tg = torch.from_numpy(stats["mu_tg"]).float()
        self.std_tg = torch.from_numpy(stats["std_tg"]).float()

        log.info(f"Loaded {len(self):,} samples for '{split}'")

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx]).float()
        y = torch.from_numpy(self.targets[idx]).float()

        x = (x - self.mu_in) / (self.std_in + self.eps)
        y = (y - self.mu_tg) / (self.std_tg + self.eps)

        x = x.permute(1, 0)

        return x, y

    def unnormalize_target(self, y_norm):
        return y_norm * (self.std_tg + self.eps) + self.mu_tg
