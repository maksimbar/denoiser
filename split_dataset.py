#!/usr/bin/env python3
import argparse
import random
import shutil
from pathlib import Path

RATIOS = (0.70, 0.15, 0.15)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split clean/noisy dataset 70-15-15.")
    p.add_argument("root", type=Path, help="Dataset root containing clean/ and noisy/")
    p.add_argument(
        "--seed", type=int, default=None, help="Seed for reproducible shuffle"
    )
    return p.parse_args()


def make_dirs(root: Path) -> None:
    for split in ("train", "valid", "test"):
        for sub in ("clean", "noisy"):
            (root / split / sub).mkdir(parents=True, exist_ok=True)


def copy_sentence(root: Path, split: str, base: str) -> None:
    """Copy one clean file and its 12 noisy variants to the split tree."""
    src_clean = root / "clean" / f"{base}.wav"
    dst_clean = root / split / "clean" / src_clean.name
    shutil.copy2(src_clean, dst_clean)

    src_noisy_dir = root / "noisy"
    dst_noisy_dir = root / split / "noisy"
    for wav in src_noisy_dir.glob(f"{base}_w*.wav"):
        shutil.copy2(wav, dst_noisy_dir / wav.name)


def main() -> None:
    args = parse_args()
    root: Path = args.root.resolve()
    random.seed(args.seed)

    clean_dir = root / "clean"
    noisy_dir = root / "noisy"
    if not (clean_dir.is_dir() and noisy_dir.is_dir()):
        raise SystemExit("Expected clean/ and noisy/ inside the provided root.")

    bases = sorted(p.stem for p in clean_dir.glob("*.wav"))
    if not bases:
        raise SystemExit("No .wav files found in clean/")

    random.shuffle(bases)

    total = len(bases)
    n_train = int(total * RATIOS[0])
    n_valid = int(total * RATIOS[1])
    n_test = total - n_train - n_valid

    splits = {
        "train": bases[:n_train],
        "valid": bases[n_train : n_train + n_valid],
        "test": bases[n_train + n_valid :],
    }

    make_dirs(root)
    for split, subset in splits.items():
        for base in subset:
            copy_sentence(root, split, base)

    print(
        f"✓ Done: {root}\n"
        f"  Train: {n_train:>4} sentences\n"
        f"  Valid: {n_valid:>4} sentences\n"
        f"  Test : {n_test:>4} sentences"
    )


if __name__ == "__main__":
    main()
