import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import time
import logging
import pandas as pd

from src.dsp import preprocess_dataset
from src.data import SpeechDataset
from src.model import RCED
from src.evaluate import evaluate


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Running preprocessing...")
    start_time = time.time()
    preprocess_dataset(cfg)
    duration = time.time() - start_time
    cache_dir = Path(cfg.data.cache).expanduser().resolve()
    logger.info(f"Preprocessing finished in {duration:.2f}s.")

    logger.info("Loading datasets...")
    train_dataset = SpeechDataset(split="train", cache_dir=cache_dir)
    valid_dataset = SpeechDataset(split="valid", cache_dir=cache_dir)
    test_dataset = SpeechDataset(split="test", cache_dir=cache_dir)
    logger.info("Datasets loaded.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        num_workers=4 if device.type == "cuda" else 0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=4 if device.type == "cuda" else 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=4 if device.type == "cuda" else 0,
    )

    model = RCED(
        filters=list(cfg.model.filters),
        kernels=list(cfg.model.kernels),
        in_ch=cfg.stft.input_frames,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized with {num_params:,} parameters.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.train.optimizer.lr,
        betas=tuple(cfg.train.optimizer.betas),
        eps=cfg.train.optimizer.eps,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.train.lr_scheduler.factor,
        patience=cfg.train.lr_scheduler.patience,
        min_lr=cfg.train.lr_scheduler.min_lr,
    )

    best_valid_loss = float("inf")
    best_model_path = output_dir / "best_model.pt"
    train_history = []

    logger.info("Starting training...")
    for epoch in range(cfg.train.epochs):
        model.train()
        train_loss_total = 0.0
        epoch_start_time = time.time()

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
        avg_train_loss = train_loss_total / len(train_loader)

        model.eval()
        valid_loss_total = 0.0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                valid_loss_total += loss.item()
        avg_valid_loss = valid_loss_total / len(valid_loader)

        scheduler.step(avg_valid_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_duration = time.time() - epoch_start_time

        logger.info(
            f"Epoch {epoch + 1:03d}/{cfg.train.epochs}: "
            f"Tr L={avg_train_loss:.4f} | Va L={avg_valid_loss:.4f} | "
            f"LR={current_lr:.1e} | Time={epoch_duration:.2f}s"
        )

        train_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "valid_loss": avg_valid_loss,
                "lr": current_lr,
            }
        )

        if avg_valid_loss < best_valid_loss:
            logger.info(
                f"  Validation loss improved ({best_valid_loss:.6f} -> {avg_valid_loss:.6f}). Saving model..."
            )
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), best_model_path)

    logger.info("Training finished.")

    history_df = pd.DataFrame(train_history)
    history_csv_path = output_dir / "losses.csv"
    history_df.to_csv(history_csv_path, index=False)

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    model.eval()
    test_loss_total = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss_total += loss.item()
    avg_test_loss = test_loss_total / len(test_loader)
    logger.info(f"Final Test Set Loss (MSE) = {avg_test_loss:.6f}")

    results_overall, results_per_snr = evaluate(model, cfg, device, logger)

    overall_metrics_list = []
    for metric, data in results_overall.items():
        overall_metrics_list.append(
            {"metric": metric, "value": data["mean"], "num_files": data["count"]}
        )
    overall_df = pd.DataFrame(overall_metrics_list)
    overall_csv_path = output_dir / "metrics_overall.csv"
    overall_df.to_csv(overall_csv_path, index=False, float_format="%.4f")

    per_snr_df = pd.DataFrame(results_per_snr)
    per_snr_df = per_snr_df[
        ["snr_level", "avg_pesq", "avg_stoi", "avg_sdr_mag", "num_files"]
    ]
    per_snr_csv_path = output_dir / "metrics_per_snr.csv"
    per_snr_df.to_csv(per_snr_csv_path, index=False, float_format="%.4f")

    logger.info("Run finished.")


if __name__ == "__main__":
    main()
