#!/usr/bin/env python3
import hydra
from omegaconf import DictConfig
import logging


from src.dsp import preprocess_dataset


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    preprocess_dataset(cfg)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
