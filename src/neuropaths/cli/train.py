"""`neuropaths-train --config path/to.yaml` -- FNO training."""

from __future__ import annotations

import argparse

from neuropaths.config import load_config
from neuropaths.data import CommittorDataset, make_dataloader
from neuropaths.models import FNO2D
from neuropaths.training import train
from neuropaths.utils import describe_device, get_device, seed_everything


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train FNO2D on committor data.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seed_everything(cfg.data.seed, deterministic=cfg.train.deterministic)
    device = get_device(cfg.train.device)
    print(f"Using device: {describe_device(device)}")

    dataset = CommittorDataset(
        cfg.data.train_csv,
        grid_size=cfg.pde.coarse_grid,
        include_finv=cfg.data.include_finv_column,
    )
    loader = make_dataloader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )

    model = FNO2D(cfg.model)
    checkpoint = train(model, loader, cfg.train, device)
    print(f"Saved checkpoint to {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
