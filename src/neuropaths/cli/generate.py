"""`neuropaths-generate --config path/to.yaml` -- dataset generation."""

from __future__ import annotations

import argparse

import numpy as np

from neuropaths.config import load_config
from neuropaths.data import generate_dataset
from neuropaths.utils import seed_everything


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate committor training data.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML.")
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="Which split to generate; picks data.train_csv or data.test_csv.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    # Seed the globals (numpy / torch / random) for any ambient call;
    # we also pass an explicit np.random.Generator into the generator so
    # the sampling itself is reproducible independent of global state.
    seed_everything(cfg.data.seed)
    split_offset = 0 if args.split == "train" else 1
    rng = np.random.default_rng(cfg.data.seed + split_offset)

    output = cfg.data.train_csv if args.split == "train" else cfg.data.test_csv
    path = generate_dataset(
        cfg.pde,
        cfg.data,
        split=args.split,
        output_path=output,
        rng=rng,
    )
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
