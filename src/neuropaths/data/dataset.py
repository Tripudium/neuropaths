"""PyTorch Dataset for the committor training CSVs.

Each CSV row is a single coarse-grid sample (x, y, b1, b2, rho[, finv])
tagged by solution_id. The Dataset groups by solution_id once in
__init__ and yields, per sample, a dict matching what
``neuralop.Trainer`` expects (batch keys "x" and "y"):

    {
        "x": (C, G, G) float32 tensor,
              C = 2 (b1, b2)        for square domains
              C = 3 (b1, b2, finv)  for curved domains
        "y": (1, G, G) float32 tensor,  # rho_react on the coarse grid
    }

The (x, y) coordinate columns are *not* loaded as channels: neuralop's
FNO appends a grid positional embedding internally (and re-instantiates
it at the evaluation resolution, which is what makes zero-shot
super-resolution well-defined). They are still used to deterministically
order rows within a solution group.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class CommittorDataset(Dataset):
    """Reads a committor CSV once, caches the per-solution tensors."""

    def __init__(
        self,
        csv_path: str | Path,
        *,
        grid_size: int,
        include_finv: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.grid_size = int(grid_size)
        self.include_finv = bool(include_finv)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        required = {"solution_id", "x", "y", "b1", "b2", "rho"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV {self.csv_path} missing columns: {sorted(missing)}")
        if self.include_finv and "finv" not in df.columns:
            raise ValueError(
                f"include_finv=True but CSV {self.csv_path} has no 'finv' column."
            )

        expected = self.grid_size * self.grid_size
        groups = list(df.groupby("solution_id", sort=True))

        self._inputs: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []
        for sid, group in groups:
            if len(group) != expected:
                raise ValueError(
                    f"solution_id={sid} has {len(group)} rows, expected {expected} "
                    f"for grid_size={self.grid_size}. CSV {self.csv_path} may be "
                    f"from a different coarse_grid."
                )

            # Match the generator's meshgrid(indexing='ij'): rows ordered
            # by (x ascending, y ascending). The (x, y) columns are used
            # only for ordering; they are not loaded as input channels.
            group = group.sort_values(["x", "y"], kind="mergesort").reset_index(drop=True)
            G = self.grid_size

            def reshape(col: str) -> np.ndarray:
                return group[col].to_numpy(dtype=np.float32).reshape(G, G)

            channels = [reshape("b1"), reshape("b2")]
            if self.include_finv:
                channels.append(reshape("finv"))
            inp = np.stack(channels, axis=0)  # (C, G, G)
            tgt = reshape("rho")[None, :, :]  # (1, G, G)

            self._inputs.append(torch.from_numpy(inp))
            self._targets.append(torch.from_numpy(tgt))

    def __len__(self) -> int:
        return len(self._inputs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x": self._inputs[idx], "y": self._targets[idx]}


def make_dataloader(
    dataset: CommittorDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
