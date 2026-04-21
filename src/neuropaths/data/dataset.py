"""PyTorch Dataset / DataLoader for the committor CSVs.

The legacy `PDEOperatorDataset` in `Neural_operator/FNO_1.py` groups by
`solution_id`, extracts `[x, y, b1, b2]` and the target `rho`, then
reshapes into `(4, G, G)` and `(G, G)` tensors. Reproduce that here,
but with:

    * explicit `grid_size` validated against row count per group,
    * optional `f^{-1}` column for the curved-domain experiment,
    * pre-computed grouping (load once, index many times) to avoid
      the pandas groupby overhead on every `__getitem__`.

TODO: port from Neural_operator/FNO_1.py `PDEOperatorDataset` +
`load_data`, keeping the FNO-friendly tensor layout [C, H, W].
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class CommittorDataset(Dataset):
    """Each item: (input_fields[C, G, G], target_rho[G, G])."""

    def __init__(
        self,
        csv_path: str | Path,
        *,
        grid_size: int,
        include_finv: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.grid_size = grid_size
        self.include_finv = include_finv
        raise NotImplementedError(
            "TODO: port from Neural_operator/FNO_1.py PDEOperatorDataset. "
            "Load CSV once, group by solution_id, cache tensors."
        )

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


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
