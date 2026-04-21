"""Training loop.

Replaces the bare `train_model(...)` in Neural_operator/FNO_1.py. This
version:

    * takes an already-instantiated model, dataloader, config, device;
    * logs per-epoch mean loss (TODO: wire tqdm + optional W&B / TB);
    * saves a checkpoint to `train_cfg.checkpoint_path` when done;
    * supports cosine / step LR schedules from TrainConfig.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from neuropaths.config import TrainConfig
from neuropaths.training.losses import RelativeL2Loss


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    *,
    val_loader: DataLoader | None = None,
) -> Path:
    """Train `model` per `cfg` and return the checkpoint path."""
    # model.to(device)
    # optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # loss_fn = RelativeL2Loss() if cfg.loss == "relative_l2" else torch.nn.MSELoss()
    # scheduler = _build_scheduler(optimiser, cfg)
    # ...
    _ = RelativeL2Loss  # keep import; silence linter
    raise NotImplementedError(
        "TODO: port from Neural_operator/FNO_1.py train_model. Use cfg.loss "
        "(default relative_l2 per dissertation eq. 11), save checkpoint to "
        "cfg.checkpoint_path, return that Path."
    )
