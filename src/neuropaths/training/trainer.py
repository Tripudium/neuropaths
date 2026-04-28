"""Training loop.

Wraps ``neuralop.Trainer`` (the reference FNO trainer) so the project's
TrainConfig drives it directly. Compared with the legacy
``Neural_operator/FNO_1.py`` loop this version:

    * trains on relative L2 (LpLoss(d=2, p=2)) -- the dissertation's
      eq. 11 metric, which the legacy code only used at evaluation;
    * evaluates on a held-out validation loader every epoch;
    * supports cosine / step LR schedules from TrainConfig;
    * saves the model via ``model.save_checkpoint`` (state_dict +
      construction kwargs as separate files) so it can be reloaded with
      ``FNO.from_checkpoint`` without needing the original ModelConfig.

``cfg.checkpoint_path`` is interpreted as a *path prefix*: the trainer
writes ``<parent>/<stem>_state_dict.pt`` and ``<parent>/<stem>_metadata.pkl``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from neuralop import LpLoss, Trainer
from torch.utils.data import DataLoader

from neuropaths.config import TrainConfig


def _build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: TrainConfig
) -> torch.optim.lr_scheduler.LRScheduler:
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.step_size, gamma=cfg.gamma
        )
    # neuralop's Trainer always calls scheduler.step(); use a constant LR
    # via StepLR with gamma=1.0 to keep the no-op case schedulers-aware.
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)


def _build_loss(cfg: TrainConfig) -> torch.nn.Module:
    if cfg.loss == "relative_l2":
        return LpLoss(d=2, p=2)
    if cfg.loss == "mse":
        return torch.nn.MSELoss()
    raise ValueError(f"Unsupported loss: {cfg.loss!r}")


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    *,
    val_loader: DataLoader | None = None,
) -> Path:
    """Train `model` per `cfg` and return the checkpoint path."""
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = _build_scheduler(optimizer, cfg)
    train_loss = _build_loss(cfg)
    eval_losses = {"l2": LpLoss(d=2, p=2)}

    test_loaders: dict[str, DataLoader] = {}
    if val_loader is not None:
        test_loaders["val"] = val_loader

    trainer = Trainer(
        model=model,
        n_epochs=cfg.epochs,
        device=device,
        verbose=True,
    )

    checkpoint_path = Path(cfg.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # When a val loader is wired, ask neuralop's Trainer to track the
    # best val_l2 across epochs and snapshot training state on every
    # improvement. Without this the saved checkpoint is whatever the
    # final epoch produced -- which on a cosine schedule is usually
    # past the val minimum.
    save_best = "val_l2" if val_loader is not None else None
    best_dir = checkpoint_path.parent / "_best"

    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=train_loss,
        eval_losses=eval_losses,
        save_best=save_best,
        save_dir=best_dir,
    )

    # If save_best was active, the in-memory model is the LAST epoch's
    # weights (overfit). Load the best-val snapshot back before writing
    # the canonical checkpoint files.
    best_state = best_dir / "best_model_state_dict.pt"
    if save_best is not None and best_state.exists():
        model.load_state_dict(
            torch.load(best_state, map_location=device, weights_only=False)
        )
        print(f"Loaded best-{save_best} weights from {best_state}.")

    # neuralop's BaseModel.save_checkpoint writes two files:
    #   <parent>/<stem>_state_dict.pt
    #   <parent>/<stem>_metadata.pkl
    model.save_checkpoint(checkpoint_path.parent, checkpoint_path.stem)
    return checkpoint_path
