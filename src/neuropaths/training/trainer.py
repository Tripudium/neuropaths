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
from neuralop import H1Loss, LpLoss, Trainer
from torch.utils.data import DataLoader

from neuropaths.config import TrainConfig


class _AbsLpLoss:
    """Adapter exposing ``LpLoss.abs`` through the standard callable API.

    neuralop's ``LpLoss.__call__`` returns ``rel`` regardless; ``abs`` is
    only reachable via the method directly. We wrap it so the Trainer can
    call ``loss(out, **sample)`` uniformly.
    """

    def __init__(self, lp_loss: LpLoss) -> None:
        self.lp_loss = lp_loss

    def __call__(self, y_pred, y, **kwargs):
        return self.lp_loss.abs(y_pred, y)


class _MSELossKW:
    """``torch.nn.MSELoss`` adapter that accepts (and ignores) extra kwargs.

    neuralop's Trainer calls the loss as ``loss(out, **sample)`` which
    passes both ``x`` and ``y`` as keyword arguments. neuralop's losses
    swallow unknown kwargs; ``nn.MSELoss`` does not, so wrap it.
    """

    def __init__(self) -> None:
        self.mse = torch.nn.MSELoss()

    def __call__(self, y_pred, y, **kwargs):
        return self.mse(y_pred, y)


@torch.no_grad()
def _final_eval(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Compute summary metrics on the validation set after training.

    Returns four metrics that together pin down a checkpoint's quality:

    * ``per_sample_mean_rel_l2``: mean of ``||p_i - y_i||_2 / ||y_i||_2``
      across val samples. This is what neuralop's Trainer reports as
      ``val_l2`` each epoch.
    * ``global_rel_l2``: ``||P - Y||_2 / ||Y||_2`` with P, Y the full
      flattened test-set tensors. Matches the dissertation's
      ``relative_l2_test_error`` (FNO_1_Test.py line 102) — weights
      samples by ``||y_i||_2``, so it is structurally immune to the
      small-target singularity that inflates the per-sample mean.
    * ``mse`` / ``mae``: pointwise on the same data, useful when targets
      are bounded (here rho ∈ [0, 1]).
    """
    model.eval()
    sq_diff = 0.0
    sq_y = 0.0
    abs_diff = 0.0
    n_cells = 0
    sum_per_sample_rel_l2 = 0.0
    n_samples = 0
    for sample in val_loader:
        x = sample["x"].to(device)
        y = sample["y"].to(device)
        out = model(x)
        diff = (out - y).flatten(1)
        y_flat = y.flatten(1)
        sq_diff += (diff * diff).sum().item()
        sq_y += (y_flat * y_flat).sum().item()
        abs_diff += diff.abs().sum().item()
        n_cells += diff.numel()
        per_sample_rel = diff.norm(dim=1) / (y_flat.norm(dim=1) + 1e-8)
        sum_per_sample_rel_l2 += per_sample_rel.sum().item()
        n_samples += y.shape[0]
    return {
        "per_sample_mean_rel_l2": sum_per_sample_rel_l2 / max(n_samples, 1),
        "global_rel_l2": (sq_diff / sq_y) ** 0.5,
        "mse": sq_diff / max(n_cells, 1),
        "mae": abs_diff / max(n_cells, 1),
    }


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


def _build_loss(cfg: TrainConfig):
    """Build the loss callable named in `cfg.loss`.

    For relative_l2 / h1, an additive ``loss_eps`` (>0) caps the
    amplification when ``||y||_2`` is small. Defaults to 0.0, in which
    case neuralop's internal default of 1e-8 is used.
    """
    eps = max(float(cfg.loss_eps), 1e-8)
    if cfg.loss == "relative_l2":
        return LpLoss(d=2, p=2, eps=eps)
    if cfg.loss == "absolute_l2":
        return _AbsLpLoss(LpLoss(d=2, p=2))
    if cfg.loss == "h1":
        # Committor BCs: Dirichlet in x, periodic in y. Telling the
        # loss about non-periodicity in x avoids spurious gradient
        # contributions across the Dirichlet boundary.
        return H1Loss(d=2, eps=eps, periodic_in_x=False, periodic_in_y=True)
    if cfg.loss == "mse":
        return _MSELossKW()
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

    # Final summary on the val set with the loaded best weights, including
    # the dissertation's flattened global rel L2 alongside neuralop's
    # per-sample mean.
    if val_loader is not None:
        metrics = _final_eval(model, val_loader, device)
        print("Final val metrics (best-checkpoint weights):")
        for name, value in metrics.items():
            print(f"  {name:<25}= {value:.4f}")

    # neuralop's BaseModel.save_checkpoint writes two files:
    #   <parent>/<stem>_state_dict.pt
    #   <parent>/<stem>_metadata.pkl
    model.save_checkpoint(checkpoint_path.parent, checkpoint_path.stem)
    return checkpoint_path
