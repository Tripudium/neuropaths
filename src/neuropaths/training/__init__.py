"""Training loops, losses, schedulers."""

from neuropaths.training.losses import RelativeL2Loss
from neuropaths.training.trainer import train

__all__ = ["RelativeL2Loss", "train"]
