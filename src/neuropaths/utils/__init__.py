"""Cross-cutting utilities: device selection, seeding, IO."""

from neuropaths.utils.device import get_device, describe_device
from neuropaths.utils.seeding import seed_everything

__all__ = ["get_device", "describe_device", "seed_everything"]
