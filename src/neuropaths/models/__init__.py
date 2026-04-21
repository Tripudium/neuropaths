"""Neural operator architectures.

Currently:
    * FNO2D -- dissertation's main model (sections 2.3 + 5).

Room for:
    * DeepONet -- the notes chapter compares against this baseline but
      no code exists yet (notes quote relative L2 = 0.3376 laminar,
      0.5138 turbulent for DeepONet; dissertation doesn't implement it).
    * Convolutional Neural Operator -- "in testing" per the notes.
    * Multipole Graph Neural Operator -- listed as future work.
"""

from neuropaths.models.fno2d import FNO2D, FourierLayer2D

__all__ = ["FNO2D", "FourierLayer2D"]
