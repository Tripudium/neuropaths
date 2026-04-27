"""Neural operator architectures.

Currently:
    * FNO2D / build_fno -- 2D Fourier Neural Operator built via
      neuraloperator's ``FNO`` (Li et al. 2021), driven by ModelConfig.
"""

from neuropaths.models.fno2d import FNO2D, build_fno

__all__ = ["FNO2D", "build_fno"]
