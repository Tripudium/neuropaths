"""Tests for neuropaths.pde.boundaries.

TODO:
    * identity_boundaries(): phi(y) == 0, psi(y) == 1 for any y in [0,1].
    * random_trig_boundaries(): phi(0) == phi(1) == 0, psi(0) == psi(1) == 1
      (top/bottom y-periodicity requirement).
    * random_trig_boundaries(): min_y |phi(y) - psi(y)| >= eps.
    * Derivatives via autograd.elementwise_grad at y=0, 0.5, 1 are finite.
"""
