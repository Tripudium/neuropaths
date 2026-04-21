"""Tests for neuropaths.pde.solvers.

TODO:
    * Trivial case b = 0, phi = 0, psi = 1: q^+(x, y) = x analytically
      (1D Laplace with Dirichlet 0/1). The FD solver should recover
      this to second-order accuracy in h.
    * Symmetric case b = (B, 0) constant, phi=0, psi=1: q^+ has a known
      exponential profile; check rate of convergence.
    * Periodicity check: q^+(x, 0) == q^+(x, 1) on the discrete grid.
    * Boundary sanity: q^+(0, y) == 0 and q^+(1, y) == 1 exactly.
    * Cross-validation: solve_backward_committor(-b) should agree with
      solve_forward_committor(b) up to the boundary flip.
"""
