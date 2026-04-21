"""PDE side of the codebase.

Contents:
    * solvers      -- FD solvers for forward/backward committor PDEs
    * velocity     -- synthetic k^{-5/3} turbulent velocity fields
    * boundaries   -- trig-polynomial boundary parametrisations (phi, psi)
    * transforms   -- coordinate map Omega <-> [0,1]^2 and its derivatives

Mathematical claims (dissertation sections 3-4):
    * Divergence-free, y-periodic drift b (turbulence algorithm, 4.2.2).
    * Steady Fokker-Planck reduces to a backward Kolmogorov problem
      for q^+ and a forward Kolmogorov problem for q^-, both with
      Dirichlet x-BCs (0/1 and 1/0) and periodic y-BCs (eqs. 16, 19).
    * Theorem 4.1 transforms the PDE under (x,y) -> (f(x,y), y) with
      f(x, y) = (x - phi(y)) / (psi(y) - phi(y)); the result has no
      cross-derivative mixing except through f_y (eq. 33).
    * rho_steady = 1 / |Omega| => rho_react = q^+ * q^- / |Omega|
      (eq. 18); legacy code takes rho_react = q^+ * q^- i.e. it drops
      the |Omega| factor. Flag when porting.
"""
