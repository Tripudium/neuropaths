"""neuropaths -- operator learning for transition paths in turbulent domains.

The package learns the coefficient-(and-domain)-to-solution map

    (b(x, y), f^{-1}(x, y)) -> rho_react(x, y),

where

    - b is a 2D turbulent, divergence-free, y-periodic velocity field;
    - f maps a curved domain Omega to the unit square [0,1]^2 via
      f(x, y) = (x - phi(y)) / (psi(y) - phi(y));
    - rho_react = q^+ * q^- is the reactive probability density built
      from the forward and backward committor functions, which solve
      two (transformed) steady advection-diffusion PDEs with Dirichlet
      conditions in x and periodic conditions in y.

This follows Higham's MA4K9 dissertation "Learning Transition Paths in
Turbulent Domains with Neural Operators" (papers/rproject.pdf).

Submodules:

    neuropaths.pde         -- FD solvers, velocity fields, boundary params
    neuropaths.data        -- dataset generation and torch Dataset/DataLoader
    neuropaths.models      -- FNO2D (and room for DeepONet, CNO)
    neuropaths.training    -- training loops, losses, schedulers
    neuropaths.evaluation  -- metrics, plotting, resolution studies
    neuropaths.config      -- dataclass-based experiment configs + YAML I/O
    neuropaths.utils       -- device selection, seeding, IO helpers
    neuropaths.cli         -- console_scripts entry points
"""

__version__ = "0.1.0"
