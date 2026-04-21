"""Dataclass schemas for experiment configs.

Designed to exactly cover the hyperparameters that appear hardcoded in
the legacy scripts plus the knobs the dissertation sweeps over:

    * Section 5 "Training the Neural Operator" -- 1000 / 5000 PDE
      instances, 100 epochs, batch sizes 32/64, k_max in {12, 16},
      d_c in {128, 256}, input channel width 256, lr = 0.001,
      grid sizes s in {16, 32, 63} for training and s' for testing.
    * Section 4 "Generating Training Data" -- fine FD grid 311x311,
      Reynolds = 10, L = 0.2, nu = 1 (note: legacy code uses Re=8).
    * Section 4.1 boundary generator -- N_max = 8, epsilon tolerance.

Defaults here track the dissertation's canonical "square 32x32" run so
that `ExperimentConfig()` without overrides reproduces the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DomainKind = Literal["square", "curved"]
DeviceStr = Literal["auto", "cuda", "mps", "cpu"]


@dataclass
class PDEConfig:
    """FD solver + velocity-field parameters."""

    # Fine FD grid used to solve the committor PDEs before subsampling.
    # Dissertation fixes this to 311x311 so 16/32/63 subsample cleanly.
    fine_grid: int = 311

    # Coarse grid the dataset is subsampled onto; also the FNO input grid.
    coarse_grid: int = 32

    # Turbulence: Reynolds number, characteristic length, viscosity.
    # Dissertation picks Re=10, L=0.2, nu=1 (=> U = nu*Re/L = 50).
    # Legacy `training_data_1_generator.py` passes Reynolds=8.0 -- flag.
    reynolds: float = 10.0
    char_length: float = 0.2
    viscosity: float = 1.0

    # Whether to use curved domains (random trig-polynomial boundaries)
    # or the identity boundaries phi=0, psi=1 that the legacy code runs.
    domain: DomainKind = "square"

    # Trig-polynomial boundary generator (section 4.1).
    boundary_n_max: int = 8
    boundary_eps: float = 0.1

    # Epsilon neighbourhood padding used by turbulent_velocity_field.
    eps_1: float = 0.0
    eps_2: float = 0.0


@dataclass
class DataConfig:
    """Dataset generation + loader config."""

    num_train_solutions: int = 1000
    num_test_solutions: int = 200

    train_csv: str = "data/train.csv"
    test_csv: str = "data/test.csv"

    # CSV column layout produced by the generator. Matches legacy schema
    # ['solution_id', 'x', 'y', 'b1', 'b2', 'rho']; extended by 'finv'
    # for the curved-domain experiment.
    include_finv_column: bool = False

    batch_size: int = 32
    num_workers: int = 0  # keep 0 by default; DataLoader workers interact
    # poorly with some MPS setups.

    # PRNG seed -- passed to seed_everything and the velocity-field draws.
    seed: int = 2203_2025


@dataclass
class ModelConfig:
    """FNO2D architecture."""

    # Number of input channels fed to the lifting layer R.
    # Square domain: (x, y, b1, b2) -> 4.
    # Curved domain: (x, y, b1, b2, f^{-1}) -> 5.
    in_channels: int = 4
    out_channels: int = 1

    # Fourier mode truncation k_max (same in both spatial dims).
    modes: int = 12

    # Fourier channel width d_c.
    width: int = 128

    # Depth L (number of Fourier layers).
    depth: int = 4

    # Projection layer hidden width (Q).
    projection_hidden: int = 128

    activation: Literal["gelu", "relu"] = "gelu"


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0

    # Dissertation uses relative L2 loss (eq. 10) for training and eval.
    # Legacy `FNO_1.py` uses MSELoss -- flag this as a reconciliation task.
    loss: Literal["relative_l2", "mse"] = "relative_l2"

    scheduler: Literal["none", "cosine", "step"] = "none"
    step_size: int = 30
    gamma: float = 0.5

    # Checkpoint output.
    checkpoint_path: str = "checkpoints/fno2d.pth"

    # Device.
    device: DeviceStr = "auto"
    deterministic: bool = False


@dataclass
class EvalConfig:
    checkpoint_path: str = "checkpoints/fno2d.pth"
    output_dir: str = "results"

    # Zero-shot super-resolution study (Table 1/2 in the dissertation):
    # train at `s`, evaluate at each of these.
    test_resolutions: list[int] = field(default_factory=lambda: [16, 32, 63])

    # Number of PDEs visualised in the qualitative prediction plots.
    num_plot_samples: int = 3


@dataclass
class ExperimentConfig:
    """Top-level config for a single reproducible experiment."""

    name: str = "square_32_baseline"
    output_dir: str = "runs/${name}"

    pde: PDEConfig = field(default_factory=PDEConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
