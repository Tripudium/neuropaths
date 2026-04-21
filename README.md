# neuropaths

Learning transition paths in turbulent 2D domains with Fourier Neural
Operators. Implements the FNO pipeline from Higham's MA4K9 dissertation
*Learning Transition Paths in Turbulent Domains with Neural Operators*
(`papers/rproject.pdf`).

## What it does

Given a 2D domain `Omega` bounded by curves `x = phi(y)` (set A) and
`x = psi(y)` (set B), vertically periodic, carrying a divergence-free
turbulent drift `b`, the package learns the coefficient-to-solution map

    (b(x, y), f^{-1}(x, y))  ->  rho_react(x, y) = q^+(x, y) * q^-(x, y)

where `q^+`, `q^-` are the forward and backward committor functions
(probabilities of next/last visiting A vs B) and `rho_react` highlights
dynamical bottlenecks on transition paths from A to B.

## Layout

    src/neuropaths/
        pde/          # FD solvers, velocity fields, boundaries, transforms
        data/         # CSV generation + torch Dataset
        models/       # FNO2D (room for DeepONet, CNO, MGNO)
        training/     # trainer, relative-L2 loss, schedulers
        evaluation/   # metrics + prediction plots
        config/       # dataclass ExperimentConfig + YAML loader
        cli/          # neuropaths-{generate,train,evaluate}
        utils/        # device selection, seeding, IO
    configs/          # example experiment YAMLs (square_{16,32,63}, curved_32)
    tests/            # placeholder tests (one per submodule, all TODOs)
    paper/            # LaTeX notes (chapter-in-progress)
    papers/           # dissertation PDF

`Neural_operator/` (gitignored) holds the original research scripts; it
is a local reference during the port and will be removed once the
scaffold's `TODO: port from ...` markers are all resolved.

## Install

Requires Python 3.13 (pinned in `.python-version`) and [uv](https://docs.astral.sh/uv/):

    uv sync                      # creates .venv, installs deps from pyproject
    uv sync --extra dev          # include pytest / ruff / mypy

Without uv:

    python -m venv .venv && source .venv/bin/activate
    pip install -e '.[dev]'

## Run the experiments

The scaffold currently contains stubs with `TODO` markers pointing at
the legacy file that should be ported; once those are filled in:

    # 1. Generate data (FD solves on 311x311, subsample to coarse_grid):
    uv run neuropaths-generate --config configs/square_32.yaml --split train
    uv run neuropaths-generate --config configs/square_32.yaml --split test

    # 2. Train the FNO:
    uv run neuropaths-train    --config configs/square_32.yaml

    # 3. Evaluate + plot:
    uv run neuropaths-evaluate --config configs/square_32.yaml

Other configs:

  - `configs/square_16.yaml`, `configs/square_63.yaml` -- resolution sweep for
    Table 1 of the dissertation (Section 5.1).
  - `configs/curved_32.yaml`  -- 5000 PDEs with random trig-polynomial
    boundaries, `f^{-1}` fed as an extra input channel (Section 5.2).

## On Warwick's Blythe GPU cluster

`scrtp/` contains offline copies of the Blythe docs. The scaffold picks
CUDA automatically (`TrainConfig.device = "auto"`); force it with
`device: cuda` in the YAML to fail loudly on CPU-only nodes.

## Status

This is a scaffold. The working code still lives in `Neural_operator/`
and is imported/ported incrementally by following the `TODO: port
from ...` markers in each `src/neuropaths/*` module.
