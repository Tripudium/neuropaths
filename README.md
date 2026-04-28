# neuropaths

Learning transition paths in turbulent 2D domains with Fourier Neural
Operators. Implements the FNO pipeline from Higham's MA4K9 dissertation
*Learning Transition Paths in Turbulent Domains with Neural Operators*.

## What it does

Given a 2D domain `Omega` bounded by curves `x = phi(y)` (set A) and
`x = psi(y)` (set B), vertically periodic, carrying a divergence-free
turbulent drift `b`, the package learns the coefficient-to-solution map

    (b_1(x, y), b_2(x, y), [f^{-1}(x, y)])  ->  rho_react(x, y) = q^+(x, y) * q^-(x, y)

where `q^+`, `q^-` are the forward and backward committor functions
(probabilities of next/last visiting A vs B) and `rho_react` is the
reactive trajectory density. The pipeline is:

1. **Generate data**: solve the committor PDEs by finite differences on
   a 311×311 grid for many random divergence-free velocity fields,
   subsample to a coarse grid (16/32/63), write a CSV.
2. **Train**: fit `neuralop.models.FNO` to the (b → rho_react) map with
   the relative-L2 loss (`neuralop.LpLoss`).
3. **Evaluate**: zero-shot super-resolution at the other grid sizes
   (Table 1 of the dissertation).

The (x, y) coordinates are *not* fed as input channels — neuralop's
grid positional embedding regenerates them at the evaluation
resolution, which is what makes zero-shot super-resolution well-defined.

## Layout

    src/neuropaths/
        pde/          # FD solvers, velocity fields, boundaries, transforms
        data/         # CSV generation + torch Dataset
        models/       # FNO builder around neuralop.models.FNO
        training/     # neuralop.Trainer wrapper + LpLoss
        evaluation/   # metrics + prediction plots
        config/       # dataclass ExperimentConfig + YAML loader
        cli/          # neuropaths-{generate,train,evaluate}
        utils/        # device selection, seeding, IO
    configs/          # experiment YAMLs (square_{16,32,63}, curved_32, smoke_test)
    slurm/            # SLURM job scripts for Warwick's Blythe cluster
    tests/            # pytest suite (PDE solvers + rejection sampling)
    paper/            # LaTeX notes
    papers/           # reference PDFs (dissertation, FNO, SDE)

## Installation

Requires Python 3.13 (pinned in `.python-version`) and
[uv](https://docs.astral.sh/uv/). The `neuraloperator` library is
declared as a dependency and resolved automatically.

To install uv on Avon, try
```bash
curl -LsSf https://astral.sh/uv/install.sh | env CARGO_HOME=~/.cargo sh
```
and then add to your `.bashrc`

    export PATH="/springbrook/share/maths/maskbg/.cargo/bin:$PATH"

Install the package with

```bash
git clone <this repo> neuropaths
cd neuropaths
uv sync --frozen        # creates .venv and installs from uv.lock
```

The first `uv sync` materialises a venv at `.venv/`. After that, all
commands run via `uv run <cmd>` so you don't need to activate the venv
explicitly. The reason we are doing it this way is because it seems more
convenient to have all the right packages available than having to use
what is available on Avon.

To pull in test/lint tooling:

```bash
uv sync --extra dev
uv run pytest
```

## Running locally

The pipeline is driven by a single YAML config that specifies the PDE
parameters, dataset size, model architecture, and training schedule.
See `configs/square_32.yaml` for the canonical setup (32×32 coarse
grid, 1000 train + 200 test PDE solutions, 100 epochs, FNO width 128).

For running on Avon or Blythe interactively, you might need to request 
and interactive shell, as outlined in 
[https://docs.scrtp.warwick.ac.uk/hpc-pages/hpc-interactive.html](https://docs.scrtp.warwick.ac.uk/hpc-pages/hpc-interactive.html)

### Generate data

```bash
uv run neuropaths-generate --config configs/square_32.yaml --split train --num-workers 8
uv run neuropaths-generate --config configs/square_32.yaml --split test  --num-workers 8
```

Outputs `runs/square_32/{train,test}.csv` with schema
`(solution_id, x, y, b1, b2, rho)`. `--num-workers` fans the
PDE solves out across CPU cores via `multiprocessing.Pool`; the CSV is
byte-identical regardless of worker count for a given seed.

**Rejection sampling.** Solutions whose committor product `q^+ · q^-`
is essentially zero everywhere (no detectable transitions) carry no
learning signal and detonate relative-L2 losses. The generator
discards them: it draws `ceil(N · oversample_factor)` candidates and
keeps the first `N` whose `rho.max() ≥ rho_min_max`. Defaults are
`rho_min_max = 0.01` and `oversample_factor = 1.5`. Acceptance stats
are printed at the end of each split.

### Train

```bash
uv run neuropaths-train --config configs/square_32.yaml
```

Trains via `neuralop.Trainer` with `LpLoss(d=2, p=2)` and Adam.
Validation loss is reported each epoch on `cfg.data.test_csv`.
Checkpoints are written through neuralop's `BaseModel.save_checkpoint`,
which produces two files at the path prefix:

    runs/square_32/fno2d_state_dict.pt   # clean state_dict
    runs/square_32/fno2d_metadata.pkl    # init kwargs (model can be rebuilt without the YAML)

Reload with:

```python
from pathlib import Path
from neuralop.models import FNO

ckpt = Path("runs/square_32/fno2d.pth")
model = FNO.from_checkpoint(ckpt.parent, ckpt.stem, map_location="cuda")
```

### Available configs

- `configs/smoke_test.yaml` — 20 train + 5 test, 5 epochs, FNO width 64.
  End-to-end pipeline check; runs in <10 min on a single L40 or M-series Mac.
- `configs/square_{16,32,63}.yaml` — resolution sweep for Table 1 of
  the dissertation (Section 5.1). All identity-domain (square).
- `configs/section2_laminar_32.yaml` — laminar variant (`velocity_kmax = 8`).
- `configs/curved_32.yaml` — 5000 PDEs with random trig-polynomial
  boundaries, `f^{-1}` fed as an extra input channel (Section 5.2).

### Evaluate (work-in-progress)

`neuropaths-evaluate --config <yaml>` is scaffolded but not yet
ported (see `cli/evaluate.py`); for now, use the two helper scripts
described below to inspect a trained run.

### Plot the training curve

`scripts/plot_training_curve.py` parses the per-epoch log lines that
`neuralop.Trainer(verbose=True)` emits and writes a train/val loss PNG
on a log y-scale.

```bash
uv run python scripts/plot_training_curve.py slurm/logs/neuropaths-train-12345.out
# -> slurm/logs/neuropaths-train-12345.png  (next to the log)

uv run python scripts/plot_training_curve.py runs/square_32/train.log -o curve.png
```

It looks for lines like `[N] ... train_err=...` and `Eval: val_l2=...`
and ignores everything else, so passing a full SLURM `.out` file works
even though it has setup noise interleaved.

### Inference on a held-out PDE

`scripts/inference_demo.py` regenerates one PDE solution with an
independent seed (so it cannot have leaked into train/test), runs it
through a checkpoint, and writes a 3-panel comparison PNG: the input
velocity magnitude `|b|`, the FD ground-truth `rho_react`, and the
FNO's prediction (with shared colour scale on the latter two so the
visual comparison is honest).

```bash
uv run python scripts/inference_demo.py --config configs/square_32.yaml
# -> runs/square_32/inference_demo.png

uv run python scripts/inference_demo.py \
    --config configs/square_32.yaml \
    --seed 1234 \
    --checkpoint runs/square_32/fno2d.pth \
    --output figs/demo_seed1234.png
```

The script prints per-sample relative L2 and mean absolute error so
you can compare to the train/val numbers from the training curve.
Pass `--seed` to look at multiple held-out draws; the default (999)
is fixed so the demo is reproducible.

## Running on Blythe (SLURM)

`scrtp/` holds offline copies of Warwick's Blythe HPC cluster docs;
the SLURM scripts in `slurm/` follow those conventions. There are
three scripts:

| script | partition | resources | purpose |
|---|---|---|---|
| `slurm/smoke_test.slurm` | gpu | 1 L40, 10 CPUs, 30 min | smoke-test the pipeline end-to-end on `configs/smoke_test.yaml` |
| `slurm/generate_full.slurm` | (default CPU) | 168 CPUs, 2 h | generate train + test for a full run |
| `slurm/train_full.slurm` | gpu | 1 L40, 10 CPUs, 4 h | train on already-generated data |

### One-time setup

1. Clone the repo somewhere you have write access. Anywhere works:
   `$HOME/neuropaths`, a project share, scratch — the SLURM scripts
   key off `${SLURM_SUBMIT_DIR}` (= the directory you `sbatch` from)
   so they're not tied to a specific path.
2. Materialise the venv once on a login node:

   ```bash
   cd path/to/neuropaths
   uv sync --frozen
   ```

3. Submit jobs from that same directory:

   ```bash
   cd path/to/neuropaths
   sbatch slurm/train_full.slurm
   ```

   Each script does `cd "${SLURM_SUBMIT_DIR}"` and activates `.venv/`
   relative to that, so subsequent runs start in seconds with no
   `uv sync` overhead per job.

If you ever change `pyproject.toml` or `uv.lock`, re-run `uv sync
--frozen` once to rebuild the venv before the next submission.

> **CUDA driver pin.** `pyproject.toml` constrains `torch` to
> `>=2.5,<2.11`. The 2.11 PyPI Linux wheel is built against CUDA ≥
> 12.6, which needs a newer NVIDIA driver than Blythe currently runs
> (driver 12.0.x → CUDA 12.0). Versions 2.5–2.10 ship CUDA wheels that
> work on the cluster's driver, so don't relax that upper bound
> without checking `nvidia-smi` on a GPU node first.

### Optional `slurm/env.sh` (per-machine overrides)

The SLURM scripts source `slurm/env.sh` if it happens to exist. The
file is gitignored so each clone can carry its own. You only need it
when you want to override defaults — for instance on Blythe where the
`$HOME` quota is tight and it's helpful to redirect caches into the
project directory, or when you want a shared MLflow tracking URI.

A typical Blythe-flavoured `slurm/env.sh`:

```bash
# Project root (defaults to ${SLURM_SUBMIT_DIR} if env.sh is absent).
export PROJECT_DIR="/springbrook/share/maths/maskbg/neuropaths"
export VENV_DIR="${PROJECT_DIR}/.venv"

# Redirect caches off $HOME.
CACHE_ROOT="${PROJECT_DIR}/.cache"
export UV_CACHE_DIR="${CACHE_ROOT}/uv"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TMPDIR="${CACHE_ROOT}/tmp/${SLURM_JOB_ID:-local}"
mkdir -p "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}" \
         "${MPLCONFIGDIR}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}" "${TMPDIR}"

# Optional: shared MLflow / wandb-offline.
export MLFLOW_TRACKING_URI="file:///springbrook/share/maths/maskbg/mlruns"
export WANDB_MODE=offline
```

On a machine where you don't need any of this (e.g. running from
`$HOME` on Avon with plenty of quota), just don't create the file and
the scripts use sensible defaults — `VENV_DIR=.venv`,
`PROJECT_DIR=${SLURM_SUBMIT_DIR}`, `$HOME`-default caches.

### Smoke test

Confirms the full generate→train pipeline works on Blythe before you
commit to a long run.

```bash
sbatch slurm/smoke_test.slurm
```

Logs land in `slurm/logs/neuropaths-smoke-<jobid>.{out,err}`.

### Full run (chained generate → train)

The recommended pattern: generation parallelizes well across many CPUs
(each PDE solve is a few seconds, embarrassingly parallel), training
needs a GPU. Submit them as two jobs with a SLURM dependency so the
GPU node only allocates after generation succeeds:

```bash
gen=$(sbatch --parsable slurm/generate_full.slurm)
sbatch --dependency=afterok:$gen slurm/train_full.slurm
```

By default both scripts use `configs/square_32.yaml`. Override with
the `CONFIG` env var:

```bash
gen=$(CONFIG=configs/curved_32.yaml sbatch --parsable slurm/generate_full.slurm)
sbatch --dependency=afterok:$gen --export=ALL,CONFIG=configs/curved_32.yaml slurm/train_full.slurm
```

### Notes on parallel generation

- The PDE solver is single-threaded (`scipy.sparse.linalg.spsolve` +
  numpy). The `--num-workers` flag uses `multiprocessing.Pool`, so
  speedup is near-linear in CPU count until you saturate the node.
- All four BLAS env vars (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`) are set to 1 in the
  SLURM scripts. This is **load-bearing**: without it, every worker
  fans BLAS calls out across the whole node, oversubscribing cores
  and slowing the run by an order of magnitude.
- Output is byte-identical regardless of worker count, so you can
  develop locally with `--num-workers 4` and reproduce the same data
  on a 168-CPU node.

## Tests

```bash
uv run pytest                              # full suite
uv run pytest tests/test_pde_solvers.py    # FD solver tests
uv run pytest tests/test_data.py           # generation + rejection sampling
```

The PDE solver tests cover: analytic recovery on `b=0`
(`q^+(x,y) = x`), Dirichlet/periodic boundary correctness, the
discrete maximum principle on turbulent fields, and forward/backward
consistency under drift reversal.

## Why neuralop

The original `Neural_operator/FNO_1.py` used real-weight Fourier
layers with `fftn` and no per-block skip connection — a (probably
unintended) simplification of Li et al. 2021 that capped relative L2
around 0.4-0.5. The current `models.fno2d.build_fno` is a thin
wrapper around `neuralop.models.FNO`, which uses complex weights with
`rfftn`, the proper `W_l` skip path, and supports all of neuralop's
layer factorisations. Training goes through `neuralop.Trainer` for
the same reason — battle-tested loop with checkpoint resume,
mixed-precision, etc.
