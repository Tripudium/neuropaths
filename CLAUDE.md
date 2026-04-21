# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository shape

Two independent sub-projects share this repo; they have no build coupling.

- `Neural_operator/` — Python research code. Trains a Fourier Neural Operator (FNO2D) to approximate the committor function `ρ = q⁺ · q⁻` of a 2D advection–diffusion problem on `[0,1]²` with a random turbulent velocity field. Finite-difference PDE solutions on a fine 311×311 grid are subsampled (default 32×32) to produce supervised training pairs.
- `paper/` — LaTeX paper (`amsart`) built with pdflatex + bibtex via `latexmk`. Contains `main.tex`, `macros.tex`, plus `sections/` and `refs/` (both currently empty) and an `Operator_learning_chapter/` draft.

Other top-level dirs: `papers/` (reference PDFs), `scrtp/` (offline HTML of Warwick's Blythe HPC cluster docs — target deployment environment), `scripts/` and `slurm/` (empty placeholders). `README.md` is empty; `pyproject.toml` declares no dependencies despite the Python code needing several — there is no installable package yet.

## Python (Neural_operator/)

No package metadata, no test suite, no lockfile. Runtime deps must be installed manually: `torch`, `numpy`, `pandas`, `scipy`, `autograd`, `scikit-learn`, `matplotlib`. `.python-version` pins 3.13.

Pipeline (run from inside `Neural_operator/`):

```bash
python training_data_1_generator.py   # writes train_data_32_square_2500_minv80_up.csv
python FNO_1.py                       # trains, writes train_data_32_square_2500_minv80_Test.pth
python FNO_1_Test.py                  # loads .pth, writes prediction_comparison.png etc.
```

File paths, grid size, number of solutions, and output filenames are **hardcoded in each `main()`** — edit the module to change them. The CSV schema is `solution_id,x,y,b1,b2,rho`, grouped by `solution_id`. Device is `mps` if available else `cpu` (no CUDA path is wired up — add one before running on Blythe GPU nodes).

**Gotcha: module-level `main()` calls.** `FNO_1.py` and `training_data_1_generator.py` end with a bare `main()` (no `if __name__ == "__main__"` guard). `FNO_1_Test.py` does `from FNO_1 import FNO2D, PDEOperatorDataset`, which will trigger a full training run on import. Either guard `main()` before importing from these modules, or split the model/dataset classes into a separate file.

### Code dependency map

- `training_data_1_generator.py` → `Finite_difference_forward_comittor.py` (solves `q⁺`), `Finite_difference_backwards_committor.py` (solves `q⁻`), `turbulent_velocity_field.py` (random `b_x, b_y` via FFT-based synthesis on a 311×311 grid, returned as `RegularGridInterpolator`s). `Trig_polynomial_boundary.py` exists but is commented out — boundaries are currently the identity (`phi=0, psi=1`).
- FD solvers use the `autograd` library (not `torch.autograd`) for coordinate-transform derivatives; keep arrays as `autograd.numpy` when they flow through those functions.

## Paper (paper/)

`cd paper` then use the Makefile. Engine is **pdflatex + bibtex** (not biber — `latexmkrc` sets this explicitly). Bibliography lives at `bib/refs.bib` per `main.tex`, though the `refs/` directory in the repo is currently empty.

```bash
make           # == make draft  (latexmk -pdf, nonstop)
make final     # halts on first error
make watch     # latexmk -pvc continuous rebuild
make clean     # latexmk -C plus .bbl, .run.xml
make lint      # chktex -n1 -n8 -n46 on main.tex and sections/*.tex
make format    # latexindent -w (config in .latexindent.yaml: 2-space, 100-col wrap)
make arxiv     # clean + stage a submission tree under ./arxiv/
MINTED=1 make  # re-run with -shell-escape for minted
```

Previewer is Skim (macOS). `.editorconfig` enforces LF, UTF-8, 2-space indent, trim trailing whitespace (LaTeX line length unrestricted).
