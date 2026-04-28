"""Generate one held-out PDE solution and compare against a checkpoint.

Usage:

    uv run python scripts/inference_demo.py --config configs/square_32.yaml
    uv run python scripts/inference_demo.py --config configs/square_32.yaml \
        --seed 999 --output runs/square_32/inference_demo.png

Pipeline:

    1. Re-derive the same FD pipeline the generator uses (turbulent
       velocity field -> q^+ / q^- -> rho_react -> coarse subsample).
       A user-supplied seed (default 999) keeps this draw distinct from
       any training/test sample.
    2. Load the trained FNO via ``neuralop.models.FNO.from_checkpoint``
       (so the model rebuilds itself from the saved init kwargs and
       doesn't need ModelConfig).
    3. Run a forward pass on the coarse-grid input.
    4. Render a 3-panel figure: |b| (input), ground-truth rho, predicted
       rho. Ground truth and prediction share a colour scale so their
       difference is visually honest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from neuralop.models import FNO

from neuropaths.config.loader import load_config
from neuropaths.data.generator import _generate_one_solution, _SolutionJob


def _input_target_from_rows(
    rows: list[list[float]], coarse_grid: int, include_finv: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Reshape a generator's row list into (C, G, G) input + (G, G) rho."""
    arr = np.asarray(rows, dtype=np.float32)
    # Columns: solution_id, x, y, b1, b2, rho, [finv]
    # The generator emits rows in (i, j) lexicographic order on the
    # transformed coarse grid, matching meshgrid(indexing='ij').
    G = coarse_grid
    arr = arr.reshape(G, G, -1)
    b1 = arr[..., 3]
    b2 = arr[..., 4]
    rho = arr[..., 5]
    channels = [b1, b2]
    if include_finv:
        channels.append(arr[..., 6])
    inp = np.stack(channels, axis=0)  # (C, G, G)
    return inp, rho


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run a held-out PDE through a trained FNO and plot.")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=999,
                   help="Independent seed for this demo PDE (default 999).")
    p.add_argument("--checkpoint", default=None,
                   help="Override checkpoint path prefix (default: cfg.eval.checkpoint_path).")
    p.add_argument("--output", type=Path, default=None,
                   help="Output PNG path (default: <output_dir>/inference_demo.png).")
    args = p.parse_args(argv)

    cfg = load_config(args.config)

    # Reuse the generator's worker so the data pipeline is identical to
    # what the model saw during training.
    seed_seq = np.random.SeedSequence(args.seed)
    job = _SolutionJob(
        sid=0, pde_cfg=cfg.pde, data_cfg=cfg.data, seed_seq=seed_seq
    )
    print(f"Solving demo PDE (seed={args.seed}) at fine_grid={cfg.pde.fine_grid}...")
    _, rows, rho_max = _generate_one_solution(job)
    print(f"  rho.max() = {rho_max:.4f}")

    inp_np, target_np = _input_target_from_rows(
        rows, cfg.pde.coarse_grid, cfg.data.include_finv_column
    )

    # Load the trained model. neuralop's BaseModel.from_checkpoint takes
    # (folder, name) and rebuilds the architecture from the metadata file.
    ckpt_path = Path(args.checkpoint or cfg.eval.checkpoint_path)
    print(f"Loading checkpoint from prefix {ckpt_path}...")
    model = FNO.from_checkpoint(ckpt_path.parent, ckpt_path.stem, map_location="cpu")
    model.eval()

    with torch.no_grad():
        inp_t = torch.from_numpy(inp_np).unsqueeze(0)  # (1, C, G, G)
        pred_t = model(inp_t)
    pred_np = pred_t.squeeze().cpu().numpy()

    rel_l2 = np.linalg.norm(pred_np - target_np) / max(
        np.linalg.norm(target_np), 1e-12
    )
    abs_err = np.abs(pred_np - target_np).mean()
    print(f"Demo metrics: rel L2 = {rel_l2:.4f}  mean |err| = {abs_err:.4f}")

    # Render a three-panel comparison. Share the colour scale between
    # ground truth and prediction so the eye isn't misled.
    bmag = np.sqrt(inp_np[0] ** 2 + inp_np[1] ** 2)
    vmin = float(min(target_np.min(), pred_np.min()))
    vmax = float(max(target_np.max(), pred_np.max()))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)
    extent = (0.0, 1.0, 0.0, 1.0)

    im0 = axes[0].imshow(bmag.T, origin="lower", extent=extent, cmap="viridis")
    axes[0].set_title(r"$|b|$ (input magnitude)")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)

    im1 = axes[1].imshow(target_np.T, origin="lower", extent=extent,
                         cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[1].set_title(r"$\rho_{\mathrm{react}}$ (FD ground truth)")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)

    im2 = axes[2].imshow(pred_np.T, origin="lower", extent=extent,
                         cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[2].set_title(rf"$\hat\rho$  (rel L2 = {rel_l2:.3f})")
    fig.colorbar(im2, ax=axes[2], shrink=0.85)

    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")

    out = args.output or (Path(cfg.output_dir) / "inference_demo.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
