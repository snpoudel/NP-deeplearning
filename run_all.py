"""Orchestrator: trains all 6 models for every seed in SEEDS, aggregates
predictions and loss curves across seeds, then runs evaluation.

Configuration is entirely driven by shared/hyperparameters.py:
  SEEDS  — list of random seeds (e.g. [42] for a single run, [42, 123, 456] for multi-seed)
  DEVICE — "auto" to detect CUDA at runtime, or "cpu" / "cuda" to force a specific device

Usage:
    python run_all.py              # full pipeline: train → aggregate → evaluate
    python run_all.py --skip-eval  # stop after aggregation (skip 07_evaluate.py)
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)  # flush every line so tail -f works

import numpy as np
import pandas as pd
import torch

from shared.hyperparameters import DEVICE, DEV_SEEDS, PROD_SEEDS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(device_cfg: str) -> str:
    """Resolve 'auto' to 'cuda' or 'cpu' based on hardware availability."""
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def _load_script(path: str):
    """Load a Python script as a module by file path (handles digit-prefixed names)."""
    spec = importlib.util.spec_from_file_location("_run_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

DL_MODELS = [
    "lstm", "transformer",
    "glofas_lstm", "grfr_lstm",
    "glofas_transformer", "grfr_transformer",
]

PRED_BASE = Path("output/predictions")
MODEL_DIR = Path("output/model")


def aggregate_predictions(seeds: list[int]) -> None:
    """Average per-seed qsim predictions and write final files to the top-level
    output/predictions/{model}/ directory (without a seed subdirectory).

    For single-seed runs this is effectively a copy. For multi-seed runs the
    qsim values are averaged date-by-date across seeds; qobs and physical-model
    columns (qglofas/qgrfr) are taken from the first seed since they are
    deterministic (data-derived, not model-derived).
    """
    print("\nAggregating predictions across seeds...")
    for model_name in DL_MODELS:
        model_pred_dir = PRED_BASE / model_name
        first_seed_dir = model_pred_dir / f"seed{seeds[0]}"

        if not first_seed_dir.exists():
            print(f"  Warning: {first_seed_dir} not found — skipping {model_name}")
            continue

        gauge_files = sorted(first_seed_dir.glob("*.parquet"))
        for gauge_file in gauge_files:
            # Load predictions for each seed
            seed_dfs = []
            for seed in seeds:
                seed_path = model_pred_dir / f"seed{seed}" / gauge_file.name
                if seed_path.exists():
                    seed_dfs.append(pd.read_parquet(seed_path))
                else:
                    print(f"  Warning: missing {seed_path}")

            if not seed_dfs:
                continue

            if len(seed_dfs) == 1:
                merged = seed_dfs[0].copy()
            else:
                # Base columns from first seed (qobs and physical model cols are identical)
                non_sim_cols = [c for c in seed_dfs[0].columns if c != "qsim"]
                merged = seed_dfs[0][non_sim_cols].copy()
                # Average qsim; all seeds have the same dates in the same order
                qsim_stack = np.stack(
                    [df.set_index("date")["qsim"].values for df in seed_dfs], axis=0
                )
                merged["qsim"] = qsim_stack.mean(axis=0)

            out_path = model_pred_dir / gauge_file.name.replace(".parquet", "_mean.parquet")
            merged.to_parquet(out_path, index=False)

        print(f"  {model_name}: {len(gauge_files)} gauges aggregated")


def aggregate_loss_curves(seeds: list[int]) -> None:
    """Average per-seed loss curves epoch-by-epoch and save to the standard
    (seed-free) path that 07_evaluate.py expects.

    If seeds stopped at different epochs due to early stopping, curves are
    truncated to the minimum epoch length before averaging.
    """
    print("\nAggregating loss curves across seeds...")
    for model_name in DL_MODELS:
        seed_curves = []
        for seed in seeds:
            p = MODEL_DIR / f"{model_name}_seed{seed}_loss_curves.parquet"
            if p.exists():
                seed_curves.append(pd.read_parquet(p))
            else:
                print(f"  Warning: missing {p}")

        if not seed_curves:
            continue

        # Truncate to shortest run (early stopping may differ across seeds)
        min_len = min(len(df) for df in seed_curves)
        truncated = [df.head(min_len).reset_index(drop=True) for df in seed_curves]

        avg_df = truncated[0][["epoch"]].copy()
        avg_df["train_loss"] = np.mean(
            [df["train_loss"].values for df in truncated], axis=0
        )
        avg_df["val_loss"] = np.mean(
            [df["val_loss"].values for df in truncated], axis=0
        )

        out_path = MODEL_DIR / f"{model_name}_loss_curves.parquet"
        avg_df.to_parquet(out_path, index=False)
        print(f"  {model_name}: averaged {len(seed_curves)} seed(s) over {min_len} epochs")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SCRIPT_ORDER = [
    ("01_run_lstm.py",               "Script 01 — LSTM"),
    ("02_run_transformer.py",        "Script 02 — Transformer"),
    ("03_run_glofas_lstm.py",        "Script 03 — GloFAS+LSTM"),
    ("04_run_grfr_lstm.py",          "Script 04 — GRFR+LSTM"),
    ("05_run_glofas_transformer.py", "Script 05 — GloFAS+Transformer"),
    ("06_run_grfr_transformer.py",   "Script 06 — GRFR+Transformer"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip 07_evaluate.py after aggregation"
    )
    parser.add_argument(
        "--mode", choices=["dev", "production"], default="dev",
        help="Hyperparameter profile: 'dev' (default, fast smoke-test) or 'production' (full run)",
    )
    args = parser.parse_args()

    seeds = DEV_SEEDS if args.mode == "dev" else PROD_SEEDS

    device = _resolve_device(DEVICE)
    print(f"{'='*60}")
    print(f"NP-deeplearning — full pipeline")
    print(f"  Device : {device}")
    print(f"  Seeds  : {seeds}")
    print(f"  Mode   : {args.mode}")
    print(f"  Runs   : {len(seeds)} seed(s) × {len(SCRIPT_ORDER)} models = "
          f"{len(seeds) * len(SCRIPT_ORDER)} total")
    print(f"{'='*60}\n")

    # Load each training script once (exec_module runs module-level code once)
    print("Loading training scripts...")
    scripts = {path: _load_script(path) for path, _ in SCRIPT_ORDER}

    # Delete any existing timing CSV so we start fresh for this complete run
    timing_path = Path("output/training_times.csv")
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    if timing_path.exists():
        timing_path.unlink()
        print("Cleared previous training_times.csv\n")

    # ------------------------------------------------------------------
    # Training loop across seeds
    # ------------------------------------------------------------------
    pipeline_start = time.time()

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")
        for path, label in SCRIPT_ORDER:
            print(f"\n--- {label} | seed={seed} ---")
            scripts[path].main(seed=seed, device=device, mode=args.mode)

    print(f"\n{'='*60}")
    print(f"All training complete in {(time.time() - pipeline_start) / 60:.1f} min")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Aggregate predictions and loss curves across seeds
    # ------------------------------------------------------------------
    aggregate_predictions(seeds)
    aggregate_loss_curves(seeds)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    if not args.skip_eval:
        print(f"\n{'='*60}")
        print("Running 07_evaluate.py...")
        print(f"{'='*60}\n")
        evaluate = _load_script("07_evaluate.py")
        evaluate.main()

    total_time = (time.time() - pipeline_start) / 60
    print(f"\nPipeline finished in {total_time:.1f} min.")


if __name__ == "__main__":
    main()
