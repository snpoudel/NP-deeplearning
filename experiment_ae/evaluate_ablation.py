"""
experiment_ae/evaluate_ablation.py

Evaluate the input-feature ablation for both LSTM and Transformer and produce
a 2x2 boxplot figure comparing NSE and PBIAS across the 15 Nepal basins.

Default output uses seed 42. Predictions are read from
experiment_ae/run_ablation.py (LSTM) and
experiment_ae/run_ablation_transformer.py (Transformer)'s seed42 output:
    output/predictions/lstm_dynamic/seed42/*.parquet
    output/predictions/lstm_static/seed42/*.parquet
    output/predictions/transformer_dynamic/seed42/*.parquet
    output/predictions/transformer_static/seed42/*.parquet

The "alphaearth" variant is not trained separately: it is identical to the
main pipeline's production lstm/transformer model (same features, same
hyperparameters), so its numbers are read directly from the main pipeline's
own seed42 output instead of a redundant re-run:
    output/predictions/lstm/seed42/*.parquet
    output/predictions/transformer/seed42/*.parquet

Outputs (default, seed 42):
    output/figures/fig_input_ablation_cdf.png
    output/figures/fig_input_ablation_cdf.svg
    output/metrics_input_ablation.parquet
    Printed table of median NSE, KGE, RMSE, PBIAS per variant.

Pass --seed N to plot a different single seed instead (output goes to a
_seedN-suffixed file). Pass --mean to plot the 5-seed mean instead (output
goes to a _mean-suffixed file).

Usage:
    python experiment_ae/evaluate_ablation.py
    python experiment_ae/evaluate_ablation.py --seed 123
    python experiment_ae/evaluate_ablation.py --mean
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.hyperparameters import SPLIT_DATES
from shared.metrics import kge, nse, pbias, rmse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ARCHITECTURES = {
    "lstm":        "LSTM",
    "transformer": "Transformer",
}

INPUT_VARIANTS = {
    "dynamic":    "Dynamic\nonly",
    "static":     "Dynamic +\nbasin attributes",
    "alphaearth": "Dynamic +\nAlphaEarth",
}

# Okabe-Ito palette (matches the model colors used in 07_evaluate.py)
COLORS = {
    "dynamic":    "#009E73",
    "static":     "#0072B2",
    "alphaearth": "#D55E00",
}

PRED_BASE   = Path("output/predictions")
FIGURES_DIR = Path("output/figures")
TEST_START  = pd.Timestamp(SPLIT_DATES["test"][0])
TEST_END    = pd.Timestamp(SPLIT_DATES["test"][1])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_mean_predictions(pred_dir: Path, file_stem: str) -> dict[str, pd.DataFrame]:
    """Load seed-averaged (*_mean.parquet) test-period predictions, keyed by gauge_id."""
    if not pred_dir.exists():
        raise FileNotFoundError(
            f"Prediction directory not found: {pred_dir}\n"
            "Run experiment_ae/run_ablation.py and "
            "experiment_ae/run_ablation_transformer.py first."
        )

    preds = {}
    for path in sorted(pred_dir.glob(f"nepal_*_{file_stem}_mean.parquet")):
        gauge_id = re.sub(
            rf"^nepal_|_{re.escape(file_stem)}_mean\.parquet$", "", path.name
        )
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)].reset_index(drop=True)
        if len(df) > 0:
            preds[gauge_id] = df

    return preds


def _load_single_seed_predictions(pred_dir: Path, file_stem: str, seed: int) -> dict[str, pd.DataFrame]:
    """Load one seed's (not averaged) test-period predictions, keyed by gauge_id."""
    seed_dir = pred_dir / f"seed{seed}"
    if not seed_dir.exists():
        raise FileNotFoundError(f"Seed directory not found: {seed_dir}")

    preds = {}
    for path in sorted(seed_dir.glob(f"nepal_*_{file_stem}.parquet")):
        gauge_id = re.sub(rf"^nepal_|_{re.escape(file_stem)}\.parquet$", "", path.name)
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)].reset_index(drop=True)
        if len(df) > 0:
            preds[gauge_id] = df

    return preds


def load_variant_predictions(
    architecture: str, input_variant: str, seed: int | None = None
) -> dict[str, pd.DataFrame]:
    """Load test-period predictions for one variant, keyed by gauge_id.

    "dynamic" and "static" come from this experiment's own multi-seed ablation
    run; "alphaearth" is read straight from the main pipeline's production
    model, since the two are identical by construction (see module docstring).

    If seed is given, loads that single seed's raw predictions instead of the
    5-seed mean (from the same directories either way).
    """
    if input_variant == "alphaearth":
        pred_dir = PRED_BASE / architecture
        file_stem = architecture
    else:
        pred_dir = PRED_BASE / f"{architecture}_{input_variant}"
        file_stem = f"{architecture}_{input_variant}"

    if seed is None:
        preds = _load_mean_predictions(pred_dir, file_stem)
    else:
        preds = _load_single_seed_predictions(pred_dir, file_stem, seed)

    print(f"  {architecture}_{input_variant}: {len(preds)} gauges loaded (from {pred_dir})")
    return preds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Compute NSE, KGE, RMSE, and PBIAS per variant per gauge."""
    records = []
    for variant_name, gauge_preds in preds.items():
        architecture, input_variant = variant_name.split("_", 1)
        for gauge_id, df in gauge_preds.items():
            q_obs = df["qobs"].to_numpy(dtype=np.float64)
            q_sim = df["qsim"].to_numpy(dtype=np.float64)
            records.append({
                "variant":       variant_name,
                "architecture":  architecture,
                "input_variant": input_variant,
                "gauge_id":      gauge_id,
                "nse":           nse(q_obs, q_sim),
                "kge":           kge(q_obs, q_sim),
                "rmse":          rmse(q_obs, q_sim),
                "pbias":         pbias(q_obs, q_sim),
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

METRICS_META = [
    ("nse",   "NSE",       None),
    ("pbias", "PBIAS (%)", 0),
]

def plot_boxplots(metrics_df: pd.DataFrame) -> plt.Figure:
    """2x2 boxplot figure: rows = NSE, PBIAS (shared y-label per row),
    columns = LSTM, Transformer (labeled at top)."""
    variant_keys = list(INPUT_VARIANTS.keys())
    arch_keys = list(ARCHITECTURES.keys())
    x_pos = np.arange(len(variant_keys))
    labels = [INPUT_VARIANTS[v] for v in variant_keys]

    n_rows, n_cols = len(METRICS_META), len(arch_keys)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6, 5), sharey="row")

    rng = np.random.default_rng(0)
    col_labels = iter("abcdefghijklmnop")

    for col, architecture in enumerate(arch_keys):
        arch_df = metrics_df[metrics_df["architecture"] == architecture]

        for row, (metric, ylabel, refline) in enumerate(METRICS_META):
            ax = axes[row, col]
            data_per_variant = [
                arch_df[arch_df["input_variant"] == v][metric].dropna().to_numpy()
                for v in variant_keys
            ]

            bp = ax.boxplot(
                data_per_variant,
                positions=x_pos,
                widths=0.30,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=1.8),
                whiskerprops=dict(linewidth=1.0),
                capprops=dict(linewidth=1.0),
                flierprops=dict(marker=""),
                showfliers=False,
            )

            for patch, vk in zip(bp["boxes"], variant_keys):
                patch.set_facecolor(COLORS[vk])
                patch.set_alpha(0.45)
                patch.set_edgecolor(COLORS[vk])

            # Jittered dots
            for xi, (vals, vk) in enumerate(zip(data_per_variant, variant_keys)):
                jitter = rng.uniform(-0.08, 0.08, size=len(vals))
                ax.scatter(xi + jitter, vals, color=COLORS[vk], s=16, zorder=3, alpha=0.85, linewidths=0)

            # Median label inside the box, nudged below the median line
            all_vals = np.concatenate(data_per_variant)
            y_offset = 0.06 * (np.nanmax(all_vals) - np.nanmin(all_vals))
            for xi, vals in enumerate(data_per_variant):
                med = np.median(vals)
                ax.text(
                    xi, med - y_offset, f"{med:.2f}",
                    ha="center", va="center", fontsize=7, color="white",
                    fontweight="bold", zorder=5,
                    path_effects=[pe.withStroke(linewidth=2, foreground="#222222")],
                )

            if refline is not None:
                ax.axhline(refline, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

            ax.set_xticks(x_pos)
            if row == n_rows - 1:
                ax.set_xticklabels(labels, fontsize=7.5, rotation=20, ha="right")
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            else:
                ax.tick_params(left=False, labelleft=False)
            ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)

        axes[0, col].set_title(
            f"({next(col_labels)}) {ARCHITECTURES[architecture]}",
            fontsize=11,
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(metrics_df: pd.DataFrame) -> None:
    print("\n--- Median metrics across 15 gauges ---")
    summary = (
        metrics_df.groupby("variant")[["nse", "kge", "rmse", "pbias"]]
        .median()
        .round(3)
    )
    summary = summary.sort_values("nse", ascending=False)
    print(summary.to_string())
    best_nse = summary["nse"].idxmax()
    best_kge = summary["kge"].idxmax()
    print(f"\nBest median NSE: {best_nse}")
    print(f"Best median KGE: {best_kge}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed: int | None = 42) -> None:
    tag = " (5-seed mean)" if seed is None else f" (seed {seed} only)"
    print(f"=== Input Ablation Evaluation{tag} ===\n")

    # Load predictions for all architecture x input-variant combinations
    print("Loading predictions...")
    preds = {}
    for architecture in ARCHITECTURES:
        for input_variant in INPUT_VARIANTS:
            variant_name = f"{architecture}_{input_variant}"
            preds[variant_name] = load_variant_predictions(architecture, input_variant, seed=seed)

    # Compute metrics
    print("\nComputing NSE, KGE, RMSE, PBIAS...")
    metrics_df = compute_metrics(preds)

    # Save metrics
    if seed is None:
        suffix = "_mean"
    elif seed == 42:
        suffix = ""
    else:
        suffix = f"_seed{seed}"
    metrics_path = Path("output") / f"metrics_input_ablation{suffix}.parquet"
    metrics_df.to_parquet(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

    # Summary table
    print_summary(metrics_df)

    # Figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_boxplots(metrics_df)

    for ext in ("png", "svg"):
        out = FIGURES_DIR / f"fig_input_ablation_cdf{suffix}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"\nFigure saved to {out}")

    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the input-feature ablation")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Which single seed's predictions to plot (default: 42). "
             "Non-42 values are written to a _seedN-suffixed file.",
    )
    parser.add_argument(
        "--mean", action="store_true",
        help="Plot the 5-seed mean instead of a single seed. "
             "Output goes to a _mean-suffixed file.",
    )
    args = parser.parse_args()
    main(seed=None if args.mean else args.seed)
