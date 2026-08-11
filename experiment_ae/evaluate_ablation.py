"""
experiment_ae/evaluate_ablation.py

Evaluate the input-feature ablation for both LSTM and Transformer and produce
a 2x3 boxplot figure comparing NSE, PBIAS, and HFB (high-flow bias) across the
15 Nepal basins.

Reads predictions written by experiment_ae/run_ablation.py (LSTM) and
experiment_ae/run_ablation_transformer.py (Transformer):
    output/predictions/lstm_dynamic/seed42/
    output/predictions/lstm_static/seed42/
    output/predictions/lstm_alphaearth/seed42/
    output/predictions/transformer_dynamic/seed42/
    output/predictions/transformer_static/seed42/
    output/predictions/transformer_alphaearth/seed42/

Outputs:
    output/figures/fig_input_ablation_cdf.png
    output/figures/fig_input_ablation_cdf.svg
    Printed table of median NSE, KGE, RMSE, PBIAS, HFB per variant.

Usage:
    python experiment_ae/evaluate_ablation.py
"""

from __future__ import annotations

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

SEED = 42

ARCHITECTURES = {
    "lstm":        "LSTM",
    "transformer": "Transformer",
}

INPUT_VARIANTS = {
    "dynamic":    "Dynamic\nonly",
    "static":     "Dynamic +\nbasin attributes",
    "alphaearth": "Dynamic +\nAlphaEarth",
}

# Okabe-Ito palette — mirrors the model colors used in 07_evaluate.py
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

def load_variant_predictions(variant_name: str) -> dict[str, pd.DataFrame]:
    """Load test-period predictions for one variant, keyed by gauge_id."""
    pred_dir = PRED_BASE / variant_name / f"seed{SEED}"
    if not pred_dir.exists():
        raise FileNotFoundError(
            f"Prediction directory not found: {pred_dir}\n"
            "Run experiment_ae/run_ablation.py and "
            "experiment_ae/run_ablation_transformer.py first."
        )

    preds = {}
    for path in sorted(pred_dir.glob(f"nepal_*_{variant_name}.parquet")):
        gauge_id = re.sub(
            rf"^nepal_|_{re.escape(variant_name)}\.parquet$", "", path.name
        )
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)].reset_index(drop=True)
        if len(df) > 0:
            preds[gauge_id] = df

    print(f"  {variant_name}: {len(preds)} gauges loaded")
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
    # Sort by descending median NSE
    summary = summary.sort_values("nse", ascending=False)
    print(summary.to_string())
    best_nse = summary["nse"].idxmax()
    best_kge = summary["kge"].idxmax()
    print(f"\nBest median NSE: {best_nse}")
    print(f"Best median KGE: {best_kge}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Input Ablation Evaluation ===\n")

    # Load predictions for all architecture x input-variant combinations
    print("Loading predictions...")
    preds = {}
    for architecture in ARCHITECTURES:
        for input_variant in INPUT_VARIANTS:
            variant_name = f"{architecture}_{input_variant}"
            preds[variant_name] = load_variant_predictions(variant_name)

    # Compute metrics
    print("\nComputing NSE, KGE, RMSE, PBIAS...")
    metrics_df = compute_metrics(preds)

    # Save metrics
    metrics_path = Path("output") / "metrics_input_ablation.parquet"
    metrics_df.to_parquet(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

    # Summary table
    print_summary(metrics_df)

    # Figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_boxplots(metrics_df)

    for ext in ("png", "svg"):
        out = FIGURES_DIR / f"fig_input_ablation_cdf.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"\nFigure saved to {out}")

    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
