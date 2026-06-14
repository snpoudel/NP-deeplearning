"""
experiment_ae/evaluate_ablation.py

Evaluate the three input-feature LSTM variants and produce a 2-panel CDF
figure comparing NSE and KGE across the 15 Nepal basins.

Reads predictions written by experiment_ae/run_ablation.py:
    output/predictions/lstm_dynamic/seed42/
    output/predictions/lstm_static/seed42/
    output/predictions/lstm_alphaearth/seed42/

Outputs:
    output/figures/fig_input_ablation_cdf.png
    output/figures/fig_input_ablation_cdf.svg
    Printed table of median NSE and KGE per variant.

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
from shared.models import kge, nse, pbias, rmse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 42

VARIANTS = {
    "lstm_dynamic":    "Dynamic only",
    "lstm_static":     "Dynamic + Static",
    "lstm_alphaearth": "Dynamic + AlphaEarth",
}

# Okabe-Ito palette — mirrors the model colors used in 07_evaluate.py
# lstm_dynamic → lstm (#009E73), lstm_static → glofas_lstm (#0072B2),
# lstm_alphaearth → grfr_lstm (#D55E00)
COLORS = {
    "lstm_dynamic":    "#009E73",
    "lstm_static":     "#0072B2",
    "lstm_alphaearth": "#D55E00",
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
            "Run experiment_ae/run_ablation.py first."
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
        for gauge_id, df in gauge_preds.items():
            q_obs = df["qobs"].to_numpy(dtype=np.float64)
            q_sim = df["qsim"].to_numpy(dtype=np.float64)
            records.append({
                "variant":  variant_name,
                "gauge_id": gauge_id,
                "nse":      nse(q_obs, q_sim),
                "kge":      kge(q_obs, q_sim),
                "rmse":     rmse(q_obs, q_sim),
                "pbias":    pbias(q_obs, q_sim),
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

METRICS_META = [
    ("nse",   "NSE",   None),
    ("kge",   "KGE",   None),
    ("rmse",  "RMSE",  None),
    ("pbias", "PBIAS (%)", 0),   # reference line at 0 for bias
]

def plot_boxplots(metrics_df: pd.DataFrame) -> plt.Figure:
    """2×2 boxplot figure: NSE, KGE, RMSE, PBIAS."""
    variant_keys = list(VARIANTS.keys())
    x_pos = np.arange(len(variant_keys))
    labels = [VARIANTS[v] for v in variant_keys]

    fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True)
    axes_flat = axes.flatten()

    rng = np.random.default_rng(0)

    for ax, (metric, ylabel, refline) in zip(axes_flat, METRICS_META):
        data_per_variant = [
            metrics_df[metrics_df["variant"] == v][metric].dropna().to_numpy()
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
            fmt = f"{med:.2f}" if metric != "rmse" else f"{med:.1f}"
            ax.text(
                xi, med - y_offset, fmt,
                ha="center", va="center", fontsize=7, color="white",
                fontweight="bold", zorder=5,
                path_effects=[pe.withStroke(linewidth=2, foreground="#222222")],
            )

        if refline is not None:
            ax.axhline(refline, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    # x-tick labels only on bottom row
    for ax in axes[1]:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=8.5, rotation=15, ha="right")

    fig.suptitle("Input feature ablation — LSTM (seed 42, test 2005–2014)", fontsize=11)
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
        .rename(index=VARIANTS)
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

    # Load predictions for all variants
    print("Loading predictions...")
    preds = {}
    for variant_name in VARIANTS:
        preds[variant_name] = load_variant_predictions(variant_name)

    # Compute metrics
    print("\nComputing NSE and KGE...")
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
