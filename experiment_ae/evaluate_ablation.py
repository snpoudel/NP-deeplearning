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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.hyperparameters import SPLIT_DATES
from shared.models import kge, nse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 42

VARIANTS = {
    "lstm_dynamic":    "Dynamic only",
    "lstm_static":     "Dynamic + Static",
    "lstm_alphaearth": "Dynamic + AlphaEarth",
}

# Colorblind-safe (Wong palette)
COLORS = {
    "lstm_dynamic":    "#009E73",
    "lstm_static":     "#0072B2",
    "lstm_alphaearth": "#D55E00",
}

LINESTYLES = {
    "lstm_dynamic":    "-",
    "lstm_static":     "--",
    "lstm_alphaearth": "-.",
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
    """Compute NSE and KGE per variant per gauge.

    Returns a DataFrame with columns: variant, gauge_id, nse, kge.
    """
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
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CDF helper
# ---------------------------------------------------------------------------

def _cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted_values, cumulative_probabilities) for a CDF plot."""
    s = np.sort(values)
    n = len(s)
    p = np.arange(1, n + 1) / (n + 1)
    return s, p


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_cdf(metrics_df: pd.DataFrame) -> plt.Figure:
    """Two-panel CDF figure: left = NSE, right = KGE."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), sharey=True)

    for metric, ax, xlabel in zip(
        ["nse", "kge"],
        axes,
        ["NSE", "KGE"],
    ):
        for variant_name, label in VARIANTS.items():
            subset = metrics_df[metrics_df["variant"] == variant_name][metric].dropna()
            if len(subset) == 0:
                continue
            vals, probs = _cdf(subset.to_numpy())
            ax.plot(
                vals,
                probs,
                label=label,
                color=COLORS[variant_name],
                linestyle=LINESTYLES[variant_name],
                linewidth=1.8,
            )

        ax.axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_xlim(left=min(-0.5, metrics_df[metric].min() - 0.05))
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Cumulative probability", fontsize=11)
    axes[1].legend(
        frameon=False,
        fontsize=9,
        loc="lower right",
    )

    fig.suptitle("Input feature ablation — LSTM (seed 42, test 2005–2014)", fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(metrics_df: pd.DataFrame) -> None:
    print("\n--- Median metrics across 15 gauges ---")
    summary = (
        metrics_df.groupby("variant")[["nse", "kge"]]
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
    fig = plot_cdf(metrics_df)

    for ext in ("png", "svg"):
        out = FIGURES_DIR / f"fig_input_ablation_cdf.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"\nFigure saved to {out}")

    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
