"""
experiment_hp_tuning/evaluate_tuning.py

Evaluate the hidden-size hyperparameter sweep (LSTM hidden_size and
Transformer d_model, both in {64, 128, 256}, AlphaEarth input, seed 42) and
produce a line-plot-with-markers figure of best validation loss vs. hidden
size, one line per architecture.

Reads loss curves written by experiment_hp_tuning/run_tuning_lstm.py and
run_tuning_transformer.py:
    output/model/lstm_h{64,128,256}_seed42_loss_curves.parquet
    output/model/transformer_h{64,128,256}_seed42_loss_curves.parquet

Outputs:
    output/figures/fig_hp_tuning_val_loss.png
    output/figures/fig_hp_tuning_val_loss.svg
    output/metrics_hp_tuning.parquet
    Printed table of best validation loss per variant.

Usage:
    python experiment_hp_tuning/evaluate_tuning.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 42
HIDDEN_SIZES = [64, 128, 256]

ARCHITECTURES = {
    "lstm":        "LSTM",
    "transformer": "Transformer",
}

# Okabe-Ito palette — matches lstm/transformer colors used in 07_evaluate.py
COLORS = {
    "lstm":        "#009E73",
    "transformer": "#F0E442",
}

# Hidden size actually used for the production models (highlighted in the figure)
CHOSEN_HIDDEN_SIZE = {
    "lstm":        256,
    "transformer": 128,
}

MODEL_DIR   = Path("output/model")
FIGURES_DIR = Path("output/figures")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_best_val_losses() -> pd.DataFrame:
    """Load each variant's loss curves and extract the best (minimum) val loss."""
    records = []
    for architecture in ARCHITECTURES:
        for hidden_size in HIDDEN_SIZES:
            variant_name = f"{architecture}_h{hidden_size}"
            path = MODEL_DIR / f"{variant_name}_seed{SEED}_loss_curves.parquet"
            if not path.exists():
                raise FileNotFoundError(
                    f"Loss curves not found: {path}\n"
                    "Run experiment_hp_tuning/run_tuning_lstm.py and "
                    "experiment_hp_tuning/run_tuning_transformer.py first."
                )
            loss_df = pd.read_parquet(path)
            train_time = (
                loss_df["train_time_seconds"].iloc[0]
                if "train_time_seconds" in loss_df.columns else float("nan")
            )
            records.append({
                "variant":            variant_name,
                "architecture":       architecture,
                "hidden_size":        hidden_size,
                "best_val_loss":      loss_df["val_loss"].min(),
                "epochs_run":         len(loss_df),
                "train_time_seconds": train_time,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_val_loss(metrics_df: pd.DataFrame) -> plt.Figure:
    """Line-plot-with-markers: best validation loss vs. hidden size, one line
    per architecture."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for architecture, label in ARCHITECTURES.items():
        arch_df = metrics_df[metrics_df["architecture"] == architecture].sort_values("hidden_size")
        ax.plot(
            arch_df["hidden_size"], arch_df["best_val_loss"],
            linewidth=1.8, color=COLORS[architecture], label=label, zorder=2,
        )
        sizes = [
            220 if h == CHOSEN_HIDDEN_SIZE[architecture] else 70
            for h in arch_df["hidden_size"]
        ]
        ax.scatter(
            arch_df["hidden_size"], arch_df["best_val_loss"],
            s=sizes, color=COLORS[architecture],
            edgecolor="#222222", linewidths=0.8, zorder=3,
        )

    ax.set_xticks(HIDDEN_SIZES)
    ax.set_xticklabels([str(h) for h in HIDDEN_SIZES])
    ax.set_xlabel("Hidden size", fontsize=10)
    ax.set_ylabel("Validation loss (MSE)", fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.margins(y=0.15)
    ax.legend(frameon=False, fontsize=9.5)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(metrics_df: pd.DataFrame) -> None:
    print("\n--- Best validation loss per variant ---")
    summary = metrics_df.set_index("variant")[
        ["architecture", "hidden_size", "epochs_run", "best_val_loss", "train_time_seconds"]
    ].round(4)
    print(summary.to_string())
    best_row = metrics_df.loc[metrics_df["best_val_loss"].idxmin()]
    print(f"\nOverall best: {best_row['variant']} (val_loss={best_row['best_val_loss']:.4f})")
    for architecture, label in ARCHITECTURES.items():
        arch_df = metrics_df[metrics_df["architecture"] == architecture]
        best = arch_df.loc[arch_df["best_val_loss"].idxmin()]
        print(f"Best {label}: hidden_size={best['hidden_size']} (val_loss={best['best_val_loss']:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Hidden-Size Hyperparameter Tuning Evaluation ===\n")

    print("Loading loss curves...")
    metrics_df = load_best_val_losses()

    metrics_path = Path("output") / "metrics_hp_tuning.parquet"
    metrics_df.to_parquet(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

    print_summary(metrics_df)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_val_loss(metrics_df)

    for ext in ("png", "svg"):
        out = FIGURES_DIR / f"fig_hp_tuning_val_loss.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"\nFigure saved to {out}")

    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
