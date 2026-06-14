"""
08_residual_analysis.py

Four residual-analysis figures for all post-processor combinations
(LSTM | Transformer) × (GloFAS | GRFR).

Each figure: 3 sites × 2 panels (correction scatter + residual time series).
Layout: GridSpec with spanning row labels for publication quality.
Max figure size: 8 × 8 in, 300 DPI.

Outputs:
  fig9a_lstm_glofas.png / .svg
  fig9b_lstm_grfr.png   / .svg
  fig9c_transformer_glofas.png / .svg
  fig9d_transformer_grfr.png   / .svg
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import pandas as pd

from shared.hyperparameters import SPLIT_DATES

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PRED_DIR    = Path("output/predictions")
FIGURES_DIR = Path("output/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Sites — fixed across all 4 figures for cross-figure comparability
# Per-panel ΔNSE is computed at runtime for each specific combination.
# ---------------------------------------------------------------------------

SELECTED_SITES = ["447", "610", "438"]
SITE_LETTERS   = {"447": "a", "610": "b", "438": "c"}

# ---------------------------------------------------------------------------
# Visual style (colorblind-safe Wong palette)
# ---------------------------------------------------------------------------

COLORS = {
    "glofas":              "#E69F00",   # amber
    "grfr":               "#56B4E9",   # sky blue
    "lstm":               "#009E73",   # green
    "transformer":        "#CC79A7",   # reddish purple
    "glofas_lstm":        "#0072B2",   # dark blue
    "grfr_lstm":          "#D55E00",   # vermillion
    "glofas_transformer": "#0072B2",
    "grfr_transformer":   "#D55E00",
}
LS = {
    "glofas":              "-",
    "grfr":               "-",
    "lstm":               "--",
    "transformer":        "--",
    "glofas_lstm":        "-.",
    "grfr_lstm":          ":",
    "glofas_transformer": "-.",
    "grfr_transformer":   ":",
}
MODEL_LABELS = {
    "glofas":              "GloFAS",
    "grfr":               "GRFR",
    "lstm":               "LSTM",
    "transformer":        "Transformer",
    "glofas_lstm":        "GloFAS+LSTM",
    "grfr_lstm":          "GRFR+LSTM",
    "glofas_transformer": "GloFAS+Transformer",
    "grfr_transformer":   "GRFR+Transformer",
}

FONT  = 7
TITLE = 7.5
DPI   = 300

plt.rcParams.update({
    "font.size":        FONT,
    "axes.titlesize":   TITLE,
    "axes.labelsize":   FONT,
    "xtick.labelsize":  FONT - 0.5,
    "ytick.labelsize":  FONT - 0.5,
    "legend.fontsize":  FONT - 0.5,
    "axes.linewidth":   0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

TEST_START = pd.Timestamp(SPLIT_DATES["test"][0])
TEST_END   = pd.Timestamp(SPLIT_DATES["test"][1])

# ---------------------------------------------------------------------------
# Figure configurations
# ---------------------------------------------------------------------------

FIGURE_CONFIGS: List[Dict] = [
    dict(outname="fig9a_lstm_glofas",
         title="LSTM Post-Processing of GloFAS Errors",
         physical_key="glofas", phys_col="qglofas",
         dl_key="lstm",        hybrid_key="glofas_lstm"),
    dict(outname="fig9b_lstm_grfr",
         title="LSTM Post-Processing of GRFR Errors",
         physical_key="grfr",  phys_col="qgrfr",
         dl_key="lstm",        hybrid_key="grfr_lstm"),
    dict(outname="fig9c_transformer_glofas",
         title="Transformer Post-Processing of GloFAS Errors",
         physical_key="glofas", phys_col="qglofas",
         dl_key="transformer", hybrid_key="glofas_transformer"),
    dict(outname="fig9d_transformer_grfr",
         title="Transformer Post-Processing of GRFR Errors",
         physical_key="grfr",  phys_col="qgrfr",
         dl_key="transformer", hybrid_key="grfr_transformer"),
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_DL_DIRS     = ["lstm", "transformer"]
_HYBRID_DIRS = ["glofas_lstm", "grfr_lstm", "glofas_transformer", "grfr_transformer"]


def _read(subdir: str, gauge: str) -> Optional[pd.DataFrame]:
    path = PRED_DIR / subdir / f"nepal_{gauge}_{subdir}_mean.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_site(gauge: str) -> Dict[str, pd.DataFrame]:
    """Return {model_key: DataFrame(date, qobs, qsim[, qphysical])} for test period."""
    result: Dict[str, pd.DataFrame] = {}

    for m in _DL_DIRS + _HYBRID_DIRS:
        df = _read(m, gauge)
        if df is None:
            continue
        df = (df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)]
              .copy().reset_index(drop=True))
        result[m] = df

    # Physical baselines from hybrid files
    for hybrid, phys_col, base_key in [
        ("glofas_lstm",        "qglofas", "glofas"),
        ("grfr_lstm",          "qgrfr",   "grfr"),
        ("glofas_transformer", "qglofas", "glofas"),
        ("grfr_transformer",   "qgrfr",   "grfr"),
    ]:
        if base_key not in result:
            df = result.get(hybrid)
            if df is not None and phys_col in df.columns:
                result[base_key] = (df[["date", "qobs", phys_col]]
                                    .rename(columns={phys_col: "qsim"})
                                    .copy())
    return result


# ---------------------------------------------------------------------------
# Time-series window: auto-detect first shared 2-year window with data
# ---------------------------------------------------------------------------

def find_shared_ts_window(
    all_data: Dict[str, Dict],
    sites: List[str],
    window_years: int = 2,
    min_days: int = 300,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start, end) of first consecutive window_years period common to all sites."""
    site_valid: List[set] = []
    for gauge in sites:
        site = all_data[gauge]
        # Use first available dataframe for this site
        df = next(iter(site.values()))
        counts = df.groupby(df["date"].dt.year)["qobs"].count()
        site_valid.append(set(int(y) for y, c in counts.items() if c >= min_days))

    common = sorted(set.intersection(*site_valid))

    for i in range(len(common) - window_years + 1):
        if all(common[i + j + 1] - common[i + j] == 1 for j in range(window_years - 1)):
            y0, y1 = common[i], common[i + window_years - 1]
            return pd.Timestamp(f"{y0}-01-01"), pd.Timestamp(f"{y1}-12-31")

    # Fallback: first single available year
    if common:
        y0 = common[0]
        return pd.Timestamp(f"{y0}-01-01"), pd.Timestamp(f"{y0}-12-31")

    raise ValueError("No common time-series window found for selected sites.")


# ---------------------------------------------------------------------------
# NSE metric (for per-panel ΔNSE annotation)
# ---------------------------------------------------------------------------

def _nse(qobs: np.ndarray, qsim: np.ndarray) -> float:
    denom = np.nansum((qobs - np.nanmean(qobs)) ** 2)
    if denom == 0:
        return float("nan")
    return float(1.0 - np.nansum((qobs - qsim) ** 2) / denom)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid(ax) -> None:
    ax.grid(True, linestyle="--", alpha=0.25, linewidth=0.35)


def _shade_monsoon(ax, ts_start: pd.Timestamp, ts_end: pd.Timestamp) -> None:
    yr = ts_start.year
    while yr <= ts_end.year:
        s = pd.Timestamp(f"{yr}-06-01")
        e = pd.Timestamp(f"{yr}-10-31")
        if s <= ts_end and e >= ts_start:
            ax.axvspan(max(s, ts_start), min(e, ts_end),
                       color="lightgray", alpha=0.40, zorder=0, lw=0)
        yr += 1


# ---------------------------------------------------------------------------
# Panel functions
# ---------------------------------------------------------------------------

def _panel_scatter(
    ax,
    site_data: Dict,
    hybrid_key: str,
    phys_col: str,
    physical_key: str,
) -> None:
    """Scatter dots: true physical-model residual vs DL-predicted correction.
    ΔNSE annotation computed from this combination's actual data.
    """
    df = site_data.get(hybrid_key)
    if df is None or phys_col not in df.columns:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", fontsize=FONT)
        return

    true_resid = (df["qobs"]  - df[phys_col]).values
    pred_corr  = (df["qsim"]  - df[phys_col]).values

    ax.scatter(true_resid, pred_corr, s=1.5, alpha=0.10,
               color=COLORS[hybrid_key], rasterized=True, zorder=2)

    lim = max(
        np.nanpercentile(np.abs(true_resid), 99),
        np.nanpercentile(np.abs(pred_corr),  99),
    ) * 1.15
    lim = max(lim, 1.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", zorder=1)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle=":",  zorder=1)
    ax.plot([-lim, lim], [-lim, lim], color="black",
            linewidth=0.7, linestyle="--", zorder=3)
    _grid(ax)

    # Per-panel ΔNSE annotation (computed from actual data for this combination)
    phys_df = site_data.get(physical_key)
    if phys_df is not None:
        nse_hybrid = _nse(df["qobs"].values, df["qsim"].values)
        nse_phys   = _nse(phys_df["qobs"].values, phys_df["qsim"].values)
        delta = nse_hybrid - nse_phys
        if not np.isnan(delta):
            sign = "+" if delta >= 0 else ""
            ax.text(0.97, 0.97,
                    f"ΔNSE = {sign}{delta:.2f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=FONT - 1,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              alpha=0.90, edgecolor="lightgray", lw=0.4))


def _panel_timeseries(
    ax,
    site_data: Dict,
    physical_key: str,
    dl_key: str,
    hybrid_key: str,
    ts_start: pd.Timestamp,
    ts_end: pd.Timestamp,
) -> None:
    """Residual time series for the shared TS window."""
    has_data = False
    for model in [physical_key, dl_key, hybrid_key]:
        df = site_data.get(model)
        if df is None:
            continue
        w = df[(df["date"] >= ts_start) & (df["date"] <= ts_end)]
        if w.empty:
            continue
        has_data = True
        resid = w["qobs"].values - w["qsim"].values
        ax.plot(w["date"], resid, color=COLORS[model], linestyle=LS[model],
                linewidth=0.9, label=MODEL_LABELS[model])

    if not has_data:
        ax.text(0.5, 0.5, "no data in window", transform=ax.transAxes,
                ha="center", va="center", fontsize=FONT - 1, color="gray")
        return

    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", zorder=1)
    _shade_monsoon(ax, ts_start, ts_end)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    ax.tick_params(axis="x", rotation=30, labelsize=FONT - 1)
    _grid(ax)


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def build_figure(
    all_data: Dict[str, Dict],
    cfg: Dict,
    ts_start: pd.Timestamp,
    ts_end: pd.Timestamp,
) -> None:
    physical_key = cfg["physical_key"]
    phys_col     = cfg["phys_col"]
    dl_key       = cfg["dl_key"]
    hybrid_key   = cfg["hybrid_key"]
    phys_name    = MODEL_LABELS[physical_key]
    dl_name      = MODEL_LABELS[dl_key]

    fig = plt.figure(figsize=(6.5, 5.4))
    outer = GridSpec(3, 1, figure=fig, hspace=0.16)

    axes_sc: List = []
    axes_ts: List = []

    for row_i, gauge in enumerate(SELECTED_SITES):
        inner = GridSpecFromSubplotSpec(
            2, 2,
            subplot_spec=outer[row_i],
            height_ratios=[0.11, 1],
            hspace=0.02,
            wspace=0.26,
        )

        # Spanning row label
        label_ax = fig.add_subplot(inner[0, :])
        label_ax.set_axis_off()
        letter = SITE_LETTERS[gauge]
        label_ax.text(
            0.0, 0.5,
            f"{letter})  Site {gauge}",
            transform=label_ax.transAxes,
            va="center", ha="left",
            fontweight="bold", fontsize=FONT + 0.5,
        )

        ax_sc = fig.add_subplot(inner[1, 0])
        ax_ts = fig.add_subplot(inner[1, 1])
        axes_sc.append(ax_sc)
        axes_ts.append(ax_ts)

        site_data = all_data[gauge]
        _panel_scatter(ax_sc, site_data, hybrid_key, phys_col, physical_key)
        _panel_timeseries(ax_ts, site_data, physical_key, dl_key, hybrid_key,
                          ts_start, ts_end)

        # y-labels every row
        ax_sc.set_ylabel(f"{dl_name} correction\n(mm d⁻¹)", fontsize=FONT - 0.5)
        ax_ts.set_ylabel("Error  (mm d⁻¹)",                 fontsize=FONT - 0.5)

        # x-labels bottom row only
        if row_i == 2:
            ax_sc.set_xlabel(f"{phys_name} residual  (mm d⁻¹)", fontsize=FONT - 0.5)
            ax_ts.set_xlabel("Date",                              fontsize=FONT - 0.5)

    # Shared legend below figure
    legend_handles = [
        mlines.Line2D([], [], color=COLORS[m], linestyle=LS[m],
                      linewidth=1.1, label=MODEL_LABELS[m])
        for m in [physical_key, dl_key, hybrid_key]
    ] + [
        mlines.Line2D([], [], color="black", linestyle="--",
                      linewidth=0.7, label="1:1 perfect  (scatter)"),
        mlines.Line2D([], [], color="lightgray", linestyle="-",
                      linewidth=5, alpha=0.7, label="Monsoon  (Jun–Oct)"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=5,
        bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=FONT - 0.5,
    )

    for ext in ("png", "svg"):
        fig.savefig(
            FIGURES_DIR / f"{cfg['outname']}.{ext}",
            dpi=DPI, bbox_inches="tight",
        )
    plt.close(fig)
    print(f"  Saved {cfg['outname']}.png/svg")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading prediction data...")
    all_data = {g: load_site(g) for g in SELECTED_SITES}
    for gauge in SELECTED_SITES:
        models = sorted(all_data[gauge].keys())
        print(f"  Site {gauge}: {models}")

    print("\nDetecting shared time-series window...")
    ts_start, ts_end = find_shared_ts_window(all_data, SELECTED_SITES, window_years=1)
    print(f"  Selected window: {ts_start.date()} → {ts_end.date()}")

    for cfg in FIGURE_CONFIGS:
        print(f"\nBuilding {cfg['outname']}...")
        build_figure(all_data, cfg, ts_start, ts_end)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
