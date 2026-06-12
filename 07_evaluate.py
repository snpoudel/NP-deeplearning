# Evaluate all eight models against observed streamflow during the test period
# (2005–2014) and produce publication-quality figures.
#
# Models evaluated:
#   Physical (baseline): GloFAS, GRFR
#   Deep learning:       LSTM, Transformer
#   Hybrid:              GloFAS+LSTM, GRFR+LSTM, GloFAS+Transformer, GRFR+Transformer
#
# Outputs:
#   output/metrics.parquet       — per-model per-gauge metrics table
#   output/figures/fig1_*.png    — nine figures at 300 dpi

from __future__ import annotations

import re
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from shared.hyperparameters import SPLIT_DATES
from shared.models import kge, nse, pbias, pbias_high, pbias_low, pbias_mid, rmse

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INPUT_DIR = Path("input")
PRED_DIR = Path("output/predictions")
FIGURES_DIR = Path("output/figures")
METRICS_PATH = Path("output/metrics.parquet")
GLOFAS_PATH = Path("input/physical_model/glofas_selected_qobs.parquet")
GRFR_PATH = Path("input/physical_model/grfr_selected_qobs.parquet")
SHAPEFILE_PATH = Path("input/shapefile/merged_shapefile.shp")
STATIONS_CSV = Path("input/selected_hydro_stations.csv")

# ---------------------------------------------------------------------------
# Styling — Okabe-Ito color-blind-friendly palette
# ---------------------------------------------------------------------------

# Loading order (used internally for compute_all_metrics / load_all_predictions)
MODEL_NAMES = [
    "glofas", "grfr", "lstm", "transformer",
    "glofas_lstm", "grfr_lstm", "glofas_transformer", "grfr_transformer",
]

# Display order for all figures: LSTM group, then Transformer group
DISPLAY_ORDER = [
    "lstm", "transformer",
    "glofas", "glofas_lstm", "glofas_transformer",
    "grfr", "grfr_lstm", "grfr_transformer",
]

MODEL_LABELS = {
    "glofas":             "GloFAS",
    "grfr":               "GRFR",
    "lstm":               "LSTM",
    "transformer":        "Transformer",
    "glofas_lstm":        "GloFAS+LSTM",
    "grfr_lstm":          "GRFR+LSTM",
    "glofas_transformer": "GloFAS+Transformer",
    "grfr_transformer":   "GRFR+Transformer",
}
COLORS = {
    "glofas":             "#E69F00",
    "grfr":               "#56B4E9",
    "lstm":               "#009E73",
    "transformer":        "#F0E442",
    "glofas_lstm":        "#0072B2",
    "grfr_lstm":          "#D55E00",
    "glofas_transformer": "#CC79A7",
    "grfr_transformer":   "#222222",
}
LINESTYLES = {
    "glofas": "-",  "grfr": "-",
    "lstm": "--",   "transformer": "--",
    "glofas_lstm": "-.", "grfr_lstm": ":",
    "glofas_transformer": "-.", "grfr_transformer": ":",
}

FONT_SIZE = 8
TITLE_SIZE = 9
DPI = 300

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
})

# ---------------------------------------------------------------------------
# Gauge ID normalisation and site labels
# ---------------------------------------------------------------------------

def _norm_gauge_id(v) -> str:
    """Normalise gauge_id: 120.0→'120', 259.2→'259.2', '120'→'120'."""
    try:
        f = float(v)
        i = int(f)
        return str(i) if f == i else str(f)
    except (ValueError, TypeError):
        return str(v)


def _build_site_labels() -> dict[str, str]:
    """Return dict gauge_id → 'River@Location(ID)' from the stations CSV."""
    df = pd.read_csv(STATIONS_CSV)
    result = {}
    for _, row in df.iterrows():
        gid = _norm_gauge_id(row["station"])
        result[gid] = f"{row['river']}@{row['location']}({gid})"
    return result


SITE_LABELS: dict[str, str] = _build_site_labels()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _gauge_id_from_filename(path: Path) -> str:
    """Extract gauge_id from a prediction filename, e.g. nepal_120_lstm.parquet → '120'."""
    return _norm_gauge_id(re.sub(r"^nepal_|_\w+\.parquet$", "", path.name))


def load_all_predictions() -> dict[str, dict[str, pd.DataFrame]]:
    """Load test-period predictions for all 8 models.

    Returns:
        Nested dict: preds[model_name][gauge_id] = DataFrame(date, qobs, qsim)
    """
    test_start, test_end = SPLIT_DATES["test"]
    preds: dict[str, dict[str, pd.DataFrame]] = {}

    # --- Deep learning and hybrid models ---
    dl_models = {
        "lstm":               ("lstm",               "qsim"),
        "transformer":        ("transformer",         "qsim"),
        "glofas_lstm":        ("glofas_lstm",         "qsim"),
        "grfr_lstm":          ("grfr_lstm",           "qsim"),
        "glofas_transformer": ("glofas_transformer",  "qsim"),
        "grfr_transformer":   ("grfr_transformer",    "qsim"),
    }
    for model_name, (folder, sim_col) in dl_models.items():
        gauge_dict = {}
        for path in sorted((PRED_DIR / folder).glob("*_mean.parquet")):
            gauge_id = _gauge_id_from_filename(path)
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
            df = df.rename(columns={sim_col: "qsim"})[["date", "qobs", "qsim"]].dropna()
            gauge_dict[gauge_id] = df
        preds[model_name] = gauge_dict

    # --- Physical model baselines ---
    lstm_qobs = {gid: df.set_index("date")["qobs"] for gid, df in preds["lstm"].items()}

    for model_name, phys_path in [("glofas", GLOFAS_PATH), ("grfr", GRFR_PATH)]:
        raw = pd.read_parquet(phys_path)
        raw["date"] = pd.to_datetime(raw["date"], format="mixed")
        raw = raw[(raw["date"] >= test_start) & (raw["date"] <= test_end)]
        raw = raw.melt(id_vars="date", var_name="gauge_id", value_name="qsim")

        gauge_dict = {}
        for gauge_id, group in raw.groupby("gauge_id"):
            gid = _norm_gauge_id(gauge_id)
            if gid not in lstm_qobs:
                continue
            df = group[["date", "qsim"]].copy().set_index("date")
            df["qobs"] = lstm_qobs[gid]
            df = df.dropna().reset_index()
            gauge_dict[gid] = df[["date", "qobs", "qsim"]]
        preds[model_name] = gauge_dict

    return preds


def load_training_qobs() -> dict[str, pd.Series]:
    """Load qobs for the training period from merged input parquets."""
    train_start, train_end = SPLIT_DATES["train"]
    result = {}
    for path in sorted(INPUT_DIR.glob("nepal_*_merged.parquet")):
        gauge_id = _norm_gauge_id(re.sub(r"^nepal_|_merged\.parquet$", "", path.name))
        df = pd.read_parquet(path)[["date", "qobs"]]
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= train_start) & (df["date"] <= train_end)].dropna()
        if len(df) > 0:
            result[gauge_id] = df.set_index("date")["qobs"]
    return result


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_all_metrics(preds: dict) -> pd.DataFrame:
    """Compute NSE, KGE, RMSE, and PBIAS variants for all models × gauges."""
    rows = []
    for model_name in MODEL_NAMES:
        if model_name not in preds:
            continue
        for gauge_id, df in preds[model_name].items():
            obs = df["qobs"].values
            sim = df["qsim"].values
            if len(obs) < 10 or obs.std() == 0:
                continue
            rows.append({
                "model":      model_name,
                "gauge_id":   gauge_id,
                "nse":        nse(obs, sim),
                "kge":        kge(obs, sim),
                "rmse":       rmse(obs, sim),
                "pbias":      pbias(obs, sim),
                "pbias_high": pbias_high(obs, sim),
                "pbias_low":  pbias_low(obs, sim),
                "pbias_mid":  pbias_mid(obs, sim),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helper: CDF from a 1-D array
# ---------------------------------------------------------------------------

def _cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted_values, cumulative_probabilities) for a CDF plot."""
    s = np.sort(values)
    n = len(s)
    p = np.arange(1, n + 1) / (n + 1)
    return s, p


# ---------------------------------------------------------------------------
# Country boundary helper
# ---------------------------------------------------------------------------

_COUNTRY_CACHE = Path("input/shapefile/cache/ne_50m_admin_0_countries.zip")
_COUNTRY_URL   = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"


def _load_country_boundaries() -> gpd.GeoDataFrame | None:
    """Download (once) and return Natural Earth 50 m country polygons."""
    _COUNTRY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not _COUNTRY_CACHE.exists():
        print("  Downloading Natural Earth country boundaries...")
        try:
            import urllib.request
            urllib.request.urlretrieve(_COUNTRY_URL, _COUNTRY_CACHE)
        except Exception as exc:
            print(f"  Warning: download failed — {exc}")
            return None
    try:
        gdf = gpd.read_file(str(_COUNTRY_CACHE))
        if gdf.crs and not gdf.crs.is_geographic:
            gdf = gdf.to_crs(epsg=4326)
        return gdf
    except Exception as exc:
        print(f"  Warning: could not open country shapefile — {exc}")
        return None


# ---------------------------------------------------------------------------
# Figure 1 — Basin map + observation availability (merged)
# ---------------------------------------------------------------------------

def fig1_basin_overview(basins_gdf: gpd.GeoDataFrame) -> None:
    """Two-panel figure: basin location map (top) + observation availability (bottom)."""
    import matplotlib.lines as mlines

    fig, (ax_map, ax_avail) = plt.subplots(
        2, 1, figsize=(7, 9),
        gridspec_kw={"height_ratios": [3, 2]},
    )

    # ── TOP: Basin location map ──────────────────────────────────────────────
    if basins_gdf.crs and not basins_gdf.crs.is_geographic:
        basins_gdf = basins_gdf.to_crs(epsg=4326)

    bounds = basins_gdf.total_bounds
    pad_left, pad_right, pad_bottom, pad_top = 0.5, 0.3, 0.5, 0.5
    xlim = (bounds[0] - pad_left, bounds[2] + pad_right)
    ylim = (bounds[1] - pad_bottom, bounds[3] + pad_top)
    ax_map.set_xlim(xlim)
    ax_map.set_ylim(ylim)

    basemap_label = "Basemap: OpenTopoMap"
    try:
        import contextily as ctx
        for provider, label in [
            (ctx.providers.OpenTopoMap,           "Basemap: OpenTopoMap"),
            (ctx.providers.Esri.WorldShadedRelief, "Basemap: Esri WorldShadedRelief"),
        ]:
            try:
                ctx.add_basemap(ax_map, crs="EPSG:4326", source=provider,
                                zoom=9, attribution=False, alpha=0.5)
                basemap_label = label
                break
            except Exception:
                continue
    except Exception as exc:
        print(f"  Note: elevation basemap not added — {exc}")
        ax_map.set_facecolor("#e8e0d8")
        basemap_label = "No basemap"

    countries = _load_country_boundaries()
    name_col = None
    if countries is not None:
        name_col = next((c for c in ["NAME", "ADMIN", "name"] if c in countries.columns), None)
        clip = countries.cx[xlim[0]:xlim[1], ylim[0]:ylim[1]]
        clip.plot(ax=ax_map, facecolor="none", edgecolor="#888888", linewidth=0.4, zorder=2)
        if name_col:
            nepal = countries[countries[name_col] == "Nepal"]
            if len(nepal) > 0:
                nepal.plot(ax=ax_map, facecolor="none", edgecolor="#333333",
                           linewidth=1.2, zorder=3)
        if name_col:
            area_threshold = (xlim[1] - xlim[0]) * (ylim[1] - ylim[0]) * 0.01
            for _, row in clip.iterrows():
                if row.geometry.area < area_threshold:
                    continue
                cx_c = max(xlim[0], min(xlim[1], row.geometry.centroid.x))
                cy_c = max(ylim[0], min(ylim[1], row.geometry.centroid.y))
                ax_map.text(cx_c, cy_c, row[name_col],
                            fontsize=6, color="#444444", ha="center", va="center",
                            style="italic", zorder=7,
                            bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                                      alpha=0.55, edgecolor="none"))

    basins_gdf.plot(ax=ax_map, facecolor="none", edgecolor="#1A5276",
                    linewidth=1, zorder=4)

    # Label gauges with numeric ID only on map (cleaner at small size)
    texts = []
    for _, row in basins_gdf.iterrows():
        gid = _norm_gauge_id(row["gauge_id"])
        cx_b = row.geometry.centroid.x
        cy_b = row.geometry.centroid.y
        t = ax_map.text(cx_b, cy_b, gid,
                        fontsize=5.5, ha="center", va="center", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  alpha=0.8, edgecolor="none"))
        texts.append(t)

    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax_map, expand_text=(1.2, 1.4),
                    arrowprops=dict(arrowstyle="-", color="#555555", lw=0.4))
    except ImportError:
        pass

    ax_map.set_xlabel("Longitude (°E)", fontsize=FONT_SIZE)
    ax_map.set_ylabel("Latitude (°N)", fontsize=FONT_SIZE)
    ax_map.grid(True, linestyle="--", alpha=0.3, linewidth=0.4, zorder=5)
    ax_map.tick_params(labelsize=FONT_SIZE)

    # North arrow
    ax_map.annotate("N", xy=(0.955, 0.955), xycoords="axes fraction",
                    fontsize=9, ha="center", va="bottom", fontweight="bold", zorder=9)
    ax_map.annotate("", xy=(0.955, 0.955), xytext=(0.955, 0.905), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3), zorder=9)

    # Scale bar
    km_per_deg = 111.32 * np.cos(np.radians(28))
    bar_deg = 100 / km_per_deg
    bar_x1 = xlim[0] + 0.97 * (xlim[1] - xlim[0])
    bar_x0 = bar_x1 - bar_deg
    bar_y  = ylim[0] + 0.855 * (ylim[1] - ylim[0])
    ax_map.plot([bar_x0, bar_x1], [bar_y, bar_y],
                color="black", linewidth=2.0, zorder=9, solid_capstyle="butt")
    ax_map.text((bar_x0 + bar_x1) / 2, bar_y + 0.012 * (ylim[1] - ylim[0]),
                "100 km", ha="center", va="bottom", fontsize=6, zorder=9)

    legend_handles = [
        mlines.Line2D([], [], color="#1A5276", linewidth=0.9, label="Study basin"),
        mpatches.Patch(facecolor="none", edgecolor="none", label=basemap_label),
    ]
    ax_map.legend(handles=legend_handles, loc="upper right",
                  fontsize=6, frameon=True, fancybox=False,
                  edgecolor="#bbbbbb", facecolor="white",
                  bbox_to_anchor=(1.0, 0.83))
    ax_map.set_title("(a) Study Basins — Nepal", fontsize=TITLE_SIZE)

    # ── BOTTOM: Observation availability ────────────────────────────────────
    train_start, train_end = SPLIT_DATES["train"]
    val_start,   val_end   = SPLIT_DATES["val"]
    test_start,  test_end  = SPLIT_DATES["test"]

    gauge_ids = sorted(
        [_norm_gauge_id(re.sub(r"^nepal_|_merged\.parquet$", "", p.name))
         for p in INPUT_DIR.glob("nepal_*_merged.parquet")],
        key=float,
    )
    site_map = {gid: i for i, gid in enumerate(gauge_ids)}

    ax_avail.axvspan(pd.Timestamp(val_start), pd.Timestamp(val_end),
                     color="#009E73", alpha=0.12, zorder=1,
                     label=f"Val ({val_start[:4]}–{val_end[:4]})")
    ax_avail.axvspan(pd.Timestamp(train_start), pd.Timestamp(train_end),
                     color="#F0A500", alpha=0.12, zorder=1,
                     label=f"Train ({train_start[:4]}–{train_end[:4]})")
    ax_avail.axvspan(pd.Timestamp(test_start), pd.Timestamp(test_end),
                     color="#0072B2", alpha=0.12, zorder=1,
                     label=f"Test ({test_start[:4]}–{test_end[:4]})")

    all_dates  = [pd.Timestamp(d) for d in
                  (train_start, train_end, val_start, val_end, test_start, test_end)]
    proj_start = min(all_dates)
    proj_end   = max(all_dates)

    for path in sorted(INPUT_DIR.glob("nepal_*_merged.parquet")):
        gauge_id = _norm_gauge_id(re.sub(r"^nepal_|_merged\.parquet$", "", path.name))
        df = pd.read_parquet(path)[["date", "qobs"]]
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= proj_start) & (df["date"] <= proj_end)].reset_index(drop=True)
        if df.empty:
            continue
        y = site_map[gauge_id]
        has_obs = df["qobs"].notna().values.astype(int)
        changes = np.diff(has_obs, prepend=0, append=0)
        starts  = np.where(changes == 1)[0]
        ends    = np.where(changes == -1)[0] - 1
        dates   = df["date"].values
        for s, e in zip(starts, ends):
            ax_avail.hlines(y, dates[s], dates[e],
                            linewidth=3.0, colors="#1A5276", zorder=5, alpha=1.0)

    ax_avail.set_xlim(proj_start, proj_end)
    boundary_years = sorted({
        int(train_start[:4]), int(train_end[:4]) + 1,
        int(val_start[:4]),   int(val_end[:4])   + 1,
        int(test_start[:4]),  int(test_end[:4])  + 1,
    })
    ax_avail.set_xticks([pd.Timestamp(f"{y}-01-01") for y in boundary_years])
    ax_avail.set_xticklabels(boundary_years)
    ax_avail.set_yticks(range(len(gauge_ids)))
    ax_avail.set_yticklabels(
        [SITE_LABELS.get(gid, f"Site {gid}") for gid in gauge_ids],
        fontsize=FONT_SIZE - 1,
    )
    ax_avail.set_xlabel("Year")
    ax_avail.set_title("(b) Observed Streamflow Availability", fontsize=TITLE_SIZE)
    ax_avail.grid(True, axis="x", linestyle="--", alpha=0.4, linewidth=0.5, zorder=2)
    ax_avail.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False,
                    fontsize=FONT_SIZE - 1)

    plt.tight_layout(h_pad=1.0)
    plt.savefig(FIGURES_DIR / "fig1_basin_overview.png", dpi=DPI, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig1_basin_overview.svg", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved fig1_basin_overview.png")


# ---------------------------------------------------------------------------
# Figure 3 — Time series for representative basin (3×2 compound layout)
# ---------------------------------------------------------------------------

def fig3_timeseries(preds: dict, metrics_df: pd.DataFrame,
                    show_years: int = 1) -> None:
    """3-row × 2-col time series for the representative (median-NSE) basin.

    Column 0: LSTM variants  |  Column 1: Transformer variants
    Row 0: pure DL (LSTM or Transformer)
    Row 1: GloFAS + GloFAS+{LSTM|Transformer}
    Row 2: GRFR  + GRFR+{LSTM|Transformer}
    """
    gauge_nse = (
        metrics_df.groupby("gauge_id")["nse"]
        .mean()
        .sort_values()
    )
    median_idx = len(gauge_nse) // 2
    rep_gauge = gauge_nse.index[median_idx]
    site_label = SITE_LABELS.get(rep_gauge, f"Site {rep_gauge}")
    print(f"  Representative basin for Fig 3: {site_label} "
          f"(mean NSE = {gauge_nse[rep_gauge]:.3f})")

    test_start = pd.Timestamp(SPLIT_DATES["test"][0])
    plot_end   = test_start + pd.DateOffset(years=show_years)

    fig, axes = plt.subplots(3, 2, figsize=(7, 7), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.07, wspace=0.06)

    def _get_nse(model_name: str) -> float:
        sub = metrics_df[
            (metrics_df["model"] == model_name) & (metrics_df["gauge_id"] == rep_gauge)
        ]["nse"]
        return float(sub.values[0]) if len(sub) > 0 else float("nan")

    def _get_df(model_name: str) -> pd.DataFrame | None:
        full = preds.get(model_name, {}).get(rep_gauge)
        if full is None or full.empty:
            return None
        return full[(full["date"] >= test_start) & (full["date"] < plot_end)]

    # Each cell: (row, col, list of model_names to plot)
    layout = [
        (0, 0, ["lstm"]),
        (0, 1, ["transformer"]),
        (1, 0, ["glofas", "glofas_lstm"]),
        (1, 1, ["glofas", "glofas_transformer"]),
        (2, 0, ["grfr", "grfr_lstm"]),
        (2, 1, ["grfr", "grfr_transformer"]),
    ]

    for row, col, models in layout:
        ax = axes[row, col]

        # Plot observed from first available model
        obs_df = _get_df(models[0])
        if obs_df is not None and not obs_df.empty:
            ax.plot(obs_df["date"], obs_df["qobs"],
                    color="black", linewidth=0.8, zorder=4, label="Observed")

        legend_lines = [plt.Line2D([0], [0], color="black", linewidth=0.8, label="Observed")]

        for model_name in models:
            df = _get_df(model_name)
            if df is None or df.empty:
                continue
            nse_val = _get_nse(model_name)
            ax.plot(df["date"], df["qsim"],
                    color=COLORS[model_name],
                    linestyle=LINESTYLES[model_name],
                    linewidth=0.9, zorder=3)
            lbl = f"{MODEL_LABELS[model_name]} (NSE={nse_val:.2f})"
            legend_lines.append(
                plt.Line2D([0], [0], color=COLORS[model_name],
                           linestyle=LINESTYLES[model_name],
                           linewidth=0.9, label=lbl)
            )

        ax.legend(handles=legend_lines, loc="upper right",
                  fontsize=FONT_SIZE - 2, frameon=True, framealpha=0.85,
                  edgecolor="none", handlelength=1.5,
                  borderpad=0.3, labelspacing=0.2)
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.4)

        # x-ticks only on bottom row
        if row == 2:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
            ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(
                ax.xaxis.get_major_locator()))
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right",
                     fontsize=FONT_SIZE - 1)

        # y-label only on left column
        if col == 0:
            ax.set_ylabel("Flow (mm/day)", fontsize=FONT_SIZE)

    fig.suptitle(
        f"Time series — {site_label}  "
        f"({test_start.year}–{(test_start + pd.DateOffset(years=show_years)).year})",
        fontsize=TITLE_SIZE, y=0.995,
    )
    plt.savefig(FIGURES_DIR / "fig3_timeseries.png", dpi=DPI, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig3_timeseries.svg", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved fig3_timeseries.png")


# ---------------------------------------------------------------------------
# Figure 4 — CDF of NSE and KGE
# ---------------------------------------------------------------------------

def fig4_cdf_metrics(metrics_df: pd.DataFrame) -> None:
    """2×2 CDF grid: rows = model groups, columns = NSE / KGE.

    Row 0: Physical (GloFAS, GRFR) + Deep learning (LSTM, Transformer)
    Row 1: Physical (GloFAS, GRFR) + Post-processors (4 hybrids)
    """
    ROW_GROUPS = [
        ("Physical & Deep learning",  ["glofas", "grfr", "lstm", "transformer"]),
        ("Physical & Post-processors", ["glofas", "grfr",
                                        "glofas_lstm", "glofas_transformer",
                                        "grfr_lstm", "grfr_transformer"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharey=True)
    fig.subplots_adjust(hspace=0.10, wspace=0.06)

    for row_i, (row_title, row_models) in enumerate(ROW_GROUPS):
        for col_i, metric in enumerate(["nse", "kge"]):
            ax = axes[row_i, col_i]

            handles_row = []
            for model_name in row_models:
                sub = metrics_df[metrics_df["model"] == model_name][metric].dropna().values
                if len(sub) == 0:
                    continue
                vals, probs = _cdf(sub)
                median_val = float(np.median(sub))
                lbl = f"{MODEL_LABELS[model_name]} ({median_val:.2f})"
                line, = ax.plot(vals, probs,
                                color=COLORS[model_name],
                                linestyle=LINESTYLES[model_name],
                                linewidth=1.1,
                                marker="o", markersize=3,
                                label=lbl)
                handles_row.append(line)

            # Red dashed median reference
            x_min = metrics_df[metric].dropna().min()
            ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8,
                       alpha=0.75, zorder=1)
            ax.text(x_min - 0.03, 0.52, "median",
                    fontsize=FONT_SIZE - 2, color="red", va="bottom")

            ax.set_xlim(x_min - 0.05, 1.02)
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.4)

            # x-label only on bottom row
            if row_i == 1:
                ax.set_xlabel(metric.upper())
            else:
                ax.set_xticklabels([])

            # y-label only on left column
            if col_i == 0:
                ax.set_ylabel("Cumulative probability")

            # Panel title on top row only
            if row_i == 0:
                ax.set_title(f"CDF of {metric.upper()}", fontsize=TITLE_SIZE)

            # Row label as small text inside top-left
            if col_i == 0:
                ax.text(0.02, 0.98, row_title, transform=ax.transAxes,
                        fontsize=FONT_SIZE - 2, va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  alpha=0.7, edgecolor="none"))

            # Legend inside right panel of each row
            if col_i == 1:
                ax.legend(handles=handles_row, loc="lower right",
                          fontsize=FONT_SIZE - 2, frameon=True, framealpha=0.85,
                          edgecolor="none", handlelength=1.5,
                          borderpad=0.3, labelspacing=0.2)

    plt.savefig(FIGURES_DIR / "fig4_cdf_metrics.png", dpi=DPI, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig4_cdf_metrics.svg", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved fig4_cdf_metrics.png")


# ---------------------------------------------------------------------------
# Figure 5 — Bias analysis (four panels, shared x-axis)
# ---------------------------------------------------------------------------

def fig5_bias(metrics_df: pd.DataFrame) -> None:
    bias_cols = {
        "pbias":      "Overall PBIAS (%)",
        "pbias_high": "High Flow PBIAS (top 10%)",
        "pbias_low":  "Low Flow PBIAS (bottom 30%)",
        "pbias_mid":  "Medium Flow PBIAS (30–90%)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    fig.subplots_adjust(hspace=0.15, wspace=0.25)

    for i, (ax, (col, title)) in enumerate(zip(axes.flatten(), bias_cols.items())):
        data = [metrics_df[metrics_df["model"] == m][col].dropna().values
                for m in DISPLAY_ORDER]

        bp = ax.boxplot(
            data,
            patch_artist=True,
            notch=False,
            widths=0.55,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
            flierprops=dict(marker="o", markersize=3, linestyle="none",
                            markeredgewidth=0.5),
        )
        for patch, model_name in zip(bp["boxes"], DISPLAY_ORDER):
            patch.set_facecolor(COLORS[model_name])
            patch.set_alpha(0.85)
            patch.set_edgecolor("gray")
            patch.set_linewidth(0.6)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_xticks(range(1, len(DISPLAY_ORDER) + 1))
        ax.set_title(title)
        ax.set_ylabel("PBIAS (%)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3, linewidth=0.4)

        # x-tick labels only on bottom row
        if i >= 2:
            ax.set_xticklabels([MODEL_LABELS[m] for m in DISPLAY_ORDER],
                               rotation=30, ha="right", fontsize=FONT_SIZE - 1)
        else:
            ax.set_xticklabels([])

    plt.savefig(FIGURES_DIR / "fig5_bias.png", dpi=DPI, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig5_bias.svg", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved fig5_bias.png")


# ---------------------------------------------------------------------------
# Figure 6 — Peak flow CDF (standard orientation: x=flow, y=prob)
# ---------------------------------------------------------------------------

def fig6_peak_flow(preds: dict) -> None:
    """Two-panel peak-flow CDF: physical+DL (left) and physical+post-processors (right)."""
    train_qobs = load_training_qobs()
    train_peak_ref = max(s.max() for s in train_qobs.values()) if train_qobs else None

    PANEL_GROUPS = [
        ("Physical & Deep learning",   ["glofas", "grfr", "lstm", "transformer"]),
        ("Physical & Post-processors", ["glofas", "grfr",
                                        "glofas_lstm", "glofas_transformer",
                                        "grfr_lstm", "grfr_transformer"]),
    ]

    # Pre-compute observed peaks and per-model peaks once
    obs_peaks = np.array([df["qobs"].max() for df in preds["lstm"].values()])
    model_peaks: dict[str, np.ndarray] = {}
    for mn in DISPLAY_ORDER:
        peaks = [df["qsim"].max() for df in preds.get(mn, {}).values() if len(df) > 0]
        if peaks:
            model_peaks[mn] = np.array(peaks)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    fig.subplots_adjust(wspace=0.05)

    for col_i, (panel_title, panel_models) in enumerate(PANEL_GROUPS):
        ax = axes[col_i]

        for model_name in panel_models:
            if model_name not in model_peaks:
                continue
            vals, probs = _cdf(model_peaks[model_name])
            ax.plot(vals, probs,
                    color=COLORS[model_name],
                    linestyle=LINESTYLES[model_name],
                    linewidth=1.1, marker="o", markersize=3,
                    label=MODEL_LABELS[model_name])

        # Observed in both panels
        if len(obs_peaks) > 0:
            vals, probs = _cdf(obs_peaks)
            ax.plot(vals, probs, color="gray", linestyle="-", linewidth=1.5,
                    marker="o", markersize=3, label="Observed (test)", zorder=4)

        # Train peak annotation (not in legend)
        if train_peak_ref is not None:
            ax.axvline(train_peak_ref, color="red", linestyle="--", linewidth=0.9)
            ax.text(train_peak_ref * 1.015, 0.04,
                    f"Train peak\n({train_peak_ref:.1f} mm/d)",
                    rotation=90, va="bottom", ha="left",
                    fontsize=FONT_SIZE - 1, color="red")

        ax.set_xlabel("Peak flow (mm/day)")
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        ax.set_title(panel_title, fontsize=TITLE_SIZE)
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.4)
        ax.legend(frameon=False, fontsize=FONT_SIZE - 1)

        if col_i == 0:
            ax.set_ylabel("Cumulative probability")

    fig.suptitle("Peak flow distribution across basins — test period",
                 fontsize=TITLE_SIZE, y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_peak_flow.png", dpi=DPI, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig6_peak_flow.svg", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved fig6_peak_flow.png")


# ---------------------------------------------------------------------------
# Figure 7 — NSE maps (8 panels, fixed gauge_id matching)
# ---------------------------------------------------------------------------

def _detect_gauge_col(gdf: gpd.GeoDataFrame) -> str | None:
    """Try to find the column in the shapefile that holds gauge IDs."""
    candidates = ["gauge_id", "station", "STATION", "ID", "id", "GID",
                  "GaugeID", "gauge", "GAUGE_ID", "site_no"]
    for col in candidates:
        if col in gdf.columns:
            return col
    print(f"  Shapefile columns: {gdf.columns.tolist()}")
    return None


def fig7_nse_maps(metrics_df: pd.DataFrame, basins_gdf: gpd.GeoDataFrame) -> None:
    if basins_gdf.crs and not basins_gdf.crs.is_geographic:
        basins_gdf = basins_gdf.to_crs(epsg=4326)

    gauge_col = _detect_gauge_col(basins_gdf)

    countries = _load_country_boundaries()
    nepal_gdf = None
    if countries is not None:
        name_col = next((c for c in ["NAME", "ADMIN", "name"] if c in countries.columns), None)
        if name_col:
            nepal_gdf = countries[countries[name_col] == "Nepal"]

    nse_all = metrics_df["nse"].dropna()
    vmin, vmax = max(nse_all.min(), -0.5), 1.0
    cmap = plt.cm.RdYlGn
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Wider, less tall — model labels embedded inside panels
    fig, axes = plt.subplots(4, 2, figsize=(8, 6))
    fig.subplots_adjust(bottom=0.08, hspace=0.03, wspace=0.03)
    axes_flat = axes.flatten()

    for i, model_name in enumerate(DISPLAY_ORDER):
        ax = axes_flat[i]

        if nepal_gdf is not None:
            nepal_gdf.plot(ax=ax, color="#EEEEEE", edgecolor="#AAAAAA",
                           linewidth=0.5, zorder=0)

        if gauge_col and gauge_col in basins_gdf.columns:
            gdf_plot = basins_gdf.copy()
            # Fix: normalise gauge_id on both sides to avoid float/string mismatch
            nse_lookup = metrics_df[metrics_df["model"] == model_name].copy()
            nse_lookup["_key"] = nse_lookup["gauge_id"].apply(_norm_gauge_id)
            nse_dict = nse_lookup.set_index("_key")["nse"].to_dict()
            gdf_plot["nse"] = gdf_plot[gauge_col].apply(_norm_gauge_id).map(nse_dict)
            gdf_plot.plot(column="nse", ax=ax, cmap=cmap, norm=norm,
                          edgecolor="gray", linewidth=0.4, zorder=1,
                          missing_kwds={"color": "lightgray"})
        else:
            basins_gdf.plot(ax=ax, color="lightblue", edgecolor="gray",
                            linewidth=0.4, zorder=1)

        # Title embedded inside panel — top right
        ax.text(0.97, 0.97, MODEL_LABELS[model_name],
                transform=ax.transAxes, ha="right", va="top",
                fontsize=FONT_SIZE - 1,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          alpha=0.85, edgecolor="none"))

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor("black")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.15, 0.03, 0.70, 0.018])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("NSE", fontsize=FONT_SIZE, labelpad=3)
    cbar.ax.tick_params(labelsize=FONT_SIZE)

    fig.suptitle("NSE — test period (2005–2014)", y=0.995, fontsize=TITLE_SIZE + 1)
    plt.savefig(FIGURES_DIR / "fig7_nse_maps.png", dpi=DPI, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig7_nse_maps.svg", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved fig7_nse_maps.png")


# ---------------------------------------------------------------------------
# Figure 8 — Training and validation loss curves
# ---------------------------------------------------------------------------

def fig8_loss_curves() -> None:
    """Plot epoch vs MSE loss (train and val) for all 6 deep learning models."""
    _MODEL_DIR = Path("output/model")
    dl_models = [
        "lstm", "transformer",
        "glofas_lstm", "grfr_lstm",
        "glofas_transformer", "grfr_transformer",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes_flat = axes.flatten()

    for i, model_name in enumerate(dl_models):
        ax = axes_flat[i]
        loss_path = _MODEL_DIR / f"{model_name}_loss_curves.parquet"

        if not loss_path.exists():
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=FONT_SIZE, color="gray")
            ax.set_title(MODEL_LABELS[model_name], fontsize=TITLE_SIZE)
            ax.set_xlabel("Epoch", fontsize=FONT_SIZE)
            ax.set_ylabel("MSE Loss", fontsize=FONT_SIZE)
            continue

        df = pd.read_parquet(loss_path)
        ax.plot(df["epoch"], df["train_loss"],
                color=COLORS[model_name], linestyle="-", linewidth=1.2, label="Train")
        ax.plot(df["epoch"], df["val_loss"],
                color=COLORS[model_name], linestyle="--", linewidth=1.2,
                alpha=0.7, label="Validation")

        ax.set_title(MODEL_LABELS[model_name], fontsize=TITLE_SIZE)
        ax.set_xlabel("Epoch", fontsize=FONT_SIZE)
        ax.set_ylabel("MSE Loss", fontsize=FONT_SIZE)
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.4)
        ax.legend(fontsize=FONT_SIZE - 1, frameon=False)

    fig.suptitle("Training and Validation Loss Curves", fontsize=TITLE_SIZE + 1)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig8_loss_curves.png", dpi=DPI, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig8_loss_curves.svg", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved fig8_loss_curves.png")


# ---------------------------------------------------------------------------
# Figure 9 — NSE vs basin properties
# ---------------------------------------------------------------------------

def _load_basin_properties() -> pd.DataFrame:
    """Return per-gauge static and dynamic basin properties for fig9."""
    sta = pd.read_csv(STATIONS_CSV)
    sta["gauge_id"] = sta["station"].apply(_norm_gauge_id)

    rows = []
    for path in sorted(INPUT_DIR.glob("nepal_*_merged.parquet")):
        gid = _norm_gauge_id(re.sub(r"^nepal_|_merged\.parquet$", "", path.name))
        df = pd.read_parquet(path)[["qobs", "frac_snow"]].dropna(subset=["qobs"])
        rows.append({
            "gauge_id":  gid,
            "mean_q":    df["qobs"].mean(),
            "frac_snow": df["frac_snow"].mean(),
        })

    dyn = pd.DataFrame(rows)
    return sta.merge(dyn, on="gauge_id")


def fig9_basin_properties(metrics_df: pd.DataFrame) -> None:
    """3×4 scatter grid: rows = model groups, columns = basin properties.

    Row 0: Physical models (GloFAS, GRFR)
    Row 1: Deep learning (LSTM, Transformer)
    Row 2: Post-processors / hybrids (4 models)
    """
    props = _load_basin_properties()

    PROP_COLS = [
        ("drainage_area_km2", "Drainage area (km²)",    True),
        ("elevation_m",       "Mean elevation (m)",      False),
        ("mean_q",            "Mean discharge (mm/day)", True),
        ("frac_snow",         "Snow fraction (–)",       False),
    ]

    ROW_GROUPS = [
        ("Physical",        ["glofas", "grfr"]),
        ("Deep learning",   ["lstm", "transformer"]),
        ("Post-processor",  ["glofas_lstm", "glofas_transformer",
                             "grfr_lstm", "grfr_transformer"]),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(11, 8),
                             sharey=True, sharex="col")
    fig.subplots_adjust(hspace=0.10, wspace=0.08)

    for row_i, (row_label, row_models) in enumerate(ROW_GROUPS):
        for col_i, (prop_col, prop_label, use_log) in enumerate(PROP_COLS):
            ax = axes[row_i, col_i]

            scatter_handles = []
            for model_name in row_models:
                sub = metrics_df[metrics_df["model"] == model_name].copy()
                sub = sub.merge(props[["gauge_id", prop_col]],
                                on="gauge_id", how="left").dropna(subset=["nse", prop_col])
                if sub.empty:
                    continue
                sc = ax.scatter(sub[prop_col].values, sub["nse"].values,
                                color=COLORS[model_name], s=25, alpha=0.8, zorder=3,
                                label=MODEL_LABELS[model_name])
                scatter_handles.append(sc)

            if use_log:
                ax.set_xscale("log")
            ax.axhline(0, color="gray", linewidth=0.6, linestyle=":", zorder=1)
            ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.4)

            # x-label only on bottom row
            if row_i == 2:
                ax.set_xlabel(prop_label, fontsize=FONT_SIZE)

            # y-label (row group name) only on left column
            if col_i == 0:
                ax.set_ylabel(f"{row_label}\nNSE", fontsize=FONT_SIZE)

        # Legend embedded inside last column panel of each row
        leg_ax = axes[row_i, 3]
        handles_leg, labels_leg = leg_ax.get_legend_handles_labels()
        if not handles_leg:
            # collect from any panel in the row
            handles_leg, labels_leg = axes[row_i, 0].get_legend_handles_labels()
        if handles_leg:
            leg_ax.legend(handles=handles_leg, labels=labels_leg,
                          loc="upper right", fontsize=FONT_SIZE - 1,
                          frameon=True, framealpha=0.85, edgecolor="none",
                          handletextpad=0.4, labelspacing=0.3)

    fig.suptitle("NSE vs basin characteristics — test period (2005–2014)",
                 fontsize=TITLE_SIZE, y=1.005)

    plt.savefig(FIGURES_DIR / "fig9_basin_properties.png", dpi=DPI, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig9_basin_properties.svg", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved fig9_basin_properties.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading shapefile...")
    basins_gdf = gpd.read_file(SHAPEFILE_PATH)
    print(f"  Basins shapefile: {len(basins_gdf)} features, CRS={basins_gdf.crs}")
    print(f"  Columns: {basins_gdf.columns.tolist()}")

    print("Loading predictions...")
    preds = load_all_predictions()
    for model_name, gdict in preds.items():
        print(f"  {model_name}: {len(gdict)} gauges")

    print("Computing metrics...")
    metrics_df = compute_all_metrics(preds)
    metrics_df.to_parquet(METRICS_PATH, index=False)
    print(f"  Metrics saved to {METRICS_PATH}  ({len(metrics_df)} rows)")
    summary = metrics_df.groupby("model")[["nse", "kge", "rmse"]].median().round(3)
    print("\nMedian metrics (test period):\n", summary.to_string())

    print("\nGenerating figures...")
    fig1_basin_overview(basins_gdf)
    fig3_timeseries(preds, metrics_df, show_years=1)
    fig4_cdf_metrics(metrics_df)
    fig5_bias(metrics_df)
    fig6_peak_flow(preds)
    fig7_nse_maps(metrics_df, basins_gdf)
    fig8_loss_curves()
    fig9_basin_properties(metrics_df)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
