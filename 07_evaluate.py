# Evaluate all eight models against observed streamflow during the test period
# (2005–2014) and produce seven publication-quality figures.
#
# Models evaluated:
#   Physical (baseline): GloFAS, GRFR
#   Deep learning:       LSTM, Transformer
#   Hybrid:              GloFAS+LSTM, GRFR+LSTM, GloFAS+Transformer, GRFR+Transformer
#
# Outputs:
#   output/metrics.parquet       — per-model per-gauge metrics table
#   output/figures/fig1_*.png    — seven figures at 300 dpi

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

# ---------------------------------------------------------------------------
# Styling — Okabe-Ito color-blind-friendly palette
# ---------------------------------------------------------------------------

MODEL_NAMES = [
    "glofas", "grfr", "lstm", "transformer",
    "glofas_lstm", "grfr_lstm", "glofas_transformer", "grfr_transformer",
]
MODEL_LABELS = {
    "glofas": "GloFAS",
    "grfr": "GRFR",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "glofas_lstm": "GloFAS+LSTM",
    "grfr_lstm": "GRFR+LSTM",
    "glofas_transformer": "GloFAS+Transformer",
    "grfr_transformer": "GRFR+Transformer",
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
# Data loading
# ---------------------------------------------------------------------------

def _gauge_id_from_filename(path: Path) -> str:
    """Extract gauge_id from a prediction filename, e.g. nepal_120_lstm.parquet → '120'."""
    return re.sub(r"^nepal_|_\w+\.parquet$", "", path.name)


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
        for path in sorted((PRED_DIR / folder).glob("*.parquet")):
            gauge_id = _gauge_id_from_filename(path)
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
            df = df.rename(columns={sim_col: "qsim"})[["date", "qobs", "qsim"]].dropna()
            gauge_dict[gauge_id] = df
        preds[model_name] = gauge_dict

    # --- Physical model baselines ---
    # Load qobs from lstm predictions (same observed values, independent of model)
    lstm_qobs = {gid: df.set_index("date")["qobs"] for gid, df in preds["lstm"].items()}

    for model_name, phys_path in [("glofas", GLOFAS_PATH), ("grfr", GRFR_PATH)]:
        raw = pd.read_parquet(phys_path)
        raw["date"] = pd.to_datetime(raw["date"], format="mixed")
        raw = raw[(raw["date"] >= test_start) & (raw["date"] <= test_end)]
        # Wide → long
        raw = raw.melt(id_vars="date", var_name="gauge_id", value_name="qsim")

        gauge_dict = {}
        for gauge_id, group in raw.groupby("gauge_id"):
            gid = str(gauge_id)
            if gid not in lstm_qobs:
                continue
            df = group[["date", "qsim"]].copy().set_index("date")
            df["qobs"] = lstm_qobs[gid]
            df = df.dropna().reset_index()
            gauge_dict[gid] = df[["date", "qobs", "qsim"]]
        preds[model_name] = gauge_dict

    return preds


def load_training_qobs() -> dict[str, pd.Series]:
    """Load qobs for the training period from merged input parquets.

    Returns:
        dict gauge_id → Series(date, qobs) for training period.
    """
    train_start, train_end = SPLIT_DATES["train"]
    result = {}
    for path in sorted(INPUT_DIR.glob("*.parquet")):
        gauge_id = re.sub(r"^nepal_|_merged\.parquet$", "", path.name)
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
# Figure 1 — Basin location map
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


def fig1_basin_map(basins_gdf: gpd.GeoDataFrame) -> None:
    import matplotlib.lines as mlines

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # Reproject to geographic CRS if needed
    if basins_gdf.crs and not basins_gdf.crs.is_geographic:
        basins_gdf = basins_gdf.to_crs(epsg=4326)

    # Tight bounding box — focus on Nepal only
    bounds = basins_gdf.total_bounds  # [minx, miny, maxx, maxy]
    pad_left, pad_right, pad_bottom, pad_top = 0.5, 0.3, 0.5, 0.5
    xlim = (bounds[0] - pad_left, bounds[2] + pad_right)
    ylim = (bounds[1] - pad_bottom, bounds[3] + pad_top)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # --- Elevation terrain basemap (contextily) ---
    basemap_label = "Basemap: OpenTopoMap"
    try:
        import contextily as ctx
        for provider, label in [
            (ctx.providers.OpenTopoMap,          "Basemap: OpenTopoMap"),
            (ctx.providers.Esri.WorldShadedRelief, "Basemap: Esri WorldShadedRelief"),
        ]:
            try:
                ctx.add_basemap(ax, crs="EPSG:4326", source=provider,
                                zoom=9, attribution=False, alpha=0.5)
                basemap_label = label
                break
            except Exception:
                continue
    except Exception as exc:
        print(f"  Note: elevation basemap not added — {exc}")
        ax.set_facecolor("#e8e0d8")
        basemap_label = "No basemap"

    # --- Country outlines (Natural Earth 50 m) ---
    countries = _load_country_boundaries()
    name_col = None
    if countries is not None:
        name_col = next((c for c in ["NAME", "ADMIN", "name"] if c in countries.columns), None)
        clip = countries.cx[xlim[0]:xlim[1], ylim[0]:ylim[1]]

        # All visible countries: no fill, just gray edges
        clip.plot(ax=ax, facecolor="none", edgecolor="#888888",
                  linewidth=0.4, zorder=2)

        # Nepal: slightly thicker border
        if name_col:
            nepal = countries[countries[name_col] == "Nepal"]
            if len(nepal) > 0:
                nepal.plot(ax=ax, facecolor="none", edgecolor="#333333",
                           linewidth=1.2, zorder=3)

        # --- Country name labels ---
        if name_col:
            # Only label countries with meaningful area in the clip
            area_threshold = (xlim[1] - xlim[0]) * (ylim[1] - ylim[0]) * 0.01
            for _, row in clip.iterrows():
                if row.geometry.area < area_threshold:
                    continue
                cx_c = row.geometry.centroid.x
                cy_c = row.geometry.centroid.y
                # Clip centroid to be inside the map bounds and near nepal border to avoid far-out labels
                cx_c = max(xlim[0] + 0.0, min(xlim[1] - 0.0, cx_c))
                cy_c = max(ylim[0] + 0.0, min(ylim[1] - 0.0, cy_c))
                ax.text(cx_c, cy_c, row[name_col],
                        fontsize=6, color="#444444", ha="center", va="center",
                        style="italic", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                                  alpha=0.55, edgecolor="none"))

    # --- Study basin polygons — outline only, no fill so terrain is visible ---
    basins_gdf.plot(ax=ax, facecolor="none", edgecolor="#1A5276",
                    linewidth=1, zorder=4)

    # --- Basin gauge ID labels ---
    texts = []
    for _, row in basins_gdf.iterrows():
        cx_b = row.geometry.centroid.x
        cy_b = row.geometry.centroid.y
        t = ax.text(cx_b, cy_b, str(row["gauge_id"]),
                    fontsize=5.5, ha="center", va="center", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              alpha=0.8, edgecolor="none"))
        texts.append(t)

    # Use adjustText if available to push overlapping labels apart
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, expand_text=(1.2, 1.4),
                    arrowprops=dict(arrowstyle="-", color="#555555", lw=0.4))
    except ImportError:
        pass  # white backgrounds provide sufficient readability without adjustment

    # Axes decoration
    ax.set_xlabel("Longitude (°E)", fontsize=FONT_SIZE)
    ax.set_ylabel("Latitude (°N)", fontsize=FONT_SIZE)
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.4, zorder=5)
    ax.tick_params(labelsize=FONT_SIZE)

    # --- Combined top-right: north arrow + scale bar + legend ---

    # North arrow (axes fraction coordinates, top-right area)
    ax.annotate("N", xy=(0.955, 0.955), xycoords="axes fraction",
                fontsize=9, ha="center", va="bottom", fontweight="bold", zorder=9)
    ax.annotate("", xy=(0.955, 0.955), xytext=(0.955, 0.905), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3), zorder=9)

    # Scale bar: 100 km expressed in degrees at ~28°N, placed just below the arrow
    km_per_deg = 111.32 * np.cos(np.radians(28))
    bar_deg = 100 / km_per_deg
    # Right-align scale bar: ends at 97% of x range
    bar_x1 = xlim[0] + 0.97 * (xlim[1] - xlim[0])
    bar_x0 = bar_x1 - bar_deg
    bar_y  = ylim[0] + 0.855 * (ylim[1] - ylim[0])
    ax.plot([bar_x0, bar_x1], [bar_y, bar_y],
            color="black", linewidth=2.0, zorder=9, solid_capstyle="butt",
            transform=ax.transData)
    ax.text((bar_x0 + bar_x1) / 2, bar_y + 0.012 * (ylim[1] - ylim[0]),
            "100 km", ha="center", va="bottom", fontsize=6, zorder=9)

    # Legend in top-right corner (below north arrow + scale)
    legend_handles = [
        mlines.Line2D([], [], color="#1A5276", linewidth=0.9, label="Study basin"),
        mpatches.Patch(facecolor="none", edgecolor="none", label=basemap_label),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=6, frameon=True, fancybox=False,
              edgecolor="#bbbbbb", facecolor="white",
              bbox_to_anchor=(1.0, 0.83))

    ax.set_title("Study Basins — Nepal", fontsize=TITLE_SIZE)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_basin_map.png", dpi=DPI, bbox_inches="tight")
    # also save svg for vector graphics version
    plt.savefig(FIGURES_DIR / "fig1_basin_map.svg", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved fig1_basin_map.png")


# ---------------------------------------------------------------------------
# Figure 2 — Observation availability
# ---------------------------------------------------------------------------

def fig2_obs_availability() -> None:
    train_start, train_end = SPLIT_DATES["train"]
    val_start,   val_end   = SPLIT_DATES["val"]
    test_start,  test_end  = SPLIT_DATES["test"]

    gauge_ids = sorted(
        [re.sub(r"^nepal_|_merged\.parquet$", "", p.name) for p in INPUT_DIR.glob("*.parquet")],
        key=float,
    )
    site_map = {gid: i for i, gid in enumerate(gauge_ids)}

    fig, ax = plt.subplots(figsize=(8, 5))

    # Period highlights — keep alpha low and use zorder=1 so hlines sit on top
    ax.axvspan(pd.Timestamp(val_start), pd.Timestamp(val_end),
            color="#009E73", alpha=0.12, zorder=1,
            label=f"Val ({val_start[:4]}–{val_end[:4]})")
    
    ax.axvspan(pd.Timestamp(train_start), pd.Timestamp(train_end),
               color="#F0A500", alpha=0.12, zorder=1,
               label=f"Train ({train_start[:4]}–{train_end[:4]})")

    ax.axvspan(pd.Timestamp(test_start), pd.Timestamp(test_end),
               color="#0072B2", alpha=0.12, zorder=1,
               label=f"Test ({test_start[:4]}–{test_end[:4]})")

    # Availability: solid horizontal bars per contiguous observation period
    # Use the actual earliest and latest dates across all three splits,
    # since train/val order can vary (val may precede train chronologically).
    all_dates  = [pd.Timestamp(d) for d in
                  (train_start, train_end, val_start, val_end, test_start, test_end)]
    proj_start = min(all_dates)
    proj_end   = max(all_dates)

    for path in sorted(INPUT_DIR.glob("*.parquet")):
        gauge_id = re.sub(r"^nepal_|_merged\.parquet$", "", path.name)
        df = pd.read_parquet(path)[["date", "qobs"]]
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= proj_start) & (df["date"] <= proj_end)].reset_index(drop=True)
        if df.empty:
            continue

        y = site_map[gauge_id]
        has_obs = df["qobs"].notna().values.astype(int)
        # Detect contiguous runs of non-null qobs
        changes = np.diff(has_obs, prepend=0, append=0)
        starts  = np.where(changes == 1)[0]
        ends    = np.where(changes == -1)[0] - 1
        dates   = df["date"].values

        for s, e in zip(starts, ends):
            ax.hlines(y, dates[s], dates[e],
                      linewidth=3.5, colors="#1A5276", zorder=5, alpha=1.0)

    # Axes — tick at every split boundary, sorted chronologically
    ax.set_xlim(proj_start, proj_end)
    boundary_years = sorted({
        int(train_start[:4]), int(train_end[:4]) + 1,
        int(val_start[:4]),   int(val_end[:4])   + 1,
        int(test_start[:4]),  int(test_end[:4])  + 1,
    })
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in boundary_years])
    ax.set_xticklabels(boundary_years)
    ax.set_yticks(range(len(gauge_ids)))
    ax.set_yticklabels([f"Site {gid}" for gid in gauge_ids])
    ax.set_xlabel("Year")
    ax.set_ylabel("Gauge Site")
    ax.set_title("Observed Streamflow Availability")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4, linewidth=0.5, zorder=2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_obs_availability.png", dpi=DPI, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig2_obs_availability.svg", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved fig2_obs_availability.png")


# ---------------------------------------------------------------------------
# Figure 3 — Time series for representative basin
# ---------------------------------------------------------------------------

def fig3_timeseries(preds: dict, metrics_df: pd.DataFrame,
                    show_years: int = 2) -> None:
    """Plot test-period time series for the representative (median-NSE) basin.

    Args:
        show_years: Number of years to display from the start of the test period.
                    Default 2 (2005–2006) gives a cleaner visual than all 10 years.
    """
    # Find gauge with median mean-NSE across all models
    gauge_nse = (
        metrics_df.groupby("gauge_id")["nse"]
        .mean()
        .sort_values()
    )
    median_idx = len(gauge_nse) // 2
    rep_gauge = gauge_nse.index[median_idx]
    print(f"  Representative basin for Fig 3: gauge {rep_gauge} "
          f"(mean NSE = {gauge_nse[rep_gauge]:.3f})")

    test_start = pd.Timestamp(SPLIT_DATES["test"][0])
    plot_end   = test_start + pd.DateOffset(years=show_years)

    # Tighter layout: 4 rows × 2 cols, smaller overall size
    fig, axes = plt.subplots(4, 2, figsize=(8, 8), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.10, wspace=0.08)
    axes_flat = axes.flatten()

    for i, model_name in enumerate(MODEL_NAMES):
        ax = axes_flat[i]
        full_df = preds.get(model_name, {}).get(rep_gauge)
        if full_df is None or full_df.empty:
            ax.set_visible(False)
            continue

        df = full_df[(full_df["date"] >= test_start) & (full_df["date"] < plot_end)]

        gauge_nse_val = metrics_df[
            (metrics_df["model"] == model_name) & (metrics_df["gauge_id"] == rep_gauge)
        ]["nse"]
        nse_val = gauge_nse_val.values[0] if len(gauge_nse_val) > 0 else float("nan")

        ax.plot(df["date"], df["qobs"], color="black", linewidth=0.7, zorder=3)
        ax.plot(df["date"], df["qsim"], color=COLORS[model_name],
                linewidth=0.9, linestyle=LINESTYLES[model_name], zorder=2)
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.4)

        # Model name + NSE inside top-left of panel
        label = f"{MODEL_LABELS[model_name]}\nNSE = {nse_val:.2f}"
        ax.text(0.02, 0.97, label, transform=ax.transAxes,
                fontsize=FONT_SIZE - 1, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          alpha=0.85, edgecolor="none"))

        # Obs-only legend inside top-right of every panel
        ax.legend([plt.Line2D([0], [0], color="black", linewidth=0.7)],
                  ["Observed"], loc="upper right",
                  fontsize=FONT_SIZE - 1, frameon=True, framealpha=0.85,
                  edgecolor="none", handlelength=1.2)

        # x-ticks only on bottom row
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
        ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(ax.xaxis.get_major_locator()))
        if i >= 6:
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right",
                     fontsize=FONT_SIZE - 1)
        if i % 2 == 0:
            ax.set_ylabel("Flow (mm/day)", fontsize=FONT_SIZE)

    fig.suptitle(
        f"Time series — Site {rep_gauge}  "
        f"({test_start.year}–{(test_start + pd.DateOffset(years=show_years)).year})",
        fontsize=TITLE_SIZE, y=0.98 # position right on top of panels
    )
    plt.savefig(FIGURES_DIR / "fig3_timeseries.png", dpi=DPI, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig3_timeseries.svg", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved fig3_timeseries.png")


# ---------------------------------------------------------------------------
# Figure 4 — CDF of NSE and KGE
# ---------------------------------------------------------------------------

def fig4_cdf_metrics(metrics_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for ax, metric in zip(axes, ["nse", "kge"]):
        for model_name in MODEL_NAMES:
            sub = metrics_df[metrics_df["model"] == model_name][metric].dropna().values
            if len(sub) == 0:
                continue
            vals, probs = _cdf(sub)
            ax.plot(vals, probs, color=COLORS[model_name],
                    linestyle=LINESTYLES[model_name], linewidth=1.2,
                    label=MODEL_LABELS[model_name])
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(metric.upper())
        ax.set_ylabel("Cumulative probability")
        ax.set_title(f"CDF of {metric.upper()} across basins")
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.4)

    # Legend outside right panel
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right",
               bbox_to_anchor=(1.22, 0.5), frameon=False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_cdf_metrics.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    plt.savefig(FIGURES_DIR / "fig4_cdf_metrics.svg", dpi=DPI, bbox_inches="tight")
    print("Saved fig4_cdf_metrics.png")


# ---------------------------------------------------------------------------
# Figure 5 — Bias analysis (four panels)
# ---------------------------------------------------------------------------

def fig5_bias(metrics_df: pd.DataFrame) -> None:
    bias_cols = {
        "pbias":      "Overall PBIAS (%)",
        "pbias_high": "High Flow PBIAS (top 10%)",
        "pbias_low":  "Low Flow PBIAS (bottom 30%)",
        "pbias_mid":  "Medium Flow PBIAS (30–90%)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes_flat = axes.flatten()

    for ax, (col, title) in zip(axes_flat, bias_cols.items()):
        data = [metrics_df[metrics_df["model"] == m][col].dropna().values
                for m in MODEL_NAMES]

        bp = ax.boxplot(
            data,
            patch_artist=True,       # filled boxes
            notch=False,
            widths=0.55,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
            flierprops=dict(marker="o", markersize=3, linestyle="none",
                            markeredgewidth=0.5),
        )
        # Color each box with the model's Okabe-Ito color
        for patch, model_name in zip(bp["boxes"], MODEL_NAMES):
            patch.set_facecolor(COLORS[model_name])
            patch.set_alpha(0.85)
            patch.set_edgecolor("gray")
            patch.set_linewidth(0.6)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_xticks(range(1, len(MODEL_NAMES) + 1))
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_NAMES],
                           rotation=30, ha="right", fontsize=FONT_SIZE - 1)
        ax.set_ylabel("PBIAS (%)")
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3, linewidth=0.4)

    # Shared color legend (model → color patch)
    legend_handles = [
        mpatches.Patch(facecolor=COLORS[m], edgecolor="gray",
                       linewidth=0.5, label=MODEL_LABELS[m])
        for m in MODEL_NAMES
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.05), frameon=False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_bias.png", dpi=DPI, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig5_bias.svg", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved fig5_bias.png")


# ---------------------------------------------------------------------------
# Figure 6 — Peak flow CDF
# ---------------------------------------------------------------------------

def fig6_peak_flow(preds: dict) -> None:
    # Training-period qobs peaks: max qobs across all basins, single scalar
    train_qobs = load_training_qobs()
    if train_qobs:
        train_peak_ref = max(s.max() for s in train_qobs.values())
    else:
        train_peak_ref = None

    fig, ax = plt.subplots(figsize=(6, 4))

    for model_name in MODEL_NAMES:
        gauge_peaks = []
        for gauge_id, df in preds.get(model_name, {}).items():
            if len(df) == 0:
                continue
            gauge_peaks.append(df["qsim"].max())
        if not gauge_peaks:
            continue
        vals, probs = _cdf(np.array(gauge_peaks))
        # Swap: cumulative probability on x-axis, peak flow on y-axis
        ax.plot(probs, vals, color=COLORS[model_name],
                linestyle=LINESTYLES[model_name], linewidth=1.2,
                label=MODEL_LABELS[model_name])

    # Observed peak CDF
    obs_peaks = []
    for gauge_id, df in preds["lstm"].items():
        obs_peaks.append(df["qobs"].max())
    if obs_peaks:
        vals, probs = _cdf(np.array(obs_peaks))
        ax.plot(probs, vals, color="gray", linestyle="-", linewidth=1.5,
                label="Observed (test)", zorder=4)

    # Training reference line (horizontal — constant flow value)
    if train_peak_ref is not None:
        ax.axhline(train_peak_ref, color="red", linestyle="--", linewidth=1.0,
                   label=f"Train peak max ({train_peak_ref:.1f} mm/d)")

    ax.set_xlabel("Cumulative probability")
    ax.set_ylabel("Peak flow (mm/day)")
    ax.set_xlim(0, 1)
    ax.set_title("Peak flow distribution across basins — test period")
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.4)
    ax.legend(frameon=False, fontsize=FONT_SIZE - 1)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_peak_flow.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved fig6_peak_flow.png")


# ---------------------------------------------------------------------------
# Figure 7 — NSE maps (8 panels)
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

    # Load Nepal outline for background context in each panel
    countries = _load_country_boundaries()
    nepal_gdf = None
    if countries is not None:
        name_col = next((c for c in ["NAME", "ADMIN", "name"] if c in countries.columns), None)
        if name_col:
            nepal_gdf = countries[countries[name_col] == "Nepal"]

    # Global NSE range for shared colorbar
    nse_all = metrics_df["nse"].dropna()
    vmin, vmax = max(nse_all.min(), -0.5), 1.0
    cmap = plt.cm.RdYlGn
    norm = Normalize(vmin=vmin, vmax=vmax)

    # 4 rows × 2 cols layout; reserve bottom space for horizontal colorbar
    fig, axes = plt.subplots(4, 2, figsize=(7, 9))
    # make more tighter layout
    fig.subplots_adjust(bottom=0.07, hspace=0.08, wspace=0.08)
    axes_flat = axes.flatten()

    for i, model_name in enumerate(MODEL_NAMES):
        ax = axes_flat[i]

        # Nepal outline as light background
        if nepal_gdf is not None:
            nepal_gdf.plot(ax=ax, color="#EEEEEE", edgecolor="#AAAAAA", # no fill
                           linewidth=0.5, zorder=0)

        # Basin polygons colored by NSE
        if gauge_col and gauge_col in basins_gdf.columns:
            gdf_plot = basins_gdf.copy()
            gdf_plot["nse"] = gdf_plot[gauge_col].astype(str).map(
                metrics_df[metrics_df["model"] == model_name].set_index("gauge_id")["nse"]
            )
            gdf_plot.plot(column="nse", ax=ax, cmap=cmap, norm=norm,
                          edgecolor="gray", linewidth=0.4, zorder=1,
                          missing_kwds={"color": "lightgray"})
        else:
            basins_gdf.plot(ax=ax, color="lightblue", edgecolor="gray",
                            linewidth=0.4, zorder=1)

        ax.set_title(MODEL_LABELS[model_name], fontsize=TITLE_SIZE, pad=3)

        # Keep rectangular frame; hide tick labels
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor("black")

    # Horizontal colorbar at bottom, sized to match 2-column width
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.15, 0.03, 0.70, 0.013])   # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("NSE", fontsize=FONT_SIZE, labelpad=3)
    cbar.ax.tick_params(labelsize=FONT_SIZE)

    fig.suptitle("NSE — test period (2005–2014)", y=0.99, fontsize=TITLE_SIZE + 1)
    plt.savefig(FIGURES_DIR / "fig7_nse_maps.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved fig7_nse_maps.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load shapefile ---
    print("Loading shapefile...")
    basins_gdf = gpd.read_file(SHAPEFILE_PATH)
    print(f"  Basins shapefile: {len(basins_gdf)} features, CRS={basins_gdf.crs}")
    print(f"  Columns: {basins_gdf.columns.tolist()}")

    # --- Load predictions ---
    print("Loading predictions...")
    preds = load_all_predictions()
    for model_name, gdict in preds.items():
        print(f"  {model_name}: {len(gdict)} gauges")

    # --- Compute metrics ---
    print("Computing metrics...")
    metrics_df = compute_all_metrics(preds)
    metrics_df.to_parquet(METRICS_PATH, index=False)
    print(f"  Metrics saved to {METRICS_PATH}  ({len(metrics_df)} rows)")
    # Print quick summary
    summary = metrics_df.groupby("model")[["nse", "kge", "rmse"]].median().round(3)
    print("\nMedian metrics (test period):\n", summary.to_string())

    # --- Generate figures ---
    print("\nGenerating figures...")
    fig1_basin_map(basins_gdf)
    fig2_obs_availability()
    fig3_timeseries(preds, metrics_df, show_years=2)
    fig4_cdf_metrics(metrics_df)
    fig5_bias(metrics_df)
    fig6_peak_flow(preds)
    fig7_nse_maps(metrics_df, basins_gdf)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
