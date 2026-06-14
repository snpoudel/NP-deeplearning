"""
05_get_alphaearth_embeddings.py

Extract Google AlphaEarth Foundations satellite embeddings for each Nepal
basin by sampling COG tiles directly from Google Cloud Storage using rasterio.
No GEE authentication required — uses the VM's service account for GCS reads.

Dataset: gs://alphaearth_foundations/satellite_embedding/v1/annual/
  - 64 bands A00–A63, signed int8 (-127 to 127; -128 = masked)
  - 10 m resolution, UTM-projected tiles (8192×8192 px each)
  - Annual: 2017–2024
  - De-quantization: ((raw / 127.5) ** 2) * sign(raw)  → float in [-1, 1]
  - License: CC-BY 4.0 (open access)

Strategy:
  For each basin, compute the polygon centroid, then for each year (2017–2024)
  find the tile containing that centroid, sample the pixel, and average across
  years to get a stable, time-invariant static representation.

Output:
    input/alphaearth_embeddings.parquet
    Columns: gauge_id (str), emb_0 … emb_63 (float32)

Usage:
    python preprocessing/05_get_alphaearth_embeddings.py
"""

from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
import rasterio.transform

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

SHAPEFILE_PATH  = Path("input/shapefile/merged_shapefile.shp")
INDEX_CACHE     = Path("/tmp/aef_index.parquet")     # downloaded once
OUT_PATH        = Path("input/alphaearth_embeddings.parquet")
GCS_INDEX_PATH  = "gs://alphaearth_foundations/satellite_embedding/v1/annual/aef_index.parquet"
GCS_BILLING     = "deep-camels-project-465420"       # requester-pays billing project
YEARS           = list(range(2017, 2025))            # 2017–2024 inclusive
N_BANDS         = 64
MASKED_VALUE    = -128                               # int8 NoData
OVERVIEW_LEVEL  = 6   # 1280m resolution; fast (~0.6s/read), appropriate for basin-scale features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_index() -> pd.DataFrame:
    """Download the GCS tile index parquet (cached at INDEX_CACHE)."""
    if INDEX_CACHE.exists():
        print(f"Using cached index at {INDEX_CACHE}")
    else:
        print(f"Downloading tile index from {GCS_INDEX_PATH} ...")
        ret = os.system(f"gsutil -u {GCS_BILLING} cp {GCS_INDEX_PATH} {INDEX_CACHE}")
        if ret != 0:
            raise RuntimeError("gsutil download failed — check GCS_BILLING project and permissions.")
    df = pd.read_parquet(INDEX_CACHE)
    return df


def find_tile(index: pd.DataFrame, lon: float, lat: float, year: int) -> str | None:
    """Return the GCS path of the tile that contains (lon, lat) for the given year."""
    mask = (
        (index["year"] == year) &
        (index["wgs84_west"]  <= lon) & (index["wgs84_east"]  > lon) &
        (index["wgs84_south"] <= lat) & (index["wgs84_north"] > lat)
    )
    hits = index[mask]
    if hits.empty:
        return None
    return hits.iloc[0]["path"]


def sample_tile(gcs_path: str, lon: float, lat: float) -> np.ndarray | None:
    """Read the 64-band pixel at (lon, lat) from a COG tile via /vsigs/."""
    vsigs_path = gcs_path.replace("gs://", "/vsigs/")
    try:
        env_opts = dict(
            GS_USER_PROJECT=GCS_BILLING,
            GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        )
        with rasterio.Env(**env_opts):
            with rasterio.open(vsigs_path, overview_level=OVERVIEW_LEVEL) as src:
                # Reproject lat/lon → tile CRS
                crs_str = src.crs.to_epsg()
                transformer = pyproj.Transformer.from_crs(
                    "EPSG:4326", f"EPSG:{crs_str}", always_xy=True
                )
                x_proj, y_proj = transformer.transform(lon, lat)

                # Convert projected coords to pixel row/col
                row, col = rasterio.transform.rowcol(src.transform, x_proj, y_proj)
                r, c = int(row), int(col)

                # Guard against out-of-bounds
                if not (0 <= r < src.height and 0 <= c < src.width):
                    return None

                # Read all 64 bands at the single pixel
                window = rasterio.windows.Window(c, r, 1, 1)
                data = src.read(window=window)   # (64, 1, 1), int8
                values = data[:, 0, 0].astype(np.float32)

                # Mask NoData pixels
                if np.any(values == MASKED_VALUE):
                    return None

                # De-quantize: int8 → float in [-1, 1]
                values = ((values / 127.5) ** 2) * np.sign(values)
                return values

    except Exception as exc:
        print(f"    Warning: could not read {gcs_path} — {exc}")
        return None


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_basin_embeddings(
    basins_gdf: gpd.GeoDataFrame,
    index: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each basin, sample the AlphaEarth embedding at the polygon centroid
    across all years in YEARS and return the year-averaged embedding.
    """
    records = []

    for _, row in basins_gdf.iterrows():
        gauge_id = str(row["gauge_id"])
        centroid  = row.geometry.centroid
        lon, lat  = centroid.x, centroid.y

        print(f"  Basin {gauge_id}: centroid ({lat:.4f}°N, {lon:.4f}°E)")

        year_embeddings = []
        for year in YEARS:
            tile_path = find_tile(index, lon, lat, year)
            if tile_path is None:
                print(f"    {year}: no tile found — skipping")
                continue

            emb = sample_tile(tile_path, lon, lat)
            if emb is None:
                print(f"    {year}: masked/out-of-bounds pixel — skipping")
                continue

            year_embeddings.append(emb)

        if not year_embeddings:
            print(f"    WARNING: no valid embeddings found — filling with NaN")
            mean_emb = np.full(N_BANDS, np.nan, dtype=np.float32)
        else:
            mean_emb = np.mean(year_embeddings, axis=0).astype(np.float32)
            print(f"    Averaged over {len(year_embeddings)} years | "
                  f"emb[:3] = {mean_emb[:3].round(4)}")

        # Normalize: "120.0" → "120" but "259.2" stays "259.2"
        try:
            f = float(gauge_id)
            gauge_id = str(int(f)) if f == int(f) else gauge_id
        except ValueError:
            pass

        record = {"gauge_id": gauge_id}
        record.update({f"emb_{i}": mean_emb[i] for i in range(N_BANDS)})
        records.append(record)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.environ["PROJ_NETWORK"] = "OFF"   # prevent pyproj hanging on CRS network lookups
    print("=== AlphaEarth Embedding Extraction (GCS COG route) ===\n")

    # 1. Download tile index
    print("Step 1: Load tile index")
    index = download_index()
    nepal_idx = index[
        (index["wgs84_east"]  > 78) & (index["wgs84_west"]  < 88) &
        (index["wgs84_north"] > 26) & (index["wgs84_south"] < 31)
    ]
    print(f"  Nepal-relevant tiles: {len(nepal_idx)} "
          f"({nepal_idx['year'].nunique()} years, UTM zones: {sorted(nepal_idx['utm_zone'].unique())})")

    # 2. Load basin polygons
    print("\nStep 2: Load basin polygons")
    basins_gdf = gpd.read_file(SHAPEFILE_PATH).to_crs("EPSG:4326")
    print(f"  {len(basins_gdf)} basins")

    # 3. Extract embeddings
    print("\nStep 3: Sample embeddings per basin (2017–2024 mean)")
    df = extract_basin_embeddings(basins_gdf, nepal_idx)

    # 4. Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    nan_basins = df[emb_cols].isna().any(axis=1).sum()
    print(f"\nSaved {len(df)} basins × {len(emb_cols)} dims → {OUT_PATH}")
    if nan_basins:
        print(f"WARNING: {nan_basins} basin(s) have NaN embeddings — check logs above.")
    print("\nSample (gauge_id + first 5 dims):")
    print(df[["gauge_id"] + emb_cols[:5]].to_string(index=False))


if __name__ == "__main__":
    main()
