# This script merges static features and time series features to make them ready to ingest to the LSTM model
# For dynamic features use: temperature_2m_mean, total_precipitation_sum
# For static features: Use all 14 features from attributes_caravan_nepal.csv then further add features such as mean_elevation, drainage_area

import pandas as pd
import matplotlib.pyplot as plt

# set working directory to the root of the project
import os
os.chdir("..")

# read list of basins file
basins_list = pd.read_csv('rawdata/selected_hydro_stations.csv')
basins_list['gauge_id'] = basins_list['station']
# read caravan static features
caravan_static = pd.read_csv('attributes/nepal/attributes_caravan_nepal.csv')
# in gauge id column, remove the prefix "nepal_" and convert to numeric
caravan_static['gauge_id'] = caravan_static['gauge_id'].str.replace('nepal_', '').astype(float)

# merge elevation_m and drainage_area_km2 from basins_list to caravan_static on gauge_id
caravan_static = caravan_static.merge(basins_list[['gauge_id', 'elevation_m', 'drainage_area_km2']], on='gauge_id', how='left')



##########################################
# save merged csv files for each gauge_id in merged_input folder. The merged csv file should have the following columns: date, temperature_2m_mean, total_precipitation_sum, qobs, and all static features from caravan_static. The qobs column should be converted from m³/s to mm/day using the formula: qobs_mm_day = (qobs_m3_s * 86400) / (drainage_area_km2 * 1000). The merged csv file should be named as nepal_{gauge_id}_merged.csv
# --- Prep static features once ---
static = caravan_static.copy()
static["drainage_area_km2"] = pd.to_numeric(static["drainage_area_km2"], errors="coerce")
static = static.set_index("gauge_id")

# Helper functions
# ---------------------------------------------------
def clean_id(x):
    x = float(x)
    return str(int(x)) if x.is_integer() else str(x)

# STATIC (keep RAW for file names + CLEAN for matching)
# ---------------------------------------------------
static_raw_index = static.index.astype(str).str.strip()

static_clean_index = static_raw_index.map(clean_id)
static.index = static_clean_index

# also keep mapping back to raw filenames
static["raw_id"] = static_raw_index.values

# QOBS
# ---------------------------------------------------
qobs_all = pd.read_csv("rawdata/selected_qobs.csv")
qobs_all["date"] = pd.to_datetime(qobs_all["date"])

qobs = qobs_all.melt(
    id_vars="date",
    var_name="gauge_id",
    value_name="qobs"
)

qobs["gauge_id"] = qobs["gauge_id"].astype(str).str.strip().map(clean_id)
qobs["qobs"] = pd.to_numeric(qobs["qobs"], errors="coerce")
qobs = qobs.dropna(subset=["qobs"])

qobs_groups = {
    gid: g[["date", "qobs"]]
    for gid, g in qobs.groupby("gauge_id")
}

print("Basins with QOBS:", len(qobs_groups))

# LOOP
# ---------------------------------------------------
for gauge_id in static.index:

    # use RAW id for file path
    raw_id = static.loc[gauge_id, "raw_id"]

    dynamic = pd.read_csv(
        f"timeseries/csv/nepal/nepal_{raw_id}.csv",
        usecols=["date", "temperature_2m_mean", "total_precipitation_sum"],
        parse_dates=["date"]
    )

    discharge = qobs_groups.get(
        gauge_id,
        pd.DataFrame(columns=["date", "qobs"])
    )

    merged = dynamic.merge(discharge, on="date", how="left")

    merged = merged.assign(**static.loc[gauge_id].drop("raw_id").to_dict())

    merged["qobs"] = (
        merged["qobs"] * 86400 /
        (merged["drainage_area_km2"] * 1000)
    )

    merged.iloc[:, 1:] = merged.iloc[:, 1:].round(3)

    merged.to_csv(
        f"input/csv/nepal_{gauge_id}_merged.csv",
        index=False
    )

    print(f"{gauge_id}: done")


# convert csv files to parquet files for faster loading
import os
import pandas as pd
import pyarrow.parquet as pq
for filename in os.listdir("input/csv"):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join("input/csv", filename))
        df.to_parquet(os.path.join("input", filename.replace(".csv", ".parquet")), index=False)
