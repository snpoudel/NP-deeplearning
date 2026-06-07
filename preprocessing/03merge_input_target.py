# This script merges static features and time series features to make them ready to ingest to the LSTM model
# For dynamic features use: temperature_2m_mean, total_precipitation_sum
# For static features: Use all 14 features from attributes_caravan_nepal.csv then further add features such as mean_elevation, drainage_area

import pandas as pd
import matplotlib.pyplot as plt


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
# read qobs and make a plot to visualize the availability of qobs data for each gauge_id and date. This will help us understand the time period for which we have qobs data and how it overlaps with the dynamic features we have. We can use this information to decide how to handle missing qobs data when merging with the dynamic features.
qobs_all = pd.read_csv("rawdata/selected_qobs.csv")
qobs_all["date"] = pd.to_datetime(qobs_all["date"])

qobs = qobs_all.melt(
    id_vars="date",
    var_name="gauge_id",
    value_name="qobs"
).dropna(subset=["qobs"])

qobs["gauge_id"] = (
    qobs["gauge_id"]
    .str.replace("nepal_", "", regex=False)
)

sites = sorted(qobs["gauge_id"].unique(), key=float)
site_map = {s: i for i, s in enumerate(sites)}

plt.figure(figsize=(8, 6))
plt.plot(
    qobs["date"],
    qobs["gauge_id"].map(site_map),
    "|",
    markersize=4
)

plt.yticks(
    range(len(sites)),
    [f"site_{s}" for s in sites]
)

plt.xlim(qobs_all["date"].min(), qobs_all["date"].max())
plt.grid(True, alpha=0.3)
plt.xlabel("Date")
plt.ylabel("Gauge Site")
plt.title("Observed Streamflow Availability")
plt.tight_layout()
plt.savefig("figures/qobs_availability.png", dpi=300)
plt.show()



##########################################
# save merged csv files for each gauge_id in merged_input folder. The merged csv file should have the following columns: date, temperature_2m_mean, total_precipitation_sum, qobs, and all static features from caravan_static. The qobs column should be converted from m³/s to mm/day using the formula: qobs_mm_day = (qobs_m3_s * 86400) / (drainage_area_km2 * 1000). The merged csv file should be named as nepal_{gauge_id}_merged.csv
# --- Prep static features once ---
static = caravan_static.copy()
static["drainage_area_km2"] = pd.to_numeric(static["drainage_area_km2"], errors="coerce")
static = static.set_index("gauge_id")

# --- Prep qobs once ---
qobs = qobs.copy()
qobs["date"] = pd.to_datetime(qobs["date"])
qobs["qobs"] = pd.to_numeric(qobs["qobs"], errors="coerce")

qobs_groups = {
    gid: g[["date", "qobs"]]
    for gid, g in qobs.groupby("gauge_id")
}

# --- Loop over basins ---
for gauge_id in static.index:

    dynamic = pd.read_csv(
        f"timeseries/csv/nepal/nepal_{gauge_id}.csv",
        usecols=["date", "temperature_2m_mean", "total_precipitation_sum"],
        parse_dates=["date"]
    )

    discharge = qobs_groups.get(gauge_id, pd.DataFrame(columns=["date", "qobs"]))

    merged = dynamic.merge(discharge, on="date", how="left")

    # add static features
    merged = merged.assign(**static.loc[gauge_id].to_dict())

    # unit conversion: m³/s -> mm/day
    merged["qobs"] = (
        merged["qobs"] * 86400
        / (merged["drainage_area_km2"] * 1000)
    )

    # round numeric columns except date
    merged.iloc[:, 1:] = merged.iloc[:, 1:].round(3)

    merged.to_csv(f"merged_input/nepal_{gauge_id}_merged.csv", index=False)



# convert csv files to parquet files for faster loading
import os
import pandas as pd
import pyarrow.parquet as pq
for filename in os.listdir("merged_input/csv"):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join("merged_input/csv", filename))
        df.to_parquet(os.path.join("merged_input", filename.replace(".csv", ".parquet")), index=False)
