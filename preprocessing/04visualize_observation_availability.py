
import pandas as pd
import matplotlib.pyplot as plt
import os

# set path to home directory
home_dir = "C:/Sandeep/NP-deeplearning/"
os.chdir(home_dir)
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


### - Plot qobs availability
plt.figure(figsize=(8, 6))

# Highlight periods
plt.axvspan(
    pd.Timestamp("1980-01-01"),
    pd.Timestamp("1990-01-01"),
    color="orange",
    alpha=0.15,
    label="Validation (1980–1989)"
)

plt.axvspan(
    pd.Timestamp("1990-01-01"),
    pd.Timestamp("2005-01-01"),
    color="green",
    alpha=0.15,
    label="Training (1990–2004)"
)

plt.axvspan(
    pd.Timestamp("2005-01-01"),
    pd.Timestamp("2015-01-01"),
    color="blue",
    alpha=0.15,
    label="Testing (2005–2014)"
)

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

plt.xlim(pd.Timestamp("1980-01-01"), pd.Timestamp("2015-01-01"))

# Ensure first and last years are shown
years = [1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015]
plt.xticks(
    [pd.Timestamp(f"{y}-01-01") for y in years],
    years
)

plt.grid(True, alpha=0.3)
plt.xlabel("Date")
plt.ylabel("Gauge Site")
plt.title("Observed Streamflow Availability")

plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.10),
    ncol=3,
    frameon=False
)

plt.tight_layout()
plt.savefig("output/figures/qobs_availability.png", dpi=300, bbox_inches="tight")
plt.show()