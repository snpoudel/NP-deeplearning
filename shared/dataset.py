"""Dataset utilities for NP-deeplearning.

Provides feature definitions, input standardization (fit on train only, applied
to all splits), and a PyTorch Dataset class that returns sliding-window sequences
for any sequence model (LSTM, Transformer, post-processors).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, Dataset

# ---------------------------------------------------------------------------
# Feature definitions — imported by all run scripts
# ---------------------------------------------------------------------------

DYNAMIC_FEATURES = [
    "temperature_2m_mean",
    "total_precipitation_sum",
]

STATIC_FEATURES = [
    "aridity_ERA5_LAND",
    "aridity_FAO_PM",
    "frac_snow",
    "high_prec_dur",
    "high_prec_freq",
    "low_prec_dur",
    "low_prec_freq",
    "moisture_index_ERA5_LAND",
    "moisture_index_FAO_PM",
    "p_mean",
    "pet_mean_ERA5_LAND",
    "pet_mean_FAO_PM",
    "seasonality_ERA5_LAND",
    "seasonality_FAO_PM",
    "elevation_m",
    "drainage_area_km2",
]

TARGET = "qobs"

# AlphaEarth Foundations embedding columns (64-dim, from input/alphaearth_embeddings.parquet)
AE_FEATURES = [f"emb_{i}" for i in range(64)]
EMBEDDINGS_PATH = Path("input/alphaearth_embeddings.parquet")

# ALL_FEATURES uses AlphaEarth embeddings instead of hand-crafted static attributes —
# chosen by the input ablation experiment (experiment_ae/), which showed AE beats
# both dynamic-only and dynamic+static on NSE, KGE, RMSE, and PBIAS.
ALL_FEATURES = DYNAMIC_FEATURES + AE_FEATURES  # 66 input features total


# ---------------------------------------------------------------------------
# Scaler helpers
# ---------------------------------------------------------------------------

def fit_and_save_scaler(
    train_df: pd.DataFrame,
    scaler_path: str | Path,
    feature_cols: list[str] = ALL_FEATURES,
) -> StandardScaler:
    """Fit a StandardScaler on feature_cols of the training split and save it.

    Only the specified features are standardized; the target (qobs) is not.

    Args:
        train_df: Training-split DataFrame (must contain all columns in feature_cols).
        scaler_path: Path where the fitted scaler will be saved (e.g. output/model/scaler.pkl).
        feature_cols: Feature columns to fit on. Defaults to ALL_FEATURES (66 cols).

    Returns:
        The fitted StandardScaler instance.
    """
    scaler_path = Path(scaler_path)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)
    joblib.dump(scaler, scaler_path)
    return scaler


def load_scaler(scaler_path: str | Path) -> StandardScaler:
    """Load a previously saved StandardScaler from disk."""
    return joblib.load(scaler_path)


def attach_alphaearth(gauge_dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Join AlphaEarth embedding columns into each gauge's DataFrame.

    Reads input/alphaearth_embeddings.parquet (15 rows × 65 cols: gauge_id + emb_0…emb_63)
    and broadcasts the 64 embedding values as constant columns across all timesteps of
    each gauge, matching the same pattern used for static basin attributes.

    Run preprocessing/05_get_alphaearth_embeddings.py first to produce the parquet.
    """
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"{EMBEDDINGS_PATH} not found. "
            "Run preprocessing/05_get_alphaearth_embeddings.py first."
        )
    emb_df = pd.read_parquet(EMBEDDINGS_PATH)
    emb_df["gauge_id"] = emb_df["gauge_id"].astype(str)

    updated = {}
    for gauge_id, df in gauge_dfs.items():
        row = emb_df[emb_df["gauge_id"] == gauge_id]
        if row.empty:
            raise ValueError(
                f"gauge_id '{gauge_id}' not found in {EMBEDDINGS_PATH}. "
                "Re-run preprocessing/05_get_alphaearth_embeddings.py."
            )
        emb_vals = row[AE_FEATURES].iloc[0]
        df = df.copy()
        for col in AE_FEATURES:
            df[col] = emb_vals[col]
        updated[gauge_id] = df
    return updated


def apply_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler,
    feature_cols: list[str] = ALL_FEATURES,
) -> pd.DataFrame:
    """Return a copy of df with feature_cols standardized using a fitted scaler.

    The date and qobs columns are left untouched.

    Args:
        df: DataFrame containing feature_cols columns.
        scaler: A fitted StandardScaler (from fit_and_save_scaler or load_scaler).
        feature_cols: Feature columns to transform. Defaults to ALL_FEATURES (66 cols).

    Returns:
        New DataFrame with standardized feature columns.
    """
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols].values)
    return df


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StreamflowDataset(Dataset):
    """Sliding-window sequence dataset for streamflow prediction.

    Each sample is a window of seq_len consecutive time steps. The model input
    is the feature matrix for those steps (shape: seq_len × 18) and the target
    is the qobs value at the last step of the window.

    Static basin attributes are already constant per basin in the DataFrame, so
    they naturally repeat across all timesteps in the window — no extra tiling
    is needed.

    Args:
        df: Scaled DataFrame for one split (or one gauge). Must contain date,
            qobs, and all columns in ALL_FEATURES. qobs must be non-null.
        seq_len: Number of consecutive days per input sequence.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        feature_cols: list[str] = ALL_FEATURES,
    ) -> None:
        self.seq_len = seq_len
        self.X = df[feature_cols].to_numpy(dtype=np.float32)
        self.y = df[TARGET].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_seq = torch.tensor(self.X[idx : idx + self.seq_len])       # (seq_len, 18)
        target = torch.tensor([self.y[idx + self.seq_len - 1]])       # (1,)
        return x_seq, target


def build_concat_dataset(
    gauge_split_dfs: dict[str, pd.DataFrame],
    seq_len: int,
    feature_cols: list[str] = ALL_FEATURES,
) -> ConcatDataset:
    """One StreamflowDataset per gauge, concatenated to prevent cross-gauge sequences.

    Avoids ~7% of training samples that would otherwise mix two gauges' data
    at concatenation boundaries when using a single flat DataFrame.
    """
    datasets = [
        StreamflowDataset(df, seq_len, feature_cols)
        for df in gauge_split_dfs.values()
        if len(df) >= seq_len
    ]
    if not datasets:
        raise ValueError("No gauge has enough rows to form a sequence dataset.")
    return ConcatDataset(datasets)
