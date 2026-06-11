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
from torch.utils.data import Dataset

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
ALL_FEATURES = DYNAMIC_FEATURES + STATIC_FEATURES  # 18 input features total


# ---------------------------------------------------------------------------
# Scaler helpers
# ---------------------------------------------------------------------------

def fit_and_save_scaler(train_df: pd.DataFrame, scaler_path: str | Path) -> StandardScaler:
    """Fit a StandardScaler on ALL_FEATURES of the training split and save it.

    Only dynamic and static features are standardized; the target (qobs) is not.

    Args:
        train_df: Training-split DataFrame (must contain all columns in ALL_FEATURES).
        scaler_path: Path where the fitted scaler will be saved (e.g. output/model/scaler.pkl).

    Returns:
        The fitted StandardScaler instance.
    """
    scaler_path = Path(scaler_path)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    scaler.fit(train_df[ALL_FEATURES].values)
    joblib.dump(scaler, scaler_path)
    return scaler


def load_scaler(scaler_path: str | Path) -> StandardScaler:
    """Load a previously saved StandardScaler from disk."""
    return joblib.load(scaler_path)


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Return a copy of df with ALL_FEATURES standardized using a fitted scaler.

    The date and qobs columns are left untouched.

    Args:
        df: DataFrame containing ALL_FEATURES columns.
        scaler: A fitted StandardScaler (from fit_and_save_scaler or load_scaler).

    Returns:
        New DataFrame with standardized feature columns.
    """
    df = df.copy()
    df[ALL_FEATURES] = scaler.transform(df[ALL_FEATURES].values)
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

    def __init__(self, df: pd.DataFrame, seq_len: int) -> None:
        self.seq_len = seq_len
        self.X = df[ALL_FEATURES].to_numpy(dtype=np.float32)  # (N, 18)
        self.y = df[TARGET].to_numpy(dtype=np.float32)        # (N,)

    def __len__(self) -> int:
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_seq = torch.tensor(self.X[idx : idx + self.seq_len])       # (seq_len, 18)
        target = torch.tensor([self.y[idx + self.seq_len - 1]])       # (1,)
        return x_seq, target
