# LSTM post-processor for correcting GloFAS streamflow outputs.
#
# Approach (residual correction):
#   1. Compute residual = qobs - qglofas for every timestep
#   2. Train the LSTM to predict this residual from the same ERA5 + static inputs
#   3. Final prediction: qsim = qglofas + predicted_residual
#
# Everything else (features, scaler, model architecture, hyperparameters) is
# identical to 01_run_lstm.py. The scaler fitted during that run is reused here.
#
# GloFAS data: input/physical_model/glofas_selected_qobs.parquet
#   Wide format — columns: date, 120, 259.2, 260, … (one per gauge, mm/day)
#
# Output columns: date, qobs, qglofas, qsim
# Output path:    output/predictions/glofas_lstm/nepal_{gauge_id}_glofas_lstm.parquet

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shared.dataset import (
    ALL_FEATURES,
    TARGET,
    StreamflowDataset,
    apply_scaler,
    load_scaler,
)
from shared.hyperparameters import HYPERPARAMS, RANDOM_SEED, SPLIT_DATES
from shared.models import build_lstm_model, kge, nse, rmse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INPUT_DIR = Path("input")
GLOFAS_PATH = Path("input/physical_model/glofas_selected_qobs.parquet")
MODEL_DIR = Path("output/model")
PRED_DIR = Path("output/predictions/glofas_lstm")
SCALER_PATH = MODEL_DIR / "scaler.pkl"
MODEL_PATH = MODEL_DIR / "glofas_lstm_best.pt"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(input_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all gauge parquet files and parse date column."""
    gauge_dfs = {}
    for path in sorted(input_dir.glob("*.parquet")):
        gauge_id = re.sub(r"^nepal_|_merged\.parquet$", "", path.name)
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        gauge_dfs[gauge_id] = df
    return gauge_dfs


def load_glofas(glofas_path: Path) -> pd.DataFrame:
    """Load GloFAS parquet and reshape from wide to long format.

    Returns:
        DataFrame with columns: date, gauge_id, qglofas
    """
    df = pd.read_parquet(glofas_path)
    df["date"] = pd.to_datetime(df["date"])
    # Melt wide → long: each row becomes (date, gauge_id, qglofas)
    df = df.melt(id_vars="date", var_name="gauge_id", value_name="qglofas")
    return df


def merge_glofas(gauge_dfs: dict[str, pd.DataFrame], glofas_long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Merge qglofas into each gauge DataFrame on date.

    Returns:
        dict mapping gauge_id to DataFrame with added qglofas column.
    """
    merged = {}
    for gauge_id, df in gauge_dfs.items():
        g_glofas = glofas_long[glofas_long["gauge_id"] == gauge_id][["date", "qglofas"]]
        merged[gauge_id] = df.merge(g_glofas, on="date", how="left")
    return merged


# ---------------------------------------------------------------------------
# Training / evaluation helpers (identical to 01_run_lstm.py)
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """One full training pass. Returns mean loss over the epoch."""
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """One forward-only pass. Returns mean loss over the loader."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += criterion(pred, y).item() * len(x)
    return total_loss / len(loader.dataset)


def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Run the model in eval mode and return all predictions as a 1-D array."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for x, _ in loader:
            pred = model(x.to(device)).cpu().numpy().flatten()
            all_preds.append(pred)
    return np.concatenate(all_preds)


def compute_metrics(qobs_arr: np.ndarray, qsim_arr: np.ndarray) -> dict[str, float]:
    """Compute NSE, KGE, and RMSE between observed and simulated arrays."""
    return {
        "nse": nse(qobs_arr, qsim_arr),
        "kge": kge(qobs_arr, qsim_arr),
        "rmse": rmse(qobs_arr, qsim_arr),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hp = HYPERPARAMS["lstm"]
    seq_len = hp["seq_len"]
    batch_size = hp["batch_size"]
    num_epochs = hp["num_epochs"]
    patience = hp["early_stopping_patience"]
    lr = hp["learning_rate"]

    PRED_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data and merge GloFAS
    # ------------------------------------------------------------------
    print("Loading data...")
    gauge_dfs = load_data(INPUT_DIR)
    glofas_long = load_glofas(GLOFAS_PATH)
    gauge_dfs = merge_glofas(gauge_dfs, glofas_long)
    print(f"  Loaded {len(gauge_dfs)} gauges with GloFAS merged")

    # Concatenate all gauges
    all_dfs = []
    for gauge_id, df in gauge_dfs.items():
        df = df.copy()
        df["gauge_id"] = gauge_id
        all_dfs.append(df)
    all_df = pd.concat(all_dfs, ignore_index=True)

    # Filter to 1980–2014, require both qobs and qglofas non-null
    all_df = all_df[
        (all_df["date"] >= SPLIT_DATES["train"][0])
        & (all_df["date"] <= SPLIT_DATES["test"][1])
    ]
    n_before = len(all_df)
    all_df = all_df.dropna(subset=[TARGET, "qglofas"]).reset_index(drop=True)
    n_dropped = n_before - len(all_df)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} rows with missing qobs or qglofas")

    assert len(all_df) > 0, "No valid rows after filtering — check input data."

    # Compute residual and replace qobs with it for training target
    all_df["residual"] = all_df[TARGET] - all_df["qglofas"]
    all_df[TARGET] = all_df["residual"]  # dataset will predict residuals

    # ------------------------------------------------------------------
    # 2. Split by date
    # ------------------------------------------------------------------
    train_df = all_df[all_df["date"] <= SPLIT_DATES["train"][1]].reset_index(drop=True)
    val_df = all_df[
        (all_df["date"] >= SPLIT_DATES["val"][0])
        & (all_df["date"] <= SPLIT_DATES["val"][1])
    ].reset_index(drop=True)

    print(f"  Split sizes — train: {len(train_df)}, val: {len(val_df)}")

    # ------------------------------------------------------------------
    # 3. Load scaler from LSTM run
    # ------------------------------------------------------------------
    assert SCALER_PATH.exists(), (
        f"Scaler not found at {SCALER_PATH}. Run 01_run_lstm.py first."
    )
    print(f"Loading scaler from {SCALER_PATH}...")
    scaler = load_scaler(SCALER_PATH)

    train_scaled = apply_scaler(train_df, scaler)
    val_scaled = apply_scaler(val_df, scaler)

    # ------------------------------------------------------------------
    # 4. Build DataLoaders
    # ------------------------------------------------------------------
    train_ds = StreamflowDataset(train_scaled, seq_len)
    val_ds = StreamflowDataset(val_scaled, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # ------------------------------------------------------------------
    # 5. Build model, optimizer, loss
    # ------------------------------------------------------------------
    input_size = len(ALL_FEATURES)
    model = build_lstm_model(input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: LSTMModel | parameters: {total_params:,}\n")

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    print("Training on residuals (qobs - qglofas)...")
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"  Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}", end="")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(" *")
        else:
            patience_counter += 1
            print()
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    print(f"\nBest val loss: {best_val_loss:.4f} — model saved to {MODEL_PATH}")

    loss_df = pd.DataFrame({"epoch": range(1, len(train_losses) + 1),
                            "train_loss": train_losses, "val_loss": val_losses})
    loss_path = MODEL_DIR / "glofas_lstm_loss_curves.parquet"
    loss_df.to_parquet(loss_path, index=False)
    print(f"Loss curves saved to {loss_path}")

    # ------------------------------------------------------------------
    # 7. Reload best model and report validation metrics on final qsim
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # Compute final qsim = qglofas + predicted_residual on the val split
    val_df_orig = all_df[
        (all_df["date"] >= SPLIT_DATES["val"][0])
        & (all_df["date"] <= SPLIT_DATES["val"][1])
    ].reset_index(drop=True)

    predicted_residuals_val = run_inference(model, val_loader, device)
    qglofas_val = val_df_orig["qglofas"].iloc[seq_len - 1:].values
    qobs_val = val_df_orig["residual"].iloc[seq_len - 1:].values + qglofas_val  # recover original qobs
    qsim_val = qglofas_val + predicted_residuals_val

    metrics = compute_metrics(qobs_val, qsim_val)
    print(f"\nValidation metrics — final qsim (qglofas + predicted residual):")
    for k, v in metrics.items():
        print(f"  {k.upper()}: {v:.4f}")

    # ------------------------------------------------------------------
    # 8. Per-gauge inference on full 1980–2014 range
    # ------------------------------------------------------------------
    print(f"\nRunning per-gauge inference...")
    for gauge_id, raw_df in gauge_dfs.items():
        # Filter to model period, require both qobs and qglofas
        g_df = raw_df[
            (raw_df["date"] >= SPLIT_DATES["train"][0])
            & (raw_df["date"] <= SPLIT_DATES["test"][1])
        ].dropna(subset=[TARGET, "qglofas"]).reset_index(drop=True)

        # Restore original qobs (before we overwrote it with residual)
        # raw_df still has the original qobs since gauge_dfs was mutated earlier;
        # re-compute original qobs = residual + qglofas
        g_df["qobs_orig"] = g_df[TARGET] + g_df["qglofas"]
        g_df["residual"] = g_df[TARGET]  # TARGET column now holds residuals

        if len(g_df) < seq_len:
            print(f"  {gauge_id}: skipped (only {len(g_df)} rows, need ≥ {seq_len})")
            continue

        g_scaled = apply_scaler(g_df, scaler)
        g_ds = StreamflowDataset(g_scaled, seq_len)
        g_loader = DataLoader(g_ds, batch_size=batch_size, shuffle=False)

        predicted_residuals = run_inference(model, g_loader, device)

        # Align dates and values to the prediction window
        start = seq_len - 1
        dates = g_df["date"].iloc[start:].reset_index(drop=True)
        qobs_out = g_df["qobs_orig"].iloc[start:].reset_index(drop=True)
        qglofas_out = g_df["qglofas"].iloc[start:].reset_index(drop=True)
        qsim_out = qglofas_out.values + predicted_residuals

        out_df = pd.DataFrame({
            "date": dates,
            "qobs": qobs_out,
            "qglofas": qglofas_out,
            "qsim": qsim_out,
        })
        out_path = PRED_DIR / f"nepal_{gauge_id}_glofas_lstm.parquet"
        out_df.to_parquet(out_path, index=False)
        print(f"  {gauge_id}: {len(out_df)} rows -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
