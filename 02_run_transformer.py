# Train and evaluate a Transformer (sequence-to-one) for streamflow prediction.
# * Use hyperparameters from `shared/hyperparameters.py`
# * Use model from `shared/models.py`
# * Use dataset/preprocessing from `shared/dataset.py`
# * DataLoader input: `(batch, seq_len, features)`, target: `(batch, 1)`
# * Reuses the input scaler fitted during the LSTM run (output/model/scaler.pkl)

# Train with epoch loop:
# * forward pass → loss → backward pass → optimizer step
# * compute and print train/val loss each epoch
# * save best model by validation loss
# * store loss curves

# After training:
# * run inference with best model on complete dataset sequentially (all train/val/test sets)
# * save per-gauge parquet files to `output/predictions/transformer/`
# Output columns: `date, qobs, qsim`

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
from shared.models import build_transformer_model, kge, nse, rmse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INPUT_DIR = Path("input")
MODEL_DIR = Path("output/model")
PRED_DIR = Path("output/predictions/transformer")
SCALER_PATH = MODEL_DIR / "scaler.pkl"
MODEL_PATH = MODEL_DIR / "transformer_best.pt"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(input_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all parquet files from input_dir and parse date column.

    Returns:
        dict mapping gauge_id (str) to its DataFrame.
    """
    gauge_dfs = {}
    for path in sorted(input_dir.glob("*.parquet")):
        gauge_id = re.sub(r"^nepal_|_merged\.parquet$", "", path.name)
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        gauge_dfs[gauge_id] = df
    return gauge_dfs


# ---------------------------------------------------------------------------
# Training / evaluation helpers
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
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model in eval mode and collect all predictions and targets.

    Returns:
        (qsim, qobs) as 1-D numpy arrays.
    """
    model.eval()
    all_qsim, all_qobs = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).cpu().numpy().flatten()
            all_qsim.append(pred)
            all_qobs.append(y.numpy().flatten())
    return np.concatenate(all_qsim), np.concatenate(all_qobs)


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

    hp = HYPERPARAMS["transformer"]
    seq_len = hp["seq_len"]
    batch_size = hp["batch_size"]
    num_epochs = hp["num_epochs"]
    patience = hp["early_stopping_patience"]
    lr = hp["learning_rate"]

    PRED_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    gauge_dfs = load_data(INPUT_DIR)
    print(f"  Loaded {len(gauge_dfs)} gauges: {sorted(gauge_dfs)}")

    # Concatenate all gauges for global multi-basin training
    all_dfs = []
    for gauge_id, df in gauge_dfs.items():
        df = df.copy()
        df["gauge_id"] = gauge_id
        all_dfs.append(df)
    all_df = pd.concat(all_dfs, ignore_index=True)

    # Filter to training period (1980–2014) and drop rows with missing qobs
    all_df = all_df[
        (all_df["date"] >= SPLIT_DATES["train"][0])
        & (all_df["date"] <= SPLIT_DATES["test"][1])
    ]
    n_before = len(all_df)
    all_df = all_df.dropna(subset=[TARGET]).reset_index(drop=True)
    n_dropped = n_before - len(all_df)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} rows with missing qobs")

    assert len(all_df) > 0, "No valid qobs rows found after filtering — check input data."

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
    # 3. Load scaler fitted during the LSTM run
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
    model = build_transformer_model(input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"\nModel: {model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    print("Training...")
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
            print(" *")  # mark best epoch
        else:
            patience_counter += 1
            print()
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    print(f"\nBest val loss: {best_val_loss:.4f} — model saved to {MODEL_PATH}")

    # Save loss curves as parquet for later plotting
    loss_df = pd.DataFrame({"epoch": range(1, len(train_losses) + 1),
                            "train_loss": train_losses, "val_loss": val_losses})
    loss_path = MODEL_DIR / "transformer_loss_curves.parquet"
    loss_df.to_parquet(loss_path, index=False)
    print(f"Loss curves saved to {loss_path}")

    # ------------------------------------------------------------------
    # 7. Reload best model and report validation metrics
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    qsim_val, qobs_val = run_inference(model, val_loader, device)
    metrics = compute_metrics(qobs_val, qsim_val)
    print(f"\nValidation metrics (best model):")
    for k, v in metrics.items():
        print(f"  {k.upper()}: {v:.4f}")

    # ------------------------------------------------------------------
    # 8. Per-gauge inference on full 1980–2014 range
    # ------------------------------------------------------------------
    print(f"\nRunning per-gauge inference...")
    for gauge_id, raw_df in gauge_dfs.items():
        # Filter to full model period and drop missing qobs
        g_df = raw_df[
            (raw_df["date"] >= SPLIT_DATES["train"][0])
            & (raw_df["date"] <= SPLIT_DATES["test"][1])
        ].dropna(subset=[TARGET]).reset_index(drop=True)

        if len(g_df) < seq_len:
            print(f"  {gauge_id}: skipped (only {len(g_df)} rows, need ≥ {seq_len})")
            continue

        g_scaled = apply_scaler(g_df, scaler)
        g_ds = StreamflowDataset(g_scaled, seq_len)
        g_loader = DataLoader(g_ds, batch_size=batch_size, shuffle=False)

        qsim_g, qobs_g = run_inference(model, g_loader, device)

        # Align dates: first prediction corresponds to row index (seq_len - 1)
        dates = g_df["date"].iloc[seq_len - 1:].reset_index(drop=True)

        out_df = pd.DataFrame({"date": dates, "qobs": qobs_g, "qsim": qsim_g})
        out_path = PRED_DIR / f"nepal_{gauge_id}_transformer.parquet"
        out_df.to_parquet(out_path, index=False)
        print(f"  {gauge_id}: {len(out_df)} rows -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
