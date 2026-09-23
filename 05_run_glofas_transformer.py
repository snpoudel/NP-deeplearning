# Transformer post-processor for correcting GloFAS streamflow outputs.
#
# Identical to 03_run_glofas_lstm.py but uses the Transformer model.
# Approach (residual correction):
#   1. Compute residual = qobs - qglofas for every timestep
#   2. Train the Transformer to predict this residual from ERA5 + static inputs
#   3. Final prediction: qsim = qglofas + predicted_residual
#
# GloFAS data: input/physical_model/glofas_selected_qobs.parquet
#   Wide format: columns are date, 120, 259.2, 260, … (one per gauge, mm/day)
#
# Output columns: date, qobs, qglofas, qsim
# Output path:    output/predictions/glofas_transformer/seed{seed}/nepal_{gauge_id}_glofas_transformer.parquet

import re
import time
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
    attach_alphaearth,
    build_concat_dataset,
    load_scaler,
)
from shared.hyperparameters import get_hyperparams, SPLIT_DATES
from shared.models import build_transformer_model, kge, nse, rmse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INPUT_DIR = Path("input")
GLOFAS_PATH = Path("input/physical_model/glofas_selected_qobs.parquet")
MODEL_DIR = Path("output/model")
SCALER_PATH = MODEL_DIR / "scaler.pkl"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(input_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all gauge parquet files from input_dir and parse the date column."""
    gauge_dfs = {}
    for path in sorted(input_dir.glob("nepal_*_merged.parquet")):
        gauge_id = re.sub(r"^nepal_|_merged\.parquet$", "", path.name)
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        gauge_dfs[gauge_id] = df
    return gauge_dfs


def load_physical_model(path: Path, col_name: str) -> pd.DataFrame:
    """Load a physical model parquet and reshape from wide to long format."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.melt(id_vars="date", var_name="gauge_id", value_name=col_name)
    return df


def merge_physical_model(
    gauge_dfs: dict[str, pd.DataFrame], phys_long: pd.DataFrame, col_name: str
) -> dict[str, pd.DataFrame]:
    """Merge a physical model column into each gauge DataFrame on date."""
    merged = {}
    for gauge_id, df in gauge_dfs.items():
        g_phys = phys_long[phys_long["gauge_id"] == gauge_id][["date", col_name]]
        merged[gauge_id] = df.merge(g_phys, on="date", how="left")
    return merged


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
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
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
    return {"nse": nse(qobs_arr, qsim_arr), "kge": kge(qobs_arr, qsim_arr), "rmse": rmse(qobs_arr, qsim_arr)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed: int, device: str, mode: str = "dev") -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    _device = torch.device(device)
    print(f"Using device: {_device} | seed: {seed}")

    hp = get_hyperparams(mode)["transformer"]
    seq_len = hp["seq_len"]
    batch_size = hp["batch_size"]
    num_epochs = hp["num_epochs"]
    patience = hp["early_stopping_patience"]
    lr = hp["learning_rate"]

    # Seed-specific output paths
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"glofas_transformer_seed{seed}_best.pt"
    pred_dir = Path("output/predictions/glofas_transformer") / f"seed{seed}"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data and merge GloFAS
    # ------------------------------------------------------------------
    print("Loading data...")
    gauge_dfs = load_data(INPUT_DIR)
    glofas_long = load_physical_model(GLOFAS_PATH, "qglofas")
    gauge_dfs = merge_physical_model(gauge_dfs, glofas_long, "qglofas")
    gauge_dfs = attach_alphaearth(gauge_dfs)
    print(f"  Loaded {len(gauge_dfs)} gauges with GloFAS merged")

    all_dfs = []
    for gauge_id, df in gauge_dfs.items():
        df = df.copy()
        df["gauge_id"] = gauge_id
        all_dfs.append(df)
    all_df = pd.concat(all_dfs, ignore_index=True)

    all_df = all_df[
        (all_df["date"] >= SPLIT_DATES["val"][0])
        & (all_df["date"] <= SPLIT_DATES["test"][1])
    ]
    n_before = len(all_df)
    all_df = all_df.dropna(subset=[TARGET, "qglofas"]).reset_index(drop=True)
    n_dropped = n_before - len(all_df)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} rows with missing qobs or qglofas")

    assert len(all_df) > 0, "No valid rows after filtering — check input data."

    all_df["residual"] = all_df[TARGET] - all_df["qglofas"]
    all_df[TARGET] = all_df["residual"]

    # ------------------------------------------------------------------
    # 2. Split by date
    # ------------------------------------------------------------------
    train_df = all_df[(all_df["date"] >= SPLIT_DATES["train"][0]) & (all_df["date"] <= SPLIT_DATES["train"][1])].reset_index(drop=True)
    val_df = all_df[
        (all_df["date"] >= SPLIT_DATES["val"][0])
        & (all_df["date"] <= SPLIT_DATES["val"][1])
    ].reset_index(drop=True)

    print(f"  Split sizes — train: {len(train_df)}, val: {len(val_df)}")

    # ------------------------------------------------------------------
    # 3. Load scaler fitted by the LSTM run
    # ------------------------------------------------------------------
    assert SCALER_PATH.exists(), f"Scaler not found at {SCALER_PATH}. Run 01_run_lstm.py first."
    print(f"Loading scaler from {SCALER_PATH}...")
    scaler = load_scaler(SCALER_PATH)

    # ------------------------------------------------------------------
    # 4. Build per-gauge DataLoaders with residual target per gauge
    # ------------------------------------------------------------------
    train_per_gauge, val_per_gauge = {}, {}
    for gauge_id, gdf in gauge_dfs.items():
        gdf_clean = gdf[
            (gdf["date"] >= SPLIT_DATES["val"][0]) & (gdf["date"] <= SPLIT_DATES["test"][1])
        ].dropna(subset=[TARGET, "qglofas"]).copy().reset_index(drop=True)
        gdf_clean[TARGET] = gdf_clean[TARGET] - gdf_clean["qglofas"]  # residual as target
        gdf_scaled = apply_scaler(gdf_clean, scaler)
        train_per_gauge[gauge_id] = gdf_scaled[
            (gdf_scaled["date"] >= SPLIT_DATES["train"][0]) & (gdf_scaled["date"] <= SPLIT_DATES["train"][1])
        ].reset_index(drop=True)
        val_per_gauge[gauge_id] = gdf_scaled[
            (gdf_scaled["date"] >= SPLIT_DATES["val"][0]) & (gdf_scaled["date"] <= SPLIT_DATES["val"][1])
        ].reset_index(drop=True)

    train_ds = build_concat_dataset(train_per_gauge, seq_len)
    val_ds = build_concat_dataset(val_per_gauge, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # ------------------------------------------------------------------
    # 5. Build model, optimizer, loss
    # ------------------------------------------------------------------
    input_size = len(ALL_FEATURES)
    model = build_transformer_model(input_size, mode=mode).to(_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=hp["lr_scheduler_patience"], factor=hp["lr_scheduler_factor"]
    )
    criterion = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: TransformerModel | parameters: {total_params:,}\n")

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    print("Training on residuals (qobs - qglofas)...")
    train_start_time = time.time()
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, _device)
        val_loss = eval_epoch(model, val_loader, criterion, _device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={current_lr:.2e}", end="")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(" *")
        else:
            patience_counter += 1
            print()
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    train_time = time.time() - train_start_time
    print(f"\nBest val loss: {best_val_loss:.4f} — model saved to {model_path}")
    print(f"Training time: {train_time:.1f}s")

    # Save per-seed loss curves and mirror to flat path for standalone compatibility
    loss_df = pd.DataFrame({"epoch": range(1, len(train_losses) + 1),
                            "train_loss": train_losses, "val_loss": val_losses})
    loss_path = MODEL_DIR / f"glofas_transformer_seed{seed}_loss_curves.parquet"
    loss_df.to_parquet(loss_path, index=False)
    loss_df.to_parquet(MODEL_DIR / "glofas_transformer_loss_curves.parquet", index=False)
    print(f"Loss curves saved to {loss_path}")

    # Log training time to shared CSV (append mode)
    timing_path = Path("output/training_times.csv")
    timing_row = pd.DataFrame([{
        "model": "glofas_transformer", "seed": seed, "epochs_run": len(train_losses),
        "best_val_loss": round(best_val_loss, 6), "train_time_seconds": round(train_time, 2),
    }])
    timing_row.to_csv(timing_path, mode="a", index=False, header=not timing_path.exists())

    # ------------------------------------------------------------------
    # 7. Reload best model and report validation metrics on final qsim
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(model_path, map_location=_device))

    all_qobs_v, all_qsim_v = [], []
    for gauge_id in sorted(val_per_gauge.keys()):
        gdf = val_per_gauge[gauge_id]
        if len(gdf) < seq_len:
            continue
        g_ds = StreamflowDataset(gdf, seq_len)
        g_loader = DataLoader(g_ds, batch_size=batch_size, shuffle=False)
        pred_residuals = run_inference(model, g_loader, _device)
        phys_vals = gdf["qglofas"].iloc[seq_len - 1:].values
        orig_qobs = gdf[TARGET].iloc[seq_len - 1:].values + phys_vals
        all_qobs_v.extend(orig_qobs)
        all_qsim_v.extend(phys_vals + pred_residuals)

    metrics = compute_metrics(np.array(all_qobs_v), np.array(all_qsim_v))
    print(f"\nValidation metrics — final qsim (qglofas + predicted residual):")
    for k, v in metrics.items():
        print(f"  {k.upper()}: {v:.4f}")

    # ------------------------------------------------------------------
    # 8. Per-gauge inference on full 1980–2014 range
    # ------------------------------------------------------------------
    print(f"\nRunning per-gauge inference...")
    for gauge_id, raw_df in gauge_dfs.items():
        g_df = raw_df[
            (raw_df["date"] >= SPLIT_DATES["val"][0])
            & (raw_df["date"] <= SPLIT_DATES["test"][1])
        ].dropna(subset=[TARGET, "qglofas"]).reset_index(drop=True)

        if len(g_df) < seq_len:
            print(f"  {gauge_id}: skipped (only {len(g_df)} rows, need ≥ {seq_len})")
            continue

        g_scaled = apply_scaler(g_df, scaler)
        g_ds = StreamflowDataset(g_scaled, seq_len)
        g_loader = DataLoader(g_ds, batch_size=batch_size, shuffle=False)

        predicted_residuals = run_inference(model, g_loader, _device)

        start = seq_len - 1
        dates = g_df["date"].iloc[start:].reset_index(drop=True)
        qobs_out = g_df[TARGET].iloc[start:].reset_index(drop=True)  # original qobs
        qglofas_out = g_df["qglofas"].iloc[start:].reset_index(drop=True)
        qsim_out = qglofas_out.values + predicted_residuals

        out_df = pd.DataFrame({
            "date": dates,
            "qobs": qobs_out,
            "qglofas": qglofas_out,
            "qsim": qsim_out,
        })
        out_path = pred_dir / f"nepal_{gauge_id}_glofas_transformer.parquet"
        out_df.to_parquet(out_path, index=False)
        out_df.to_parquet(pred_dir.parent / f"nepal_{gauge_id}_glofas_transformer_mean.parquet", index=False)
        print(f"  {gauge_id}: {len(out_df)} rows -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    import argparse
    from shared.hyperparameters import DEVICE, SEEDS

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["dev", "production"], default="dev",
        help="Hyperparameter profile: 'dev' (default, fast smoke-test) or 'production' (full run)",
    )
    _args = parser.parse_args()

    _seed = SEEDS[0]
    _device_str = "cuda" if (DEVICE == "auto" and torch.cuda.is_available()) else (
        DEVICE if DEVICE != "auto" else "cpu"
    )
    main(seed=_seed, device=_device_str, mode=_args.mode)
