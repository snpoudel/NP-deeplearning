"""
experiment_ae/run_ablation_transformer.py

Input feature ablation experiment for Transformer.

Trains three Transformer variants with seed=42 (single seed — sufficient for a
comparison experiment), mirroring experiment_ae/run_ablation.py's LSTM ablation:

  A) transformer_dynamic      — 2 features: temperature + precipitation
  B) transformer_static       — 18 features: dynamic + existing 16 static attrs
  C) transformer_alphaearth   — 2 + emb_dim features: dynamic + AlphaEarth embeddings

Everything else (architecture, hyperparameters, train/val/test split) is
identical to 02_run_transformer.py. Predictions are saved in the same format
as run_ablation.py so evaluate_ablation.py can compare all six variants.

Prerequisite: run preprocessing/05_get_alphaearth_embeddings.py first to
produce input/alphaearth_embeddings.parquet.

Usage:
    python experiment_ae/run_ablation_transformer.py                  # production
    python experiment_ae/run_ablation_transformer.py --mode dev       # fast smoke-test
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Make sure repo root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.dataset import (
    DYNAMIC_FEATURES,
    STATIC_FEATURES,
    TARGET,
    StreamflowDataset,
    apply_scaler,
    build_concat_dataset,
    fit_and_save_scaler,
)
from shared.hyperparameters import get_hyperparams, SPLIT_DATES
from shared.models import build_transformer_model

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INPUT_DIR      = Path("input")
MODEL_DIR      = Path("output/model")
EMBEDDINGS_PATH = INPUT_DIR / "alphaearth_embeddings.parquet"
SEED           = 42



# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gauge_dfs(input_dir: Path) -> dict[str, pd.DataFrame]:
    gauge_dfs = {}
    for path in sorted(input_dir.glob("nepal_*_merged.parquet")):
        gauge_id = re.sub(r"^nepal_|_merged\.parquet$", "", path.name)
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        gauge_dfs[gauge_id] = df
    return gauge_dfs


def attach_alphaearth(gauge_dfs: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Join AlphaEarth embedding columns into each gauge's DataFrame.

    Returns the updated gauge_dfs and the list of embedding column names.
    """
    emb_df = pd.read_parquet(EMBEDDINGS_PATH)
    emb_df["gauge_id"] = emb_df["gauge_id"].astype(str)
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]

    updated = {}
    for gauge_id, df in gauge_dfs.items():
        row = emb_df[emb_df["gauge_id"] == gauge_id]
        if row.empty:
            raise ValueError(
                f"gauge_id '{gauge_id}' not found in {EMBEDDINGS_PATH}. "
                "Re-run preprocessing/05_get_alphaearth_embeddings.py."
            )
        emb_vals = row[emb_cols].iloc[0]
        df = df.copy()
        for col in emb_cols:
            df[col] = emb_vals[col]
        updated[gauge_id] = df

    return updated, emb_cols


# ---------------------------------------------------------------------------
# Training / evaluation helpers (identical to 02_run_transformer.py)
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device) -> float:
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


def eval_epoch(model, loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total_loss += criterion(model(x), y).item() * len(x)
    return total_loss / len(loader.dataset)


def run_inference(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_qsim, all_qobs = [], []
    with torch.no_grad():
        for x, y in loader:
            all_qsim.append(model(x.to(device)).cpu().numpy().flatten())
            all_qobs.append(y.numpy().flatten())
    return np.concatenate(all_qsim), np.concatenate(all_qobs)


# ---------------------------------------------------------------------------
# Single-variant training
# ---------------------------------------------------------------------------

def train_variant(
    variant_name: str,
    feature_cols: list[str],
    gauge_dfs: dict[str, pd.DataFrame],
    device: torch.device,
    mode: str = "dev",
) -> None:
    mode_tag = f" [{mode.upper()}]"
    print(f"\n{'='*60}")
    print(f"Variant: {variant_name}  |  features: {len(feature_cols)}  |  seed: {SEED}{mode_tag}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    hp = get_hyperparams(mode)["transformer"]
    seq_len    = hp["seq_len"]
    batch_size = hp["batch_size"]
    num_epochs = hp["num_epochs"]
    patience   = hp["early_stopping_patience"]
    lr         = hp["learning_rate"]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{variant_name}_seed{SEED}_best.pt"
    scaler_path = MODEL_DIR / f"scaler_{variant_name}.pkl"
    pred_dir   = Path("output/predictions") / variant_name / f"seed{SEED}"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # -- Build combined training DataFrame for scaler fitting -----------------
    all_dfs = []
    for gauge_id, df in gauge_dfs.items():
        d = df.copy()
        d["gauge_id"] = gauge_id
        all_dfs.append(d)
    all_df = pd.concat(all_dfs, ignore_index=True)
    all_df = all_df[
        (all_df["date"] >= SPLIT_DATES["val"][0])
        & (all_df["date"] <= SPLIT_DATES["test"][1])
    ].dropna(subset=[TARGET]).reset_index(drop=True)

    train_df = all_df[
        (all_df["date"] >= SPLIT_DATES["train"][0])
        & (all_df["date"] <= SPLIT_DATES["train"][1])
    ].reset_index(drop=True)

    # -- Fit scaler on training data only -------------------------------------
    print("Fitting scaler...")
    scaler = fit_and_save_scaler(train_df, scaler_path, feature_cols=feature_cols)

    # -- Per-gauge scaled splits for DataLoaders ------------------------------
    train_per_gauge, val_per_gauge = {}, {}
    for gauge_id, gdf in gauge_dfs.items():
        gdf_clean = gdf[
            (gdf["date"] >= SPLIT_DATES["val"][0])
            & (gdf["date"] <= SPLIT_DATES["test"][1])
        ].dropna(subset=[TARGET]).reset_index(drop=True)
        gdf_scaled = apply_scaler(gdf_clean, scaler, feature_cols=feature_cols)
        train_per_gauge[gauge_id] = gdf_scaled[
            (gdf_scaled["date"] >= SPLIT_DATES["train"][0])
            & (gdf_scaled["date"] <= SPLIT_DATES["train"][1])
        ].reset_index(drop=True)
        val_per_gauge[gauge_id] = gdf_scaled[
            (gdf_scaled["date"] >= SPLIT_DATES["val"][0])
            & (gdf_scaled["date"] <= SPLIT_DATES["val"][1])
        ].reset_index(drop=True)

    train_ds = build_concat_dataset(train_per_gauge, seq_len, feature_cols=feature_cols)
    val_ds   = build_concat_dataset(val_per_gauge,   seq_len, feature_cols=feature_cols)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # -- Model ----------------------------------------------------------------
    input_size = len(feature_cols)
    model = build_transformer_model(input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=hp["lr_scheduler_patience"],
        factor=hp["lr_scheduler_factor"],
    )
    criterion = nn.MSELoss()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # -- Training loop --------------------------------------------------------
    print("Training...")
    t0 = time.time()
    best_val_loss  = float("inf")
    patience_count = 0
    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        tl = train_epoch(model, train_loader, optimizer, criterion, device)
        vl = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(vl)
        train_losses.append(tl)
        val_losses.append(vl)

        lr_now = optimizer.param_groups[0]["lr"]
        marker = ""
        if vl < best_val_loss:
            best_val_loss  = vl
            patience_count = 0
            torch.save(model.state_dict(), model_path)
            marker = " *"
        else:
            patience_count += 1

        print(f"  Epoch {epoch:03d} | train={tl:.4f} | val={vl:.4f} | lr={lr_now:.2e}{marker}")

        if patience_count >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    train_time = time.time() - t0
    print(f"  Best val loss: {best_val_loss:.4f} | time: {train_time:.0f}s")

    # Save loss curves
    pd.DataFrame({
        "epoch": range(1, len(train_losses) + 1),
        "train_loss": train_losses,
        "val_loss": val_losses,
    }).to_parquet(MODEL_DIR / f"{variant_name}_seed{SEED}_loss_curves.parquet", index=False)

    # -- Per-gauge inference --------------------------------------------------
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("\nRunning per-gauge inference...")

    for gauge_id, raw_df in gauge_dfs.items():
        g_df = raw_df[
            (raw_df["date"] >= SPLIT_DATES["val"][0])
            & (raw_df["date"] <= SPLIT_DATES["test"][1])
        ].dropna(subset=[TARGET]).reset_index(drop=True)

        if len(g_df) < seq_len:
            print(f"  {gauge_id}: skipped (only {len(g_df)} rows)")
            continue

        g_scaled = apply_scaler(g_df, scaler, feature_cols=feature_cols)
        g_ds     = StreamflowDataset(g_scaled, seq_len, feature_cols=feature_cols)
        g_loader = DataLoader(g_ds, batch_size=batch_size, shuffle=False)
        qsim_g, qobs_g = run_inference(model, g_loader, device)

        dates = g_df["date"].iloc[seq_len - 1:].reset_index(drop=True)
        out_df = pd.DataFrame({"date": dates, "qobs": qobs_g, "qsim": qsim_g})
        out_path = pred_dir / f"nepal_{gauge_id}_{variant_name}.parquet"
        out_df.to_parquet(out_path, index=False)
        print(f"  {gauge_id}: {len(out_df)} rows")

    print(f"\nVariant '{variant_name}' complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(mode: str = "dev") -> None:
    from shared.hyperparameters import DEVICE

    device_str = (
        "cuda" if (DEVICE == "auto" and torch.cuda.is_available())
        else (DEVICE if DEVICE != "auto" else "cpu")
    )
    device = torch.device(device_str)
    print(f"Device: {device} | mode={mode}")

    # Load base gauge DataFrames (dynamic + static columns)
    print("\nLoading gauge data...")
    base_gauge_dfs = load_gauge_dfs(INPUT_DIR)
    print(f"  {len(base_gauge_dfs)} gauges loaded")

    # Load AlphaEarth embeddings and attach to gauge DataFrames
    if not EMBEDDINGS_PATH.exists():
        print(
            f"\nERROR: {EMBEDDINGS_PATH} not found.\n"
            "Run preprocessing/05_get_alphaearth_embeddings.py first."
        )
        raise SystemExit(1)

    ae_gauge_dfs, ae_cols = attach_alphaearth(base_gauge_dfs)
    print(f"  AlphaEarth embeddings: {len(ae_cols)} dimensions")

    # Define the three variants
    variants = {
        "transformer_dynamic":    (DYNAMIC_FEATURES,              base_gauge_dfs),
        "transformer_static":     (DYNAMIC_FEATURES + STATIC_FEATURES, base_gauge_dfs),
        "transformer_alphaearth": (DYNAMIC_FEATURES + ae_cols,    ae_gauge_dfs),
    }

    for variant_name, (feature_cols, gauge_dfs) in variants.items():
        train_variant(variant_name, feature_cols, gauge_dfs, device, mode=mode)

    print("\n=== All variants complete. Run experiment_ae/evaluate_ablation.py next. ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input ablation Transformer experiment")
    parser.add_argument(
        "--mode", choices=["dev", "production"], default="dev",
        help="Hyperparameter profile: 'dev' (default, fast smoke-test) or 'production' (full run)",
    )
    args = parser.parse_args()
    main(mode=args.mode)
