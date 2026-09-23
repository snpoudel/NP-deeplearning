"""
experiment_ae/run_ablation.py

Input feature ablation experiment for LSTM.

Trains two LSTM variants, each across every seed in the run mode's seed list
(SEEDS[mode]), then averages predictions across seeds, mirroring how the
main pipeline (01_run_lstm.py + run_all.py) produces its seed-averaged result:

  A) lstm_dynamic (2 features: temperature + precipitation)
  B) lstm_static  (18 features: dynamic + existing 16 static attrs)

A third variant, "dynamic + AlphaEarth embeddings", is intentionally not
retrained here: it is feature-for-feature and hyperparameter-for-hyperparameter
identical to the main pipeline's production `lstm` model (both train on
ALL_FEATURES = DYNAMIC_FEATURES + AE_FEATURES). Retraining it separately would
only add fresh GPU non-determinism noise without changing what it represents,
so evaluate_ablation.py instead reuses the already seed-averaged
output/predictions/lstm/*_mean.parquet produced by the main pipeline.

Everything else (architecture, hyperparameters, train/val/test split) is
identical to 01_run_lstm.py. Predictions are saved in the same format as the
main pipeline so evaluate_ablation.py can compare them directly.

Usage:
    python experiment_ae/run_ablation.py                  # production: 5 seeds
    python experiment_ae/run_ablation.py --mode dev       # fast smoke-test: 1 seed
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
from shared.hyperparameters import DEV_SEEDS, PROD_SEEDS, get_hyperparams, SPLIT_DATES
from shared.models import build_lstm_model

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INPUT_DIR = Path("input")
MODEL_DIR = Path("output/model")
PRED_BASE = Path("output/predictions")


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


# ---------------------------------------------------------------------------
# Training / evaluation helpers (identical to 01_run_lstm.py)
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
# Single-variant, single-seed training
# ---------------------------------------------------------------------------

def train_variant(
    variant_name: str,
    feature_cols: list[str],
    gauge_dfs: dict[str, pd.DataFrame],
    device: torch.device,
    seed: int,
    mode: str = "dev",
) -> None:
    mode_tag = f" [{mode.upper()}]"
    print(f"\n{'='*60}")
    print(f"Variant: {variant_name}  |  features: {len(feature_cols)}  |  seed: {seed}{mode_tag}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    hp = get_hyperparams(mode)["lstm"]
    seq_len    = hp["seq_len"]
    batch_size = hp["batch_size"]
    num_epochs = hp["num_epochs"]
    patience   = hp["early_stopping_patience"]
    lr         = hp["learning_rate"]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{variant_name}_seed{seed}_best.pt"
    scaler_path = MODEL_DIR / f"scaler_{variant_name}.pkl"
    pred_dir   = PRED_BASE / variant_name / f"seed{seed}"
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
    # Deterministic (StandardScaler has no randomness), so re-fitting per seed
    # is harmless: every seed writes back the same values.
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
    model = build_lstm_model(input_size).to(device)
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
    }).to_parquet(MODEL_DIR / f"{variant_name}_seed{seed}_loss_curves.parquet", index=False)

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

    print(f"\nVariant '{variant_name}' (seed {seed}) complete.")


# ---------------------------------------------------------------------------
# Cross-seed aggregation (mirrors run_all.py's aggregate_predictions)
# ---------------------------------------------------------------------------

def aggregate_variant_predictions(variant_name: str, seeds: list[int]) -> None:
    """Average per-seed qsim predictions across seeds; write *_mean.parquet
    files to the top-level output/predictions/{variant_name}/ directory.

    qobs is taken from the first seed since it is data-derived, not
    model-derived, and therefore identical across seeds.
    """
    pred_base = PRED_BASE / variant_name
    first_seed_dir = pred_base / f"seed{seeds[0]}"
    if not first_seed_dir.exists():
        print(f"  Warning: {first_seed_dir} not found — skipping aggregation for {variant_name}")
        return

    gauge_files = sorted(first_seed_dir.glob("*.parquet"))
    for gauge_file in gauge_files:
        seed_dfs = []
        for seed in seeds:
            seed_path = pred_base / f"seed{seed}" / gauge_file.name
            if seed_path.exists():
                seed_dfs.append(pd.read_parquet(seed_path))
            else:
                print(f"  Warning: missing {seed_path}")

        if not seed_dfs:
            continue

        if len(seed_dfs) == 1:
            merged = seed_dfs[0].copy()
        else:
            non_sim_cols = [c for c in seed_dfs[0].columns if c != "qsim"]
            merged = seed_dfs[0][non_sim_cols].copy()
            qsim_stack = np.stack(
                [df.set_index("date")["qsim"].values for df in seed_dfs], axis=0
            )
            merged["qsim"] = qsim_stack.mean(axis=0)

        out_path = pred_base / gauge_file.name.replace(".parquet", "_mean.parquet")
        merged.to_parquet(out_path, index=False)

    print(f"  {variant_name}: {len(gauge_files)} gauges aggregated across {len(seeds)} seed(s)")


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
    seeds = PROD_SEEDS if mode == "production" else DEV_SEEDS
    print(f"Device: {device} | mode={mode} | seeds={seeds}")

    # Load base gauge DataFrames (dynamic + static columns)
    print("\nLoading gauge data...")
    base_gauge_dfs = load_gauge_dfs(INPUT_DIR)
    print(f"  {len(base_gauge_dfs)} gauges loaded")

    # Define the two variants actually trained here. "dynamic + AlphaEarth" is
    # not retrained (see module docstring): it is identical to the main
    # pipeline's production `lstm` model, so evaluate_ablation.py reuses
    # output/predictions/lstm/*_mean.parquet.
    variants = {
        "lstm_dynamic": (DYNAMIC_FEATURES,                     base_gauge_dfs),
        "lstm_static":  (DYNAMIC_FEATURES + STATIC_FEATURES,   base_gauge_dfs),
    }

    for variant_name, (feature_cols, gauge_dfs) in variants.items():
        for seed in seeds:
            train_variant(variant_name, feature_cols, gauge_dfs, device, seed=seed, mode=mode)
        aggregate_variant_predictions(variant_name, seeds)

    print("\n=== All variants complete. Run experiment_ae/evaluate_ablation.py next. ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input ablation LSTM experiment")
    parser.add_argument(
        "--mode", choices=["dev", "production"], default="dev",
        help="Hyperparameter profile: 'dev' (default, fast smoke-test, 1 seed) "
             "or 'production' (full run, 5 seeds)",
    )
    args = parser.parse_args()
    main(mode=args.mode)
