# NP-deeplearning

This repository uses simulations from global physical models (GloFAS: Global Flood Awareness System and GRFR: Global Reach-Level Flood Reanalysis), implements deep learning models (LSTM, Transformer), and hybrid post-processor models that correct physical-model outputs using LSTM or transformer post-processing. Models are trained and evaluated on 15 river basins in Nepal and are compared systematically.

## Repository structure
```
preprocessing/               # watershed delineation, ERA5 downloads, and preprocessing
input/                       # prepared input parquet files per watershed
shared/                      # shared code: models, dataset, hyperparameters
    ├── models.py
    ├── dataset.py
    └── hyperparameters.py
output/                      # model outputs (figures, models, predictions)
run_all.py                   # orchestrator: runs all scripts for all seeds, aggregates, evaluates
01_run_lstm.py               # train, validate, and predict with LSTM
02_run_transformer.py        # train, validate, and predict with transformer
03_run_glofas_lstm.py        # LSTM post-processor for GloFAS
04_run_grfr_lstm.py          # LSTM post-processor for GRFR
05_run_glofas_transformer.py # transformer post-processor for GloFAS
06_run_grfr_transformer.py   # transformer post-processor for GRFR
07_evaluate.py               # evaluate all models and produce figures (fig1–fig8)
```

## Pipeline

The intended workflow is to run the full pipeline via `run_all.py`:
```
python run_all.py
```

This reads `SEEDS` and `DEVICE` from `shared/hyperparameters.py`, trains all 6 models for every seed, averages predictions and loss curves across seeds, then runs evaluation. Pass `--skip-eval` to stop before evaluation.

Individual scripts (`01_run_lstm.py` through `07_evaluate.py`) can still be run standalone for debugging. When run standalone, each script uses `SEEDS[0]` from hyperparameters.

## Configuration — shared/hyperparameters.py

All configuration lives in `shared/hyperparameters.py`. No CLI flags are needed.

Key entries:
- `SEEDS` — list of random seeds for multi-seed training (e.g. `[42, 123, 456]` for production). Each seed produces independent model weights; predictions are averaged across seeds before evaluation.
- `DEVICE` — `"auto"` detects CUDA at runtime; set `"cpu"` or `"cuda"` to force a device.
- `HYPERPARAMS["lstm"]` and `HYPERPARAMS["transformer"]` — model architecture and training settings. Values marked `# dev size` are small for fast testing; update to production values before a full run.

## Output structure

```
output/
├── model/
│   ├── scaler.pkl                              # fitted StandardScaler (deterministic, one copy)
│   ├── lstm_seed42_best.pt                     # per-seed model weights
│   ├── lstm_seed42_loss_curves.parquet         # per-seed training/validation loss history
│   ├── lstm_loss_curves.parquet                # seed-averaged loss curves (input to fig8)
│   └── ... (same pattern for all 6 models)
├── predictions/
│   ├── lstm/
│   │   ├── seed42/nepal_120_lstm.parquet       # per-seed raw predictions
│   │   └── nepal_120_lstm.parquet              # seed-averaged final predictions
│   └── ... (same for all 6 models)
├── training_times.csv                          # model, seed, epochs_run, best_val_loss, train_time_seconds
├── metrics.parquet                             # per-model per-gauge evaluation metrics
└── figures/                                    # fig1–fig8 PNG and SVG
```

## Notes
- Preprocessing is already completed. Final parquet datasets for all 15 basins are in `input/`. Don't inspect or modify anything in the `preprocessing/` folder.
- The scaler (`output/model/scaler.pkl`) is fitted once in `01_run_lstm.py` and reused by scripts 02–06. It is deterministic (StandardScaler has no randomness), so it is safe to share across seeds.
- Scripts 02–06 assert `scaler.pkl` exists before running. Always run script 01 first (or use `run_all.py`).
- Build and debug incrementally, ensuring each stage works before moving forward.
- Write clean, minimal, and maintainable code: clear structure, helpful comments, proper docstrings, and readability for a single researcher to maintain easily.
- Do not make any assumptions; ask explicitly and loudly if anything is unclear.
