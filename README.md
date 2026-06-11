# NP-deeplearning

Deep learning and hybrid post-processor models for streamflow prediction in 15 Nepal river basins, benchmarked against GloFAS and GRFR physical model outputs.

## Repository structure

```
preprocessing/               # watershed delineation, ERA5 downloads, preprocessing (complete — do not modify)
input/                       # prepared input parquet files per watershed
shared/                      # shared code: models, dataset, hyperparameters
    ├── models.py
    ├── dataset.py
    └── hyperparameters.py
output/                      # model outputs (figures, models, predictions) — git-ignored
run_all.py                   # orchestrator: runs all scripts for all seeds, aggregates, evaluates
01_run_lstm.py               # train, validate, and predict with LSTM
02_run_transformer.py        # train, validate, and predict with transformer
03_run_glofas_lstm.py        # LSTM post-processor for GloFAS
04_run_grfr_lstm.py          # LSTM post-processor for GRFR
05_run_glofas_transformer.py # transformer post-processor for GloFAS
06_run_grfr_transformer.py   # transformer post-processor for GRFR
07_evaluate.py               # evaluate all models and produce figures
```

## Setup

```bash
git clone https://github.com/snpoudel/NP-deeplearning.git
cd NP-deeplearning
python -m venv .venv
# Windows: .venv\Scripts\activate  |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

> **GPU users:** Install PyTorch with CUDA support from https://pytorch.org/get-started/locally/ before running `pip install -r requirements.txt`.

## How to run

### Full pipeline (recommended)

```bash
python run_all.py
```

This runs all 6 training scripts for every seed defined in `shared/hyperparameters.py`, averages predictions and loss curves across seeds, then runs evaluation. All configuration is read from `shared/hyperparameters.py` — no command-line flags needed.

```bash
python run_all.py --skip-eval   # stop after aggregation, skip 07_evaluate.py
```

### Individual scripts (for debugging)

Individual scripts can also be run standalone. Each uses `SEEDS[0]` and `DEVICE` from `shared/hyperparameters.py`.

| Script | What it does | Key output |
|--------|--------------|------------|
| `python 01_run_lstm.py` | LSTM baseline | `output/predictions/lstm/seed{N}/` |
| `python 02_run_transformer.py` | Transformer baseline | `output/predictions/transformer/seed{N}/` |
| `python 03_run_glofas_lstm.py` | LSTM post-processor for GloFAS | `output/predictions/glofas_lstm/seed{N}/` |
| `python 04_run_grfr_lstm.py` | LSTM post-processor for GRFR | `output/predictions/grfr_lstm/seed{N}/` |
| `python 05_run_glofas_transformer.py` | Transformer post-processor for GloFAS | `output/predictions/glofas_transformer/seed{N}/` |
| `python 06_run_grfr_transformer.py` | Transformer post-processor for GRFR | `output/predictions/grfr_transformer/seed{N}/` |
| `python 07_evaluate.py` | Metrics and figures | `output/figures/`, `output/metrics.parquet` |

Scripts 02–06 require `output/model/scaler.pkl` from script 01. Run script 01 first when running individually.

## Configuration

All configuration is in [shared/hyperparameters.py](shared/hyperparameters.py). Edit this file to change behavior — no CLI flags are needed.

| Setting | Default | Description |
|---------|---------|-------------|
| `SEEDS` | `[42]` | Random seeds for multi-seed runs. Add more seeds (e.g. `[42, 123, 456]`) for production. Predictions are averaged across seeds before evaluation. |
| `DEVICE` | `"auto"` | `"auto"` detects CUDA at runtime. Set `"cpu"` or `"cuda"` to force a device. |
| `HYPERPARAMS["lstm"]` | dev sizes | Architecture and training settings. Values marked `# dev size` are small for fast testing; update to the production values noted alongside before a full run. |
| `HYPERPARAMS["transformer"]` | dev sizes | Same as LSTM but for the Transformer architecture. |

## Outputs

After a full pipeline run, the following are written to `output/`:

| Path | Description |
|------|-------------|
| `model/scaler.pkl` | Fitted StandardScaler (deterministic, shared across seeds) |
| `model/{model}_seed{N}_best.pt` | Best model weights for each model × seed |
| `model/{model}_seed{N}_loss_curves.parquet` | Per-seed training and validation loss history |
| `model/{model}_loss_curves.parquet` | Seed-averaged loss curves (used in fig8) |
| `predictions/{model}/seed{N}/` | Per-seed raw predictions (date, qobs, qsim) |
| `predictions/{model}/` | Seed-averaged final predictions (input to evaluate.py) |
| `training_times.csv` | Training time, epochs run, and best val loss per model × seed |
| `metrics.parquet` | NSE, KGE, RMSE, PBIAS per model × gauge (test period) |
| `figures/fig1_basin_map.*` | Study basin map |
| `figures/fig2_obs_availability.*` | Observation availability timeline |
| `figures/fig3_timeseries.*` | Test-period time series for representative basin |
| `figures/fig4_cdf_metrics.*` | CDF of NSE and KGE across basins |
| `figures/fig5_bias.*` | PBIAS boxplots (overall, high, low, mid flow) |
| `figures/fig6_peak_flow.*` | Peak flow CDF |
| `figures/fig7_nse_maps.*` | Choropleth NSE maps for all models |
| `figures/fig8_loss_curves.*` | Training and validation loss curves for all DL models |
