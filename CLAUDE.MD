# NP-deeplearning

Streamflow prediction for 15 Nepal river basins. Compares GloFAS and GRFR physical model baselines against pure deep learning models (LSTM, Transformer) and four hybrid variants that use a DL post-processor to correct physical model residuals.

## Repository structure

```
preprocessing/               # watershed delineation, ERA5/AlphaEarth downloads
input/                       # per-basin parquet files + alphaearth_embeddings.parquet
shared/
    ├── models.py            # LSTMModel, TransformerModel, metrics (nse/kge/rmse/pbias)
    ├── dataset.py           # feature defs, scaler helpers, StreamflowDataset, attach_alphaearth
    └── hyperparameters.py   # seeds, splits, dev/production hyperparameter profiles
experiment_ae/               # Input ablation experiment (AlphaEarth vs static vs dynamic)
output/                      # model weights, predictions, figures, metrics
run_all.py                   # orchestrator: train → aggregate → evaluate
01_run_lstm.py               # pure LSTM
02_run_transformer.py        # pure Transformer
03_run_glofas_lstm.py        # GloFAS residual-correction LSTM
04_run_grfr_lstm.py          # GRFR residual-correction LSTM
05_run_glofas_transformer.py # GloFAS residual-correction Transformer
06_run_grfr_transformer.py   # GRFR residual-correction Transformer
07_evaluate.py               # evaluation figures (fig1–fig9)
08_residual_analysis.py      # residual analysis figures
```

## Data splits

Train: 1990–2004 | Val: 1980–1989 | Test: 2005–2014 | 15 Nepal gauge basins

## Input features (66 total)

- **2 dynamic**: `temperature_2m_mean`, `total_precipitation_sum` (ERA5-Land, daily)
- **64 static**: Google AlphaEarth Foundations satellite embeddings (`emb_0`…`emb_63`)
  - Source: `gs://alphaearth_foundations/satellite_embedding/v1/annual/` (CC-BY 4.0)
  - Extracted by `preprocessing/05_get_alphaearth_embeddings.py` → `input/alphaearth_embeddings.parquet`
  - Selected over hand-crafted static attributes by the input ablation experiment (`experiment_ae/`)

## Pipeline

```bash
python run_all.py --mode dev         # smoke-test: seed=[42], small model, 5 epochs
python run_all.py --mode production  # full run: seeds=[42,123,456,789,2024], 100 epochs
```

Trains all 6 models × N seeds → averages predictions and loss curves across seeds → runs `07_evaluate.py`. Pass `--skip-eval` to stop before evaluation. Individual scripts can be run standalone for debugging.

## Model logic

**Pure DL** (01, 02): predict `qsim` directly from 66 input features.

**Residual post-processors** (03–06): predict `residual = qobs − q_physical`; final prediction `qsim = q_physical + predicted_residual`. Reuse the scaler fitted by script 01 — always run 01 before 03–06 (`run_all.py` guarantees this order).

`build_lstm_model(input_size, mode)` and `build_transformer_model(input_size, mode)` select architecture size based on mode.

## Output structure

```
output/
├── model/
│   ├── scaler.pkl                           # StandardScaler fitted on train split by script 01; shared by all models
│   ├── lstm_seed42_best.pt                  # per-seed model weights
│   ├── lstm_seed42_loss_curves.parquet      # per-seed loss history
│   ├── lstm_loss_curves.parquet             # seed-averaged loss curves (used by fig8)
│   └── ...                                  # same pattern for all 6 models
├── predictions/
│   ├── lstm/
│   │   ├── seed42/nepal_120_lstm.parquet    # per-seed predictions (columns: date, qobs, qsim)
│   │   └── nepal_120_lstm_mean.parquet      # seed-averaged final predictions (loaded by 07_evaluate.py)
│   └── ...                                  # same for all 6 models
├── training_times.csv                       # one aggregated row per model: seeds_run, mean epochs/time/val_loss
├── metrics.parquet                          # per-model per-gauge test-period metrics
└── figures/                                 # fig1–fig9 PNG and SVG
```

## Important notes

- Use a virtual environment in the repo root. If it does not exist, create it with `python -m venv venv`, install dependencies with `pip install -r requirements.txt`, activate it with `source venv/Scripts/activate` (Bash) or `.\venv\Scripts\Activate.ps1` (PowerShell), and use it for all scripts to avoid dependency issues.
- **GPU VM (`vm-sandeep`)**: shared GCE A100 VM (`us-central1-f`, project `spherical-berm-323321`), also used by the `Multi-domain-time-series` project one directory up — see that repo's `project_management/environment_spec.md` for the full spec (cost, stop discipline, `ZONE_RESOURCE_POOL_EXHAUSTED` retry policy). On this Mac, SSH via the `vm-sandeep` host alias in `~/.ssh/config`. Repo checkout lives at `~/NP-deeplearning` on the VM (own clone, separate from local); venv at `~/NP-deeplearning/venv/bin/python`, activate with `source venv/bin/activate` (Linux — not the Windows syntax above). `DEVICE = "auto"` in `shared/hyperparameters.py` already resolves to `cuda` when available, so no code changes are needed to run there. Start the VM only right before a GPU run and stop it immediately after, per the same discipline as the sibling project.
- Preprocessing is complete — input parquets and AlphaEarth embeddings are ready in `input/`. Do not modify `preprocessing/`.
- Scripts 02–06 assert `scaler.pkl` exists before running. Always run script 01 first, or use `run_all.py`.
- The scaler is shared across all models and seeds (StandardScaler is deterministic — no randomness).
- Write clean, minimal, maintainable code with clear structure and helpful comments.
- Ask explicitly, if anything is unclear before proceeding, don't make any assumptions.
