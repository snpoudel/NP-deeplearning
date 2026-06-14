# NP-deeplearning

Deep learning and hybrid models for streamflow prediction in **15 Nepal river basins**, benchmarked against GloFAS and GRFR global physical model outputs.

## Project goal

Evaluate whether deep learning models (LSTM, Transformer) and hybrid post-processors that correct physical model residuals can outperform GloFAS and GRFR on streamflow prediction. Models are trained on ERA5-Land climate forcing and Google AlphaEarth satellite embeddings, evaluated on NSE, KGE, RMSE, and PBIAS over a held-out 2005–2014 test period.

## Models

| Type | Models | Approach |
|------|--------|----------|
| Physical (baseline) | GloFAS, GRFR | Pre-computed global reanalysis outputs |
| Pure DL | LSTM, Transformer | Predict streamflow directly from 66 input features |
| Hybrid post-processor | GloFAS+LSTM, GloFAS+Transformer, GRFR+LSTM, GRFR+Transformer | Predict `residual = qobs − q_physical`; final `qsim = q_physical + residual` |

All DL models share the same architecture, hyperparameters, and 66-feature input. Only the prediction target differs (streamflow vs. residual).

## Input features (66 total)

- **2 dynamic** (daily, ERA5-Land): `temperature_2m_mean`, `total_precipitation_sum`
- **64 static** (per basin): Google AlphaEarth Foundations satellite embeddings (`emb_0`…`emb_63`, CC-BY 4.0)
  — selected over hand-crafted static attributes via input ablation experiment (`experiment_ae/`)

## Data splits

| Split | Period | Role |
|-------|--------|------|
| Train | 1990–2004 | Model and scaler fitting |
| Validation | 1980–1989 | Early stopping |
| Test | 2005–2014 | Final evaluation (held out) |

## Repository structure

```
preprocessing/               # watershed delineation, ERA5 downloads, AlphaEarth extraction (complete — do not modify)
input/                       # per-basin parquet files (nepal_*_merged.parquet) + alphaearth_embeddings.parquet
shared/
    ├── models.py            # LSTMModel, TransformerModel, build functions, metrics (nse/kge/rmse/pbias)
    ├── dataset.py           # feature lists, scaler helpers, StreamflowDataset, attach_alphaearth
    └── hyperparameters.py   # dev/production profiles, seeds, train-val-test splits
experiment_ae/               # completed input ablation: AlphaEarth vs static vs dynamic-only
    ├── run_ablation.py      # trains 3 LSTM input variants (single seed)
    └── evaluate_ablation.py # 2×2 boxplot comparison (NSE, KGE, RMSE, PBIAS)
output/                      # model weights, predictions, figures, metrics (git-ignored)
run_all.py                   # orchestrator: train all models → aggregate across seeds → evaluate
01_run_lstm.py               # pure LSTM (fits and saves scaler)
02_run_transformer.py        # pure Transformer (loads scaler from 01)
03_run_glofas_lstm.py        # GloFAS residual-correction LSTM
04_run_grfr_lstm.py          # GRFR residual-correction LSTM
05_run_glofas_transformer.py # GloFAS residual-correction Transformer
06_run_grfr_transformer.py   # GRFR residual-correction Transformer
07_evaluate.py               # all evaluation figures (fig1–fig9)
08_residual_analysis.py      # post-processor residual analysis figures
```

## Setup

```bash
git clone https://github.com/snpoudel/NP-deeplearning.git
cd NP-deeplearning
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **GPU:** Install PyTorch with CUDA support from https://pytorch.org before running `pip install -r requirements.txt`.

## How to run

### Recommended: full pipeline

```bash
python run_all.py --mode dev         # smoke-test: 1 seed, small model, 5 epochs (~8 min on GPU)
python run_all.py --mode production  # full run: 5 seeds, production model, up to 100 epochs
python run_all.py --mode production --skip-eval  # train + aggregate only, skip evaluation
```

`run_all.py` trains all 6 DL models × N seeds in dependency order (01 → 02 → … → 06), aggregates predictions and loss curves across seeds, then runs `07_evaluate.py`.

### Individual scripts (debugging)

Scripts can be run standalone with `--mode dev|production`. Scripts 02–06 require `output/model/scaler.pkl` produced by script 01.

```bash
python 01_run_lstm.py --mode dev
python 07_evaluate.py
```

## Configuration — shared/hyperparameters.py

| Setting | Dev | Production |
|---------|-----|------------|
| Seeds | `[42]` | `[42, 123, 456, 789, 2024]` |
| `hidden_size` (LSTM) | 8 | 256 |
| `d_model` (Transformer) | 8 | 128 |
| `seq_len` | 3 | 365 days |
| `num_epochs` | 5 | 100 |
| `early_stopping_patience` | 2 | 10 |
| `DEVICE` | `"auto"` (detects CUDA) | same |

`get_hyperparams(mode)` returns the correct profile. `build_lstm_model(input_size, mode)` and `build_transformer_model(input_size, mode)` use it to set architecture size.

## Output structure

```
output/
├── model/
│   ├── scaler.pkl                                    # StandardScaler fitted on train split; shared by all 6 models
│   ├── {model}_seed{N}_best.pt                      # best model weights per seed
│   ├── {model}_seed{N}_loss_curves.parquet          # per-seed train/val loss history
│   └── {model}_loss_curves.parquet                  # seed-averaged loss curves (fig8 input)
├── predictions/
│   ├── {model}/seed{N}/nepal_{id}_{model}.parquet   # per-seed predictions (date, qobs, qsim)
│   └── {model}/nepal_{id}_{model}_mean.parquet      # seed-averaged predictions (07_evaluate.py input)
├── training_times.csv   # one row per model: seeds_run, mean epochs, mean train time, mean best val loss
├── metrics.parquet      # NSE/KGE/RMSE/PBIAS per model per gauge (test period only)
└── figures/             # fig1–fig9 PNG + SVG at 300 DPI
```

## Key design decisions

- **Scaler sharing**: fitted once in script 01 on the 1990–2004 training split; deterministic, safely reused by scripts 02–06 across all seeds and models.
- **No data leakage**: scaler never sees val or test data; per-gauge sequences never cross gauge boundaries (`build_concat_dataset`).
- **AlphaEarth embeddings**: 64-dim satellite embeddings sampled at basin centroids (2017–2024 average, ~1280 m resolution); de-quantized from int8 via `((x / 127.5)²) × sign(x)`.
- **Multi-seed aggregation**: only `qsim` is averaged across seeds date-by-date; `qobs` and physical model columns are deterministic and taken from the first seed.
