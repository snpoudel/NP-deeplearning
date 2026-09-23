# NP-deeplearning

Regional deep learning and hybrid models for streamflow prediction in the Central Himalayan basins of Nepal. Deep learning models (LSTM, Transformer) and hybrid post-processors (which correct physical model residuals) are evaluated against two global physical model baselines: GloFAS and GRFR.

Physical model streamflow products are pre-computed and downloaded from public data portals. The deep learning models are trained on ERA5-Land climate forcing (precipitation and temperature) plus Google AlphaEarth satellite embeddings (64 aggregated features), selected over hand-crafted static basin attributes by an input ablation experiment. Models are evaluated on NSE, KGE, RMSE, and PBIAS over a held-out test period.

## Models

| Type | Models | Approach |
|------|--------|----------|
| Physical (baseline) | GloFAS, GRFR | Pre-computed global reanalysis outputs |
| Pure DL | LSTM, Transformer | Predict streamflow directly from 66 input features |
| Hybrid post-processor | GloFAS+LSTM, GloFAS+Transformer, GRFR+LSTM, GRFR+Transformer | Predict `residual = qobs - q_physical`; final `qsim = q_physical + residual` |

DL models and post-processors share the same architecture, hyperparameters, and 66-feature input. Only the prediction target differs (streamflow vs. residual).

## Input features (66 total)

- **2 dynamic** (daily, ERA5-Land): `temperature_2m_mean`, `total_precipitation_sum`
- **64 static** (per basin): Google AlphaEarth Foundations satellite embeddings (`emb_0`...`emb_63`)

The input ablation experiment (`experiment_ae/`) compares three variants for both LSTM and Transformer: dynamic-only, dynamic + 16 hand-crafted static attributes, and dynamic + AlphaEarth embeddings. AlphaEarth performed best and is used as the production input.

## Hyperparameters

Production hidden size (`hidden_size=256` for LSTM, `d_model=128` for Transformer) was confirmed by a hidden-size sweep over `{64, 128, 256}` for both architectures (`experiment_hp_tuning/`).

## Data splits

| Split | Period | Role |
|-------|--------|------|
| Train | 1990-2004 | Model and scaler fitting |
| Validation | 1980-1989 | Early stopping |
| Test | 2005-2014 | Final evaluation (held out) |

15 Nepal gauge basins.

## Repository structure

```
preprocessing/               # watershed delineation, ERA5 downloads, AlphaEarth extraction (complete, do not modify)
input/                       # per-basin parquet files (nepal_*_merged.parquet) + alphaearth_embeddings.parquet
shared/
    ├── models.py            # LSTMModel, TransformerModel, build functions, metrics (nse/kge/rmse/pbias)
    ├── dataset.py           # feature lists, scaler helpers, StreamflowDataset, attach_alphaearth
    └── hyperparameters.py   # dev/production profiles, seeds, train-val-test splits
experiment_ae/                     # completed input ablation: AlphaEarth vs static vs dynamic-only
    ├── run_ablation.py             # trains 3 LSTM input variants (single seed)
    ├── run_ablation_transformer.py # trains 3 Transformer input variants (single seed)
    └── evaluate_ablation.py        # 2x2 boxplot grid: LSTM/Transformer x NSE/PBIAS
experiment_hp_tuning/              # completed hidden-size sweep: 64 / 128 / 256, AlphaEarth input
    ├── run_tuning_lstm.py          # sweeps hidden_size (single seed)
    ├── run_tuning_transformer.py   # sweeps d_model (single seed)
    └── evaluate_tuning.py          # validation-loss line plot vs. hidden size
output/                      # model weights, predictions, figures, metrics (git-ignored)
run_all.py                   # orchestrator: train all models, aggregate across seeds, evaluate
01_run_lstm.py               # pure LSTM (fits and saves scaler)
02_run_transformer.py        # pure Transformer (loads scaler from 01)
03_run_glofas_lstm.py        # GloFAS residual-correction LSTM
04_run_grfr_lstm.py          # GRFR residual-correction LSTM
05_run_glofas_transformer.py # GloFAS residual-correction Transformer
06_run_grfr_transformer.py   # GRFR residual-correction Transformer
07_evaluate.py               # all evaluation figures (fig1-fig9)
```

## Setup

```bash
git clone https://github.com/snpoudel/NP-deeplearning.git
cd NP-deeplearning
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## How to run

### Full pipeline

```bash
python run_all.py --mode dev         # smoke-test: 1 seed, small model, 5 epochs (~8 min on Tesla T4 GPU)
python run_all.py --mode production  # full run: 5 seeds, production model, up to 100 epochs
python run_all.py --mode production --skip-eval  # train + aggregate only, skip evaluation
```

`run_all.py` trains all 6 DL models x N seeds in dependency order (01 to 06), aggregates predictions and loss curves across seeds, then runs `07_evaluate.py`.

### Individual scripts (debugging)

Scripts can be run standalone with `--mode dev|production`. Scripts 02-06 require `output/model/scaler.pkl`, produced by script 01.

```bash
python 01_run_lstm.py --mode dev
python 07_evaluate.py
```

### Experiments

Both experiments use the AlphaEarth input variant, seed 42, and `--mode dev|production` (default `dev`).

```bash
# Input ablation: dynamic-only vs. + static attributes vs. + AlphaEarth
python experiment_ae/run_ablation.py --mode production              # LSTM
python experiment_ae/run_ablation_transformer.py --mode production  # Transformer
python experiment_ae/evaluate_ablation.py                           # fig_input_ablation_cdf.png

# Hidden-size tuning: hidden_size / d_model in {64, 128, 256}
python experiment_hp_tuning/run_tuning_lstm.py --mode production
python experiment_hp_tuning/run_tuning_transformer.py --mode production
python experiment_hp_tuning/evaluate_tuning.py                      # fig_hp_tuning_val_loss.png
```

## Output structure

```
output/
├── model/
│   ├── scaler.pkl                                    # StandardScaler fitted on train split; shared by all 6 models
│   ├── {model}_seed{N}_best.pt                       # best model weights per seed
│   ├── {model}_seed{N}_loss_curves.parquet           # per-seed train/val loss history
│   └── {model}_loss_curves.parquet                   # seed-averaged loss curves (fig8 input)
├── predictions/
│   ├── {model}/seed{N}/nepal_{id}_{model}.parquet    # per-seed predictions (date, qobs, qsim)
│   └── {model}/nepal_{id}_{model}_mean.parquet       # seed-averaged predictions (07_evaluate.py input)
├── training_times.csv             # one row per model: seeds_run, mean epochs, mean train time, mean best val loss
├── metrics.parquet                # NSE/KGE/RMSE/PBIAS per model per gauge (test period only)
├── metrics_input_ablation.parquet # per-gauge NSE/KGE/RMSE/PBIAS per experiment_ae/ variant
├── metrics_hp_tuning.parquet      # best val loss, epochs, train time per experiment_hp_tuning/ variant
└── figures/                       # fig1-fig9, fig_input_ablation_cdf, fig_hp_tuning_val_loss (PNG + SVG at 300 DPI)
```
