# NP-deeplearning

Deep learning and hybrid models for streamflow prediction in **15 Nepal river basins**, benchmarked against GloFAS and GRFR global physical model outputs.

## Project goal

Evaluate the performance of deep learning models (LSTM, Transformer) against global physical models (GloFAS, GRFR) and hybrid post-processors that correct physical model residuals. Physical model streamflow products are publicly available; the remaining models are trained on ERA5-Land climate forcing (precipitation and temperature) and Google AlphaEarth satellite embeddings (64 aggregated features), and evaluated on NSE, KGE, RMSE, and PBIAS over a held-out test period.

## Models

| Type | Models | Approach |
|------|--------|----------|
| Physical (baseline) | GloFAS, GRFR | Pre-computed global reanalysis outputs |
| Pure DL | LSTM, Transformer | Predict streamflow directly from 66 input features |
| Hybrid post-processor | GloFAS+LSTM, GloFAS+Transformer, GRFR+LSTM, GRFR+Transformer | Predict `residual = qobs − q_physical`; final `qsim = q_physical + residual` |

All DL models share the same architecture, hyperparameters, and 66-feature input. Only the prediction target differs (streamflow vs. residual).

## Input features (66 total)

- **2 dynamic** (daily, ERA5-Land): `temperature_2m_mean`, `total_precipitation_sum`
- **64 static** (per basin): Google AlphaEarth Foundations satellite embeddings (`emb_0`…`emb_63`)
  — selected via an input ablation experiment comparing dynamic-only vs. dynamic + basin attributes vs. dynamic + AlphaEarth (`experiment_ae/`)

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

## How to run

### Full pipeline

```bash
python run_all.py --mode dev         # smoke-test: 1 seed, small model, 5 epochs (~8 min on Tesla T4 GPU)
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