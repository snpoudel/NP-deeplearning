# NP-deeplearning

This repository implements LSTM-based deep learning models, physical models (GloFAS: Global Flood Awareness System and GRFR: Global Reach-Level Flood Reanalysis), and hybrid post-processor models that correct physical-model outputs using LSTM or transformer post-processing. Models are trained and evaluated on 15 river basins in Nepal and are compared systematically.

## Repository structure

```
preprocessing/               # watershed delineation, ERA5 downloads, and preprocessing
input/                       # prepared input parquet files per watershed
shared/                      # shared code: models, dataset, hyperparameters
    ├── models.py
    ├── dataset.py
    └── hyperparameters.py
output/                      # model outputs (figures, models, predictions) — git-ignored
01_run_lstm.py               # train, validate, and predict with LSTM
02_run_transformer.py        # train, validate, and predict with transformer
03_run_glofas_lstm.py        # LSTM post-processor for GloFAS
04_run_grfr_lstm.py          # LSTM post-processor for GRFR
05_run_glofas_transformer.py # transformer post-processor for GloFAS
06_run_grfr_transformer.py   # transformer post-processor for GRFR
07_evaluate.py               # evaluate all models and produce figures
```

## Description

- **preprocessing/**: Delineates watersheds, downloads ERA5 inputs for each watershed, and merges inputs with observed discharge. Outputs are saved as parquet files in `input/`. Preprocessing is already complete — do not modify this folder.
- **shared/**: Common code used by all experiments: model architectures (`models.py`), data-preparation utilities (`dataset.py`), and global hyperparameters (`hyperparameters.py`).
- **output/**: Stores model artifacts, predictions, and figures. This directory is git-ignored; all outputs are generated locally by running the scripts.
- **01_run_lstm.py**: Train, validate, and generate predictions using the LSTM model.
- **02_run_transformer.py**: Train, validate, and generate predictions using the transformer model.
- **03_run_glofas_lstm.py**: Train/validate/predict using an LSTM post-processor that corrects GloFAS outputs.
- **04_run_grfr_lstm.py**: Train/validate/predict using an LSTM post-processor that corrects GRFR outputs.
- **05_run_glofas_transformer.py**: Transformer post-processor for GloFAS.
- **06_run_grfr_transformer.py**: Transformer post-processor for GRFR.
- **07_evaluate.py**: Compute metrics for all eight models across all basins and produce publication-quality figures.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/snpoudel/NP-deeplearning.git
cd NP-deeplearning
```

### 2. Create and activate a virtual environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** The default `torch` install above is CPU-only. For GPU support, install PyTorch separately following the instructions at https://pytorch.org/get-started/locally/ before running `pip install -r requirements.txt`.

## How to run

Scripts must be run **in order**. Each script depends on outputs produced by the previous ones. Run all commands from the root of the repository with the virtual environment activated.

### Step 1 — LSTM baseline

Trains a global multi-basin LSTM, saves the best model and scaler, and writes per-gauge predictions to `output/predictions/lstm/`.

```bash
python 01_run_lstm.py
```

### Step 2 — Transformer baseline

Trains a global multi-basin Transformer using the same scaler fitted in Step 1, and writes per-gauge predictions to `output/predictions/transformer/`.

```bash
python 02_run_transformer.py
```

### Step 3 — GloFAS + LSTM post-processor

Trains an LSTM that corrects GloFAS physical-model output. Reads GloFAS simulations from `input/physical_model/` and writes predictions to `output/predictions/glofas_lstm/`.

```bash
python 03_run_glofas_lstm.py
```

### Step 4 — GRFR + LSTM post-processor

Same as Step 3 but for GRFR. Writes predictions to `output/predictions/grfr_lstm/`.

```bash
python 04_run_grfr_lstm.py
```

### Step 5 — GloFAS + Transformer post-processor

Transformer-based post-processor for GloFAS. Writes predictions to `output/predictions/glofas_transformer/`.

```bash
python 05_run_glofas_transformer.py
```

### Step 6 — GRFR + Transformer post-processor

Transformer-based post-processor for GRFR. Writes predictions to `output/predictions/grfr_transformer/`.

```bash
python 06_run_grfr_transformer.py
```

### Step 7 — Evaluation and figures

Computes NSE, KGE, RMSE, and PBIAS for all eight models across all 15 basins and produces publication-quality figures saved to `output/figures/`.

```bash
python 07_evaluate.py
```

## Hyperparameters

All hyperparameters (sequence length, batch size, hidden size, epochs, etc.) are defined in `shared/hyperparameters.py`. The repository ships with small development-sized values for fast testing. Before a production run, update the values marked with `# dev size` comments to the production values noted alongside them.
