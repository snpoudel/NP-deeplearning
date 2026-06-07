# NP-deeplearning

This repository implements LSTM-based deep learning models, physical models (GloFAS: Global Flood Awareness System and GRFR: Global Reach-Level Flood Reanalysis), and hybrid post-processor models that correct physical-model outputs using LSTM or transformer post-processing. Models are trained and evaluated on 15 river basins in Nepal and are compared systematically.

## Repository structure

Note: The original codebase with additional datasets is available at office computer: C:\Sandeep\NP-deeplearning (delete this line after project is completed)
```
preprocessing/         # watershed delineation, ERA5 downloads, and preprocessing
input/                 # prepared input parquet files per watershed
shared/                # shared code: models, dataset, hyperparameters
    ├── models.py
    ├── dataset.py
    └── hyperparameters.py
output/                # model outputs (figures, models, predictions)
01_run_lstm.py         # train, validate, and predict with LSTM
02_run_transformer.py  # train, validate, and predict with transformer
03_run_glofas_lstm.py  # LSTM post-processor for GloFAS
04_run_grfr_lstm.py    # LSTM post-processor for GRFR
05_run_glofas_transformer.py # transformer post-processor for GloFAS
06_run_grfr_transformer.py   # transformer post-processor for GRFR
```

## Description

- **preprocessing/**: Delineates watersheds, downloads ERA5 inputs for each watershed, and merges inputs with observed discharge. Outputs are saved as parquet files in `input/`.
- **shared/**: Contains common code used by all experiments: model architectures (`models.py`), data-preparation utilities (`dataset.py`), and global hyperparameters (`hyperparameters.py`).
- **output/**: Stores model artifacts, predictions, and figures.
- **01_run_lstm.py**: Train, validate, and generate predictions using the LSTM model.
- **02_run_transformer.py**: Train, validate, and generate predictions using the transformer model.
- **03_run_glofas_lstm.py**: Train/validate/predict using an LSTM post-processor that corrects GloFAS outputs.
- **04_run_grfr_lstm.py**: Train/validate/predict using an LSTM post-processor that corrects GRFR outputs.
- **05_run_glofas_transformer.py**: Transformer-based post-processor for GloFAS.
- **06_run_grfr_transformer.py**: Transformer-based post-processor for GRFR.