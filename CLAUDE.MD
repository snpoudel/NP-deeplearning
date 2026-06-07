# NP-deeplearning

This project implements LSTM-based deep learning models, physical models (GloFAS: Global Flood Awareness System and GRFR: Global Reach-Level Flood Reanalysis), and hybrid post-processor models that correct physical-model outputs using LSTM or transformer post-processing. Models are trained and evaluated on 15 river basins in Nepal and are compared systematically.

## Repository structure
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

## Notes
- Preprocessing is already completed. Final parquet datasets for all 15 basins are in `input/`. Don't inspect or modify anything in the `preprocessing/` folder.
- Run scripts strictly step-by-step starting from `01_run_lstm.py`. Do not proceed to the next script until the current one runs successfully and is verified.
- Build and debug incrementally, ensuring each stage works before moving forward.
- Write clean, minimal, and maintainable code:clear structure, helpful comments, proper docstrings, and readability for a single researcher to maintain easily.
- Donot make any assumptions, ask explicitly and loudly if anything is unclear.
