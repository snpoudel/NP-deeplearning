# Train and evaluate an LSTM (sequence-to-one) for streamflow prediction.
# * Use hyperparameters from `shared/hyperparameters.py`
# * Use model from `shared/models.py`
# * Use dataset/preprocessing from `shared/dataset.py`
# * DataLoader input: `(batch, seq_len, features)`, target: `(batch, 1)`

# Train with epoch loop:
# * forward pass → loss → backward pass → optimizer step
# * compute and print train/val loss each epoch
# * save best model by validation loss
# * store loss curves

# After training:
# * run inference with best model on complete dataset sequentially (all train/val/test sets)
# * save per-gauge parquet files to `output/predictions/lstm/`
# Output columns: `date, qobs, qsim`
