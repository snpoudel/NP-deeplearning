"""Global hyperparameters for experiments."""

# global hyperparameters: random seed, train/val/test split dates, and evaluation metrics
RANDOM_SEED = 42

# Seeds for multi-seed training (extend to e.g. [42, 123, 456] for production runs)
SEEDS = [42, 123, 456]

# Device selection: "auto" detects CUDA at runtime, or set "cpu" / "cuda" explicitly
DEVICE = "auto"

SPLIT_DATES = {
    "train": ("1990-01-01", "2004-12-31"),
    "val": ("1980-01-01", "1989-12-31"),
    "test": ("2005-01-01", "2014-12-31"),
}
EVAL_METRICS = ["nse", "kge", "rmse"]

LOSS_FUNCTION = "mse"
OPTIMIZER = "adam"

# model hyperparameters
HYPERPARAMS = {
    "lstm": {
        "hidden_size": 256,      # dev: 8; production: 256
        "num_layers": 1,
        "dropout": 0.4,
        "learning_rate": 1e-3,
        "lr_scheduler_patience": 3,   # epochs before LR reduction
        "lr_scheduler_factor": 0.5,   # LR reduction factor
        "early_stopping_patience": 10,  # dev: 2; production: 10
        "seq_len": 365,          # dev: 3; production: 365
        "batch_size": 128,       # dev: 256; production: 128
        "num_epochs": 100,       # dev: 5; production: 100
    },
    "transformer": {
        "d_model": 128,              # dev: 8; production: 128
        "nhead": 8,                  # dev: 1; production: 8
        "num_encoder_layers": 2,     # dev: 1; production: 2
        "dim_feedforward": 256,      # dev: 32; production: 256
        "dropout": 0.2,
        "learning_rate": 1e-4,
        "lr_scheduler_patience": 3,  # epochs before LR reduction
        "lr_scheduler_factor": 0.5,  # LR reduction factor
        "early_stopping_patience": 10,  # dev: 2; production: 10
        "seq_len": 365,              # dev: 3; production: 365
        "batch_size": 128,           # dev: 256; production: 128
        "num_epochs": 100,           # dev: 5; production: 100
    },
}
