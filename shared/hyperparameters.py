"""Global hyperparameters for experiments."""

# global hyperparameters: random seed, train/val/test split dates, and evaluation metrics
RANDOM_SEED = 42

# Seeds for multi-seed training (extend to e.g. [42, 123, 456] for production runs)
SEEDS = [42]

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
        "hidden_size": 8,       # dev size; use 128 for production
        "num_layers": 1,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "early_stopping_patience": 10,
        "seq_len": 3,            # dev size; use 365 for production
        "batch_size": 4,         # dev size; use 256 for production
        "num_epochs": 5,         # dev size; use 100 for production
    },
    "transformer": {
        "d_model": 8,               # dev size; use 64 for production
        "nhead": 1, # this most divides d_model; use 8 for production
        "num_encoder_layers": 2,
        "dim_feedforward": 32,       # dev size; use 256 for production
        "dropout": 0.1,
        "learning_rate": 1e-4,
        "early_stopping_patience": 10,
        "seq_len": 3,                # dev size; use 365 for production
        "batch_size": 4,             # dev size; use 256 for production
        "num_epochs": 5,             # dev size; use 100 for production
    },
}
