"""Global hyperparameters for experiments."""

# global hyperparameters: random seed, train/val/test split dates, and evaluation metrics
RANDOM_SEED = 42

# Dev uses a single seed for fast smoke-testing; production uses all three.
DEV_SEEDS  = [42]
PROD_SEEDS = [42, 123, 456]

# Backward-compatible alias (resolves to production seeds)
SEEDS = PROD_SEEDS

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

# model hyperparameters — two profiles selectable at runtime via --mode dev|production
DEV_HYPERPARAMS = {
    "lstm": {
        "hidden_size": 8,
        "num_layers": 1,
        "dropout": 0.4,
        "learning_rate": 1e-3,
        "lr_scheduler_patience": 3,
        "lr_scheduler_factor": 0.5,
        "early_stopping_patience": 2,
        "seq_len": 3,
        "batch_size": 256,
        "num_epochs": 5,
    },
    "transformer": {
        "d_model": 8,
        "nhead": 1,
        "num_encoder_layers": 1,
        "dim_feedforward": 32,
        "dropout": 0.2,
        "learning_rate": 1e-4,
        "lr_scheduler_patience": 3,
        "lr_scheduler_factor": 0.5,
        "early_stopping_patience": 2,
        "seq_len": 3,
        "batch_size": 256,
        "num_epochs": 5,
    },
}

PROD_HYPERPARAMS = {
    "lstm": {
        "hidden_size": 256,
        "num_layers": 1,
        "dropout": 0.4,
        "learning_rate": 1e-3,
        "lr_scheduler_patience": 3,
        "lr_scheduler_factor": 0.5,
        "early_stopping_patience": 10,
        "seq_len": 365,
        "batch_size": 128,
        "num_epochs": 100,
    },
    "transformer": {
        "d_model": 128,
        "nhead": 8,
        "num_encoder_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.2,
        "learning_rate": 1e-4,
        "lr_scheduler_patience": 3,
        "lr_scheduler_factor": 0.5,
        "early_stopping_patience": 10,
        "seq_len": 365,
        "batch_size": 128,
        "num_epochs": 100,
    },
}


# Backward-compatible alias — existing scripts import HYPERPARAMS directly
HYPERPARAMS = PROD_HYPERPARAMS


def get_hyperparams(mode: str = "dev") -> dict:
    """Return the hyperparameter dict for the given mode.

    Args:
        mode: "dev" (default) for fast smoke-test values, or "production" for full-run values.
    """
    if mode == "production":
        return PROD_HYPERPARAMS
    return DEV_HYPERPARAMS
