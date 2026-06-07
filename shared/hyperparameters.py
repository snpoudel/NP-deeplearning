"""Global hyperparameters for experiments."""

# global hyperparameters: random seed, train/val/test split dates, and evaluation metrics
RANDOM_SEED = 42
SPLIT_DATES = {
    "train": ("1980-01-01", "1989-12-31"),
    "val": ("1990-01-01", "2004-12-31"),
    "test": ("2005-01-01", "2014-12-31"),
}
EVAL_METRICS = ["nse", "kge", "rmse"]

LOSS_FUNCTION = "mse"
OPTIMIZER = "adam"

# model hyperparameters
HYPERPARAMS = {
    "lstm": {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "early_stopping_patience": 10,
    },
    "transformer": {
        "d_model": 64,
        "nhead": 4,
        "num_encoder_layers": 2,
        "dim_feedforward": 256,
        "learning_rate": 1e-4,
        "early_stopping_patience": 10,
    },
}
