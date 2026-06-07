"""Global hyperparameters for experiments."""

HYPERPARAMS = {
    "lstm": {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.1,
        "learning_rate": 1e-3,
    },
    "transformer": {
        "d_model": 64,
        "nhead": 4,
        "num_encoder_layers": 2,
        "dim_feedforward": 256,
        "learning_rate": 1e-4,
    },
}
