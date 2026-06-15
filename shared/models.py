"""Model architectures for NP-deeplearning.

LSTMModel and TransformerModel share the same forward interface:
    input:  (batch, seq_len, input_size)
    output: (batch, 1)

Post-processor variants reuse the same classes but predict residual errors
rather than streamflow directly.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from shared.hyperparameters import HYPERPARAMS, get_hyperparams


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

class LSTMModel(nn.Module):
    """Single-output LSTM for sequence-to-one regression.

    Args:
        input_size: Number of input features per timestep.
        hidden_size: Number of LSTM hidden units.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability applied between LSTM layers (not after last).
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Predictions of shape (batch, 1).
        """
        out, _ = self.lstm(x)                    # (batch, seq_len, hidden_size)
        return self.fc(self.dropout(out[:, -1, :]))  # (batch, 1) — last timestep only


def build_lstm_model(input_size: int, mode: str = "production") -> LSTMModel:
    """Construct an LSTMModel using the hyperparameters for the given mode."""
    hp = get_hyperparams(mode)["lstm"]
    return LSTMModel(
        input_size=input_size,
        hidden_size=hp["hidden_size"],
        num_layers=hp["num_layers"],
        dropout=hp["dropout"],
    )


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding added to the input embeddings."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Encoder-only Transformer for sequence-to-one regression.

    Architecture:
        input_proj  : Linear(input_size → d_model)
        pos_encoding: sinusoidal positional encoding + dropout
        encoder     : num_encoder_layers × TransformerEncoderLayer
        fc          : Linear(d_model → 1) applied to the last timestep

    Args:
        input_size: Number of input features per timestep.
        d_model: Internal embedding dimension.
        nhead: Number of attention heads (must divide d_model).
        num_encoder_layers: Number of stacked encoder layers.
        dim_feedforward: Hidden dimension of the FFN inside each encoder layer.
        dropout: Dropout probability.
    """

    def __init__(self, input_size: int, d_model: int, nhead: int,
                 num_encoder_layers: int, dim_feedforward: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Predictions of shape (batch, 1).
        """
        x = self.input_proj(x)        # (batch, seq_len, d_model)
        x = self.input_norm(x)        # normalize before positional encoding
        x = self.pos_encoding(x)      # (batch, seq_len, d_model)
        x = self.encoder(x)           # (batch, seq_len, d_model)
        return self.fc(x[:, -1, :])   # (batch, 1) — last timestep only


def build_transformer_model(input_size: int, mode: str = "production") -> TransformerModel:
    """Construct a TransformerModel using the hyperparameters for the given mode."""
    hp = get_hyperparams(mode)["transformer"]
    return TransformerModel(
        input_size=input_size,
        d_model=hp["d_model"],
        nhead=hp["nhead"],
        num_encoder_layers=hp["num_encoder_layers"],
        dim_feedforward=hp["dim_feedforward"],
        dropout=hp["dropout"],
    )


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def nse(qobs: np.ndarray, qsim: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency. Perfect = 1.0, benchmark = 0.0."""
    denominator = np.sum((qobs - qobs.mean()) ** 2)
    if denominator == 0:
        return float("nan")
    return float(1 - np.sum((qobs - qsim) ** 2) / denominator)


def kge(qobs: np.ndarray, qsim: np.ndarray) -> float:
    """Kling-Gupta Efficiency. Perfect = 1.0."""
    r = np.corrcoef(qobs, qsim)[0, 1]
    alpha = qsim.std() / qobs.std() if qobs.std() > 0 else float("nan")
    beta = qsim.mean() / qobs.mean() if qobs.mean() > 0 else float("nan")
    return float(1 - math.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def rmse(qobs: np.ndarray, qsim: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((qobs - qsim) ** 2)))


def pbias(qobs: np.ndarray, qsim: np.ndarray) -> float:
    """Percentage Bias. Perfect = 0. Positive = overestimate, negative = underestimate."""
    return float(100 * np.sum(qsim - qobs) / np.sum(qobs))


def pbias_high(qobs: np.ndarray, qsim: np.ndarray) -> float:
    """PBIAS restricted to high flows (top 10% by observed magnitude)."""
    mask = qobs >= np.percentile(qobs, 90)
    return pbias(qobs[mask], qsim[mask])


def pbias_low(qobs: np.ndarray, qsim: np.ndarray) -> float:
    """PBIAS restricted to low flows (bottom 30% by observed magnitude)."""
    mask = qobs <= np.percentile(qobs, 30)
    return pbias(qobs[mask], qsim[mask])


def pbias_mid(qobs: np.ndarray, qsim: np.ndarray) -> float:
    """PBIAS restricted to medium flows (30th–90th percentile by observed magnitude)."""
    lo, hi = np.percentile(qobs, 30), np.percentile(qobs, 90)
    mask = (qobs > lo) & (qobs < hi)
    return pbias(qobs[mask], qsim[mask])
