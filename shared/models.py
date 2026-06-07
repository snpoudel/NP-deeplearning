"""Model architectures for NP-deeplearning.

Place LSTM, transformer, and post-processor model classes here.
"""

from typing import Any


class BaseModel:
    """Minimal base model placeholder."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def train(self):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
