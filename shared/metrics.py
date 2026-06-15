"""Evaluation metrics for streamflow prediction (pure numpy, no torch dependency)."""

import math

import numpy as np


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
