"""Температурная калибровка вероятностей"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit


def safe_logit(probabilities) -> np.ndarray:
    return logit(np.clip(probabilities, 1e-7, 1 - 1e-7))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> dict:
    def brier(log_temperature: float) -> float:
        probabilities = expit(logits / np.exp(log_temperature))
        return float(np.mean((probabilities - labels) ** 2))

    result = minimize_scalar(brier, bounds=(-3., 3.), method='bounded', options={'xatol': 1e-10})
    return {'method': 'temperature', 'temperature': float(np.exp(getattr(result, 'x')))}


def calibrate(logits, spec: dict) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return expit(logits / spec['temperature'])
