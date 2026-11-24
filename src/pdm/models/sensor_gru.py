"""Direct multi-horizon sensor forecasts using GRU."""
import numpy as np
from torch import nn

from pdm.models.sensor_recurrent import RecurrentFit, fit_recurrent


def fit(x: np.ndarray, y: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
        config: dict, seed: int) -> RecurrentFit:
    """Train a GRU forecaster."""
    return fit_recurrent(nn.GRU, x, y, x_val, y_val, config, seed)
