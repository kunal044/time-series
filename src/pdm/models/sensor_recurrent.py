"""Direct recurrent forecasting with engine-disjoint validation."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


class RecurrentForecaster(nn.Module):
    """Encode a past-only sequence and emit the full forecast horizon."""

    def __init__(self, cell: type[nn.Module], inputs: int, hidden: int, horizon: int,
                 targets: int, dropout: float) -> None:
        super().__init__()
        self.encoder = cell(inputs, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, horizon * targets),
                                  nn.Unflatten(1, (horizon, targets)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use the final encoded observation for all lead times."""
        states, _ = self.encoder(x)
        return self.head(states[:, -1])


@dataclass
class RecurrentFit:
    """A recurrent model with its reconstruction parameters."""
    model: RecurrentForecaster
    architecture: dict

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict normalized deltas in bounded batches."""
        self.model.eval()
        with torch.inference_mode():
            batches = torch.from_numpy(x).split(512)
            return torch.cat([self.model(batch) for batch in batches]).numpy()

    def save(self, path: Path) -> None:
        """Save weights and reconstruction parameters."""
        torch.save({"state_dict": self.model.state_dict(), "architecture": self.architecture},
                   path.with_suffix(".pt"))


def fit_recurrent(cell: type[nn.Module], x: np.ndarray, y: np.ndarray, x_val: np.ndarray,
                  y_val: np.ndarray, config: dict, seed: int) -> RecurrentFit:
    """Train in CPU batches and restore the epoch with lowest validation loss."""
    torch.manual_seed(seed)
    torch.set_num_threads(config["threads"])
    architecture = {"inputs": x.shape[-1], "hidden": config["hidden"], "horizon": y.shape[1],
                    "targets": y.shape[2], "dropout": config["dropout"]}
    model = RecurrentForecaster(cell, **architecture)
    fitted = RecurrentFit(model, architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    train = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = torch.utils.data.DataLoader(train, batch_size=config["batch_size"], shuffle=True)
    best_loss, wait = np.inf, 0
    for _ in range(config["epochs"]):
        model.train()
        for features, expected in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(features), expected)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_loss = float(np.mean((fitted.predict(x_val) - y_val) ** 2))
        if val_loss < best_loss:
            best_loss, best_state, wait = val_loss, deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= config["patience"]:
                break
    if not np.isfinite(best_loss):
        raise ValueError("RNN training did not produce a finite validation loss")
    model.load_state_dict(best_state)
    return fitted
