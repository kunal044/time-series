"""Lead-conditioned XGBoost regression on lagged sensor history."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def features(x: np.ndarray, horizon: int) -> np.ndarray:
    """Summarize recent history and append the known forecast lead."""
    offsets = np.array_split(x, 4, axis=1)
    summary = np.concatenate([x[:, -1], *[part.mean(axis=1) for part in offsets],
                              x.std(axis=1)], axis=1)
    leads = np.tile(np.arange(1, horizon + 1, dtype=np.float32) / horizon, len(x))
    return np.column_stack([np.repeat(summary, horizon, axis=0), leads])


@dataclass
class XGBoostFit:
    """Fitted regressors in sensor order."""
    models: list
    horizon: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return sample by lead by target predictions."""
        design = features(x, self.horizon)
        pred = np.column_stack([model.predict(design) for model in self.models])
        return pred.reshape(len(x), self.horizon, len(self.models))

    def save(self, path: Path) -> None:
        """Save each booster in the portable XGBoost format."""
        path.mkdir()
        for i, model in enumerate(self.models):
            model.save_model(path / f"target-{i}.json")


def fit(x: np.ndarray, y: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
        config: dict, seed: int) -> XGBoostFit:
    """Train one lead-conditioned booster per target with early stopping."""
    from xgboost import XGBRegressor

    horizon = y.shape[1]
    if x.shape[1] < 4:
        raise ValueError("XGBoost requires at least four lag observations")
    train, val = features(x, horizon), features(x_val, horizon)
    models = []
    for i in range(y.shape[2]):
        model = XGBRegressor(tree_method="hist", objective="reg:squarederror",
                             random_state=seed, n_jobs=config["threads"], **config["booster"])
        model.fit(train, y[:, :, i].ravel(), eval_set=[(val, y_val[:, :, i].ravel())], verbose=False)
        models.append(model)
    return XGBoostFit(models, horizon)
