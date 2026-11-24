"""Shared engine-disjoint training windows for supervised sensor forecasts."""
from pathlib import Path
from time import perf_counter
import logging

import numpy as np
import pandas as pd

from pdm.data.splits import split_units
from pdm.evaluation.sensor_metrics import score_sensor_forecast

LOGGER = logging.getLogger(__name__)
SUPERVISED = {"xgboost", "lstm", "gru"}


def windows(frame: pd.DataFrame, columns: list[str], targets: list[str],
            lookback: int, horizon: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Build direct forecasts within individual engines only."""
    inputs, outputs = [], []
    for _, engine in frame.groupby("unit"):
        engine = engine.sort_values("cycle")
        if not engine["cycle"].diff().dropna().eq(1).all():
            raise ValueError("Supervised windows require consecutive engine cycles")
        x = engine[columns].to_numpy()
        y = engine[targets].to_numpy()
        for end in range(lookback, len(engine) - horizon + 1, stride):
            inputs.append(x[end - lookback:end])
            outputs.append(y[end:end + horizon])
    if not inputs:
        raise ValueError("No complete training or validation windows")
    return np.asarray(inputs), np.asarray(outputs)


def run_supervised(model: str, frame: pd.DataFrame, unit: int, history: pd.DataFrame,
                   actual: pd.DataFrame, config: dict, directory: Path) -> tuple:
    """Select hyperparameters on other engines and score a three-seed ensemble."""
    if model == "xgboost":
        from pdm.models.sensor_xgboost import fit
    elif model == "lstm":
        from pdm.models.sensor_lstm import fit
    else:
        from pdm.models.sensor_gru import fit

    started = perf_counter()
    targets = list(actual.columns)
    columns = list(dict.fromkeys(targets + config["observed"]))
    lookback, horizon, stride = config["lookback"], len(actual), config["stride"]
    anchor_length = config["anchor_length"]
    if not 1 <= anchor_length <= lookback or len(history) < lookback or stride < 1:
        raise ValueError("Invalid lookback, stride or anchor length")
    counts = frame.groupby("unit").size()
    units = counts[(counts >= lookback + horizon) & (counts.index != unit)].index.to_numpy()
    train, val = split_units(units, config["val_fraction"], config["split_seed"])

    def build(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return windows(frame[frame.unit.isin(units[indices])], columns, targets, lookback, horizon, stride)

    x_train, y_train = build(train)
    x_val, y_val = build(val)
    x_holdout = history[columns].tail(lookback).to_numpy()[None]
    # Normalize with training-only per-feature variation. Predict relative to
    # each window's recent level, which is available at inference.
    scale = x_train.reshape(-1, len(columns)).std(axis=0)
    scale[scale < 1e-8] = 1
    target_scale = scale[:len(targets)]

    def normalize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        anchor = x[:, -anchor_length:].mean(axis=1, keepdims=True)
        return ((x - anchor) / scale).astype(np.float32), anchor[:, :, :len(targets)]

    x_train, train_anchor = normalize(x_train)
    x_val, val_anchor = normalize(x_val)
    x_holdout, holdout_anchor = normalize(x_holdout)
    y_train = ((y_train - train_anchor) / target_scale).astype(np.float32)
    y_val = ((y_val - val_anchor) / target_scale).astype(np.float32)
    seeds, candidates = config["seeds"], config["candidates"]
    if not seeds or len(set(seeds)) != len(seeds) or not candidates:
        raise ValueError("Specify distinct seeds and at least one candidate")
    trials = []
    best_rmse = np.inf
    for i, candidate in enumerate(candidates):
        fits, scores = [], []
        for seed in seeds:
            LOGGER.info("Training %s candidate %d seed %d", model, i, seed)
            fitted = fit(x_train, y_train, x_val, y_val, config | candidate, seed)
            error = y_val - fitted.predict(x_val)
            scores.append(float(np.sqrt(np.mean(error ** 2))))
            fits.append(fitted)
        rmse = float(np.mean(scores))
        if not np.isfinite(rmse):
            raise ValueError("Nonfinite validation RMSE")
        trials.append({"config": candidate, "seed_rmse": scores, "mean_normalized_rmse": rmse})
        if rmse < best_rmse:
            selected, winners, best_rmse = i, fits, rmse
    training = perf_counter() - started
    started = perf_counter()
    forecasts = [pd.DataFrame((fitted.predict(x_holdout) * target_scale + holdout_anchor)[0],
                              index=actual.index, columns=targets) for fitted in winners]
    inference = perf_counter() - started
    for seed, fitted in zip(seeds, winners):
        fitted.save(directory / f"{model}-{seed}")
    diagnostics = {
        "selected": selected, "trials": trials, "seeds": seeds, "lookback": lookback,
        "anchor_length": anchor_length, "columns": columns, "targets": targets, "scale": scale.tolist(),
        "train_units": units[train].tolist(), "validation_units": units[val].tolist(),
        "training_windows": len(x_train), "validation_windows": len(x_val),
    }
    seed_scores = [{"seed": seed, **row} for seed, forecast in zip(seeds, forecasts)
                   for row in score_sensor_forecast(actual, forecast)]
    pd.DataFrame(seed_scores).to_csv(directory / "metrics" / f"{model}_seed_scores.csv", index=False)
    forecast = pd.DataFrame(np.mean(forecasts, axis=0), index=actual.index, columns=targets)
    return forecast, training, inference, diagnostics
