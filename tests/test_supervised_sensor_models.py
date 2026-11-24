"""Exercise supervised forecasting, alignment and engine boundaries."""
import json

import numpy as np
import pandas as pd
import pytest

from pdm.forecasting.benchmark_reports import reference_scores
from pdm.forecasting.supervised import run_supervised, windows
from pdm.models import sensor_gru, sensor_lstm, sensor_xgboost


def test_windows_never_cross_engine_boundaries():
    frame = pd.DataFrame([(unit, cycle, 100 * unit + cycle)
                          for unit in [1, 2] for cycle in range(1, 6)],
                         columns=["unit", "cycle", "sensor_2"])
    x, y = windows(frame, ["sensor_2"], ["sensor_2"], 2, 2, 1)
    np.testing.assert_array_equal(x[:, -1, 0], [102, 103, 202, 203])
    np.testing.assert_array_equal(y[:, 0, 0], [103, 104, 203, 204])


@pytest.mark.parametrize("adapter", [sensor_lstm, sensor_gru, sensor_xgboost])
def test_adapters_emit_direct_multi_target_forecasts(adapter, tmp_path):
    if adapter is sensor_xgboost:
        pytest.importorskip("xgboost")
    random = np.random.default_rng(5)
    x = random.normal(size=(18, 8, 3)).astype(np.float32)
    y = random.normal(size=(18, 4, 2)).astype(np.float32)
    config = {"threads": 1, "epochs": 2, "patience": 1, "batch_size": 8,
              "hidden": 8, "learning_rate": 0.003, "dropout": 0.1,
              "booster": {"n_estimators": 3, "max_depth": 2, "early_stopping_rounds": 2}}
    fitted = adapter.fit(x[:12], y[:12], x[12:16], y[12:16], config, 42)
    forecast = fitted.predict(x[16:])
    assert forecast.shape == (2, 4, 2)
    assert np.isfinite(forecast).all()
    fitted.save(tmp_path / "checkpoint")
    assert list(tmp_path.iterdir())


def test_holdout_does_not_enter_scaling_or_tuning(tmp_path, monkeypatch):
    frame = pd.DataFrame([(unit, cycle, 10.0 if unit == 1 else float(cycle))
                          for unit in range(1, 7) for cycle in range(1, 12)],
                         columns=["unit", "cycle", "sensor_2"])
    history = frame[frame.unit.eq(1)].set_index("cycle").iloc[:8]
    actual = pd.DataFrame({"sensor_2": [999., 999., 999.]}, index=pd.Index([9, 10, 11]))
    (tmp_path / "metrics").mkdir()

    class Zero:
        def predict(self, x):
            return np.zeros((len(x), 3, 1))

        def save(self, path):
            pass

    def fake_fit(x, y, x_val, y_val, config, seed):
        assert x.max() == x_val.max() == 0
        assert y.max() < 10 and y_val.max() < 10
        return Zero()

    monkeypatch.setattr(sensor_gru, "fit", fake_fit)
    config = {"lookback": 4, "anchor_length": 1, "stride": 2, "observed": [],
              "val_fraction": 0.2, "split_seed": 42, "seeds": [42, 43, 44], "candidates": [{}]}
    forecast, _, _, diagnostics = run_supervised("gru", frame, 1, history, actual, config, tmp_path)
    assert forecast["sensor_2"].eq(10).all()
    assert set(diagnostics["train_units"]).isdisjoint(diagnostics["validation_units"])
    assert 1 not in diagnostics["train_units"] + diagnostics["validation_units"]


def test_reference_forecasts_rescored_and_mismatched_actuals_rejected(tmp_path):
    reference, output = tmp_path / "reference", tmp_path / "output"
    for directory in [reference, output]:
        (directory / "metrics").mkdir(parents=True)
    manifest = {"subset": "FD001", "source": "test", "unit": 1, "history_last_cycle": 1,
                "test_cycles": [2, 3], "targets": ["sensor_2"], "models": ["gru"]}
    (reference / "manifest.json").write_text(json.dumps(manifest))
    pd.DataFrame([{"model": "persistence", "protocol": "local_engine_history",
                   "training_seconds": 0, "inference_seconds": 0}]
                 ).to_csv(reference / "metrics/model_comparison.csv", index=False)
    actual = pd.DataFrame({"sensor_2": [1., 2.]}, index=pd.Index([2, 3]))
    pd.concat({"actual": actual, "forecast": actual + 1}, axis=1).to_csv(
        reference / "metrics/persistence_forecast.csv")
    rows = reference_scores(reference, output, actual, manifest)
    assert rows[0]["rmse"] == 1
    with pytest.raises(ValueError, match="actuals differ"):
        reference_scores(reference, output, actual + 1, manifest)
