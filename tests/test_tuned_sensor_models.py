"""Regression tests for tuned fixed-origin forecasting."""
import numpy as np
import pandas as pd
import pytest

from pdm.forecasting.local_tuning import tune_local
from pdm.models import arch_garch, prophet


def test_tuning_uses_disjoint_engines_and_refits_on_observed_prefix():
    frame = pd.DataFrame([(unit, cycle, float(unit)) for unit in [1, 2, 3]
                          for cycle in range(1, 26)], columns=["unit", "cycle", "sensor_2"])
    history = frame.loc[frame.unit == 1].iloc[:20].set_index("cycle")
    cycles = pd.RangeIndex(21, 26)
    calls = []

    def model(series, index, config):
        calls.append((series.iloc[0], series.index.max()))
        return np.full(len(index), config["level"]), {}

    config = {"validation_units": [2, 3], "candidates": [{"level": 0}, {"level": 2.5}]}
    prediction, details = tune_local(model, frame, 1, history, cycles, ["sensor_2"], config)
    assert calls == [(2, 20), (3, 20), (2, 20), (3, 20), (1, 20)]
    assert prediction.index.equals(cycles)
    assert prediction.sensor_2.eq(2.5).all()
    assert details["sensor_2"]["selected"]["rmse"] == 0.5
    with pytest.raises(ValueError, match="non-benchmark"):
        tune_local(model, frame, 1, history, cycles, ["sensor_2"], config | {"validation_units": [1]})


@pytest.mark.parametrize("q", [0, 1])
@pytest.mark.parametrize("trend_window", [None, 120])
def test_arch_family_mean_and_variance(q, trend_window):
    pytest.importorskip("arch")
    history = pd.Series(np.random.default_rng(7).normal(600, 0.5, 200),
                        index=pd.RangeIndex(1, 201))
    if trend_window:
        history += 0.1 * history.index.to_numpy()
    mean, details = arch_garch.forecast(
        history, pd.RangeIndex(201, 206), {"p": 1, "q": q, "lags": 1, "trend_window": trend_window})
    assert mean.shape == (5,)
    assert np.allclose(mean, 620 if trend_window else 600, atol=1)
    assert np.all(np.asarray(details["forecast_std"]) > 0)


def test_prophet_extrapolates_cycle_trend():
    pytest.importorskip("prophet")
    history = pd.Series(np.arange(1.0, 31.0), index=pd.RangeIndex(1, 31))
    prediction, _ = prophet.forecast(history, pd.RangeIndex(31, 36), {"n_changepoints": 0})
    np.testing.assert_allclose(prediction, np.arange(31.0, 36.0), atol=0.05)


def test_tft_candidate_selection_precedes_holdout_prediction(tmp_path, monkeypatch):
    import json
    from pdm.forecasting.benchmark import _run_tft
    from pdm.models import tft

    calls = []

    class Fit:
        def predict(self, history, cycles):
            calls.append("predict")
            return pd.DataFrame({"sensor_2": [2.0, 2.0]}, index=cycles)

    def fake_fit(frame, unit, targets, horizon, config, seed, directory, resume):
        calls.append((config["width"], seed))
        directory.mkdir()
        (directory / "best_validation.json").write_text(json.dumps(10 - config["width"]))
        return Fit()

    monkeypatch.setattr(tft, "fit_tft", fake_fit)
    (tmp_path / "metrics").mkdir()
    actual = pd.DataFrame({"sensor_2": [1.0, 2.0]})
    config = {"width": 1, "seeds": [42, 43, 44], "candidates": [{}, {"width": 2}]}
    prediction, _, _ = _run_tft(actual, 1, ["sensor_2"], actual, actual, config, tmp_path, None)
    assert calls == [(width, seed) for width in [1, 2] for seed in config["seeds"]] + ["predict"] * 3
    assert prediction.sensor_2.eq(2.0).all()
    tuning = json.loads((tmp_path / "metrics" / "tft_tuning.json").read_text())
    assert tuning["selected"] == 1
