"""Prophet trend extrapolation on an evenly spaced engine-cycle axis."""
import numpy as np
import pandas as pd


def forecast(history: pd.Series, cycles: pd.Index, config: dict) -> tuple[np.ndarray, dict]:
    """Fit a piecewise linear trend with calendar effects disabled."""
    from prophet import Prophet

    training = pd.DataFrame({"ds": pd.to_datetime(history.index, unit="D", origin="unix"),
                             "y": history.to_numpy()})
    model = Prophet(daily_seasonality=False, weekly_seasonality=False,
                    yearly_seasonality=False, uncertainty_samples=0, **config)
    model.fit(training)
    future = pd.DataFrame({"ds": pd.to_datetime(cycles, unit="D", origin="unix")})
    return model.predict(future)["yhat"].to_numpy(), {}
