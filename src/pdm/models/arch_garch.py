"""Autoregressive sensor means with ARCH or GARCH forecast variance."""
import numpy as np
import pandas as pd


def forecast(history: pd.Series, cycles: pd.Index, config: dict) -> tuple[np.ndarray, dict]:
    """Fit on observed levels and return mean and predictive standard deviation."""
    from arch import arch_model

    window = config.get("trend_window")
    trend = np.zeros(len(cycles))
    if window:
        history = history.tail(window)
        origin = history.index[-1]
        basis = history.index.to_numpy() - origin
        slope, intercept = np.polyfit(basis, history.to_numpy(), 1)
        history = history - (slope * basis + intercept)
        trend = slope * (cycles.to_numpy() - origin) + intercept
    center, scale = float(history.mean()), float(history.std())
    if scale <= 0:
        raise ValueError("ARCH models require a varying sensor")
    lags = config["lags"]
    fit = arch_model((history - center) / scale, mean="AR" if lags else "Constant",
                    lags=lags, p=config["p"], q=config["q"], dist="normal",
                    rescale=False).fit(disp="off", show_warning=False)
    if fit.convergence_flag:
        raise ValueError(f"ARCH optimizer failed with status {fit.convergence_flag}")
    result = fit.forecast(horizon=len(cycles), reindex=False)
    mean = trend + center + scale * result.mean.iloc[-1].to_numpy()
    std = scale * np.sqrt(result.variance.iloc[-1].to_numpy())
    if not np.isfinite(std).all():
        raise ValueError("ARCH returned invalid predictive variance")
    return mean, {"forecast_std": std.tolist(), "parameters": fit.params.to_dict()}
