"""Select sensor forecast settings on engines outside the benchmark unit."""
from collections.abc import Callable

import numpy as np
import pandas as pd

from pdm.forecasting.benchmark_data import split_history


def tune_local(forecast: Callable, frame: pd.DataFrame, unit: int, history: pd.DataFrame,
               cycles: pd.Index, targets: list[str], config: dict) -> tuple[pd.DataFrame, dict]:
    """Select per-sensor configurations by pooled validation RMSE and refit on the prefix."""
    candidates = config["candidates"]
    units = config["validation_units"]
    if not candidates or not units or unit in units or len(set(units)) != len(units):
        raise ValueError("Provide candidates and distinct non-benchmark validation engines")
    splits = [split_history(frame, engine, targets, len(cycles)) for engine in units]
    predictions, diagnostics = {}, {}
    for target in targets:
        trials = []
        for candidate in candidates:
            try:
                errors = []
                for prefix, suffix in splits:
                    prediction, _ = forecast(prefix[target], suffix.index, candidate)
                    errors.extend((suffix[target].to_numpy() - prediction) ** 2)
                rmse = float(np.sqrt(np.mean(errors)))
                if not np.isfinite(rmse):
                    raise ValueError("Nonfinite validation error")
                trials.append({"config": candidate, "rmse": rmse})
            except (ValueError, RuntimeError, ArithmeticError) as error:
                trials.append({"config": candidate, "error": str(error)})
        successful = [trial for trial in trials if "rmse" in trial]
        if not successful:
            raise ValueError(f"No successful validation candidates for {target}: {trials}")
        chosen = min(successful, key=lambda trial: trial["rmse"])
        predictions[target], details = forecast(history[target], cycles, chosen["config"])
        diagnostics[target] = {"selected": chosen, "trials": trials, **details}
    return pd.DataFrame(predictions, index=cycles), diagnostics
