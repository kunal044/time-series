# Experiment log
| Date | Dataset/subset | Model | Seed | RMSE | NASA score | Notes |
|---|---|---|---|---|---|---|

| 2026-10-27 | C-MAPSS/FD001 | LightGBM validation (terminal only; superseded) | 42 | 1.7841 | 3.3081 | MAE 1.4784; window 30; engine holdout |
| 2026-10-27 | C-MAPSS/FD001 | LightGBM validation (terminal only; superseded) | 43 | 1.6877 | 3.2042 | MAE 1.4484; window 30; engine holdout |
| 2026-10-27 | C-MAPSS/FD001 | LightGBM validation (terminal only; superseded) | 44 | 1.5933 | 2.8209 | MAE 1.2727; window 30; engine holdout |
| 2026-10-27 | C-MAPSS/FD001 | LightGBM official test (clipped at 125; superseded) | 42,43,44 | 12.7123 | 246.3574 | MAE 9.5341; terminal window per engine |

| 2026-10-28 | C-MAPSS/FD001 | LightGBM validation all windows | 42 | 12.1780 | 9325.1010 | MAE 8.7704; engine holdout; window 30 |
| 2026-10-28 | C-MAPSS/FD001 | LightGBM validation terminal per engine | 42 | 1.7841 | 3.3081 | MAE 1.4784; engine holdout; window 30 |
| 2026-10-28 | C-MAPSS/FD001 | LightGBM validation all windows | 43 | 12.1052 | 9286.1758 | MAE 8.6502; engine holdout; window 30 |
| 2026-10-28 | C-MAPSS/FD001 | LightGBM validation terminal per engine | 43 | 1.6877 | 3.2042 | MAE 1.4484; engine holdout; window 30 |
| 2026-10-28 | C-MAPSS/FD001 | LightGBM validation all windows | 44 | 12.0107 | 9454.2132 | MAE 8.5709; engine holdout; window 30 |
| 2026-10-28 | C-MAPSS/FD001 | LightGBM validation terminal per engine | 44 | 1.5933 | 2.8209 | MAE 1.2727; engine holdout; window 30 |
| 2026-10-28 | C-MAPSS/FD001 | LightGBM official test raw RUL | ensemble | 14.0138 | 279.4752 | MAE 10.6041; terminal window per engine |
| 2026-10-28 | C-MAPSS/FD001 | LightGBM official test RUL clipped at 125 | ensemble | 12.7123 | 246.3574 | MAE 9.5341; terminal window per engine |

2026-11-22 repository verification. Ran the existing FD001 engine 1, sensor 2
stationarity analysis from a separate working directory and verified that it
wrote its CSV and plot to an isolated temporary run directory.

## 2026-11-23 sensor benchmark

FD001 engine 1, cycles 1–153 observed and 154–192 scored. Sensor RMSE is in
native units. NASA score applies to the separate RUL task.

| Model | Sensor 2 RMSE | Sensor 3 RMSE |
| --- | ---: | ---: |
| ARMA | 0.685414 | |
| ARIMAX | 0.457159 | |
| SARIMAX | 0.460752 | |
| VAR | 0.514314 | 5.746134 |
| VARMA | 0.667744 | 7.668164 |
| VARIMA | 0.548278 | 6.416757 |
| VARMAX | 0.496187 | 7.248371 |
| Holt-Winters | 0.543930 | |
| TFT CPU smoke, seed 42 | 0.875687 | 9.607451 |

The combined run is in `results/runs/sensor-skh5xq1s` with six classical
baselines and all nine additions. Full TFT training was deferred. The
training-prefix ACF of first differences at lag 10 was −0.114 against the
±0.159 screening bound. For both targets, the VARIMA ADF p-values were
above 0.05 in levels and below 0.05 after first differencing.

Verification runs `sensor-20u7rjg7`, `sensor-shbg0afe`, and
`sensor-l9588gcg` exposed a NumPy integer serialization error, missing
optional TFT dependencies, and encoding of unseen TFT engine groups,
respectively. After those fixes, the statistical-only run
`sensor-dbcylcov` and TFT CPU smoke run `sensor-a18p6pw7` completed.

## Tuned TFT, ARCH, GARCH, and Prophet on FD001

Reproduce the comparison from the repository root after installing the base
requirements and the forecasting extras.

```powershell
python -m pip install -r requirements.txt
python -m pip install -e ".[tft,sensor-forecast]"
python -m pdm.forecasting.benchmark --models persistence arch garch prophet tft --full-tft --config tft=configs/models/tft_benchmark_cpu.yaml
```

Standalone entry points are `arch_model.py`, `garch_model.py`,
`prophet_model.py`, and `tft_model.py --full-tft`. Without the CPU budget
override, TFT uses the longer training schedule in `configs/models/tft.yaml`.
Local candidate grids are in their respective model YAML files.

The comparison forecasts sensors 2 and 3 from the observed prefix of FD001
engine 1 through cycle 153. All approaches score the same 39 unseen cycles
(154-192). ARCH, GARCH and Prophet select settings by pooled validation RMSE
from separate engines 2-8, fitting each validation engine's observed prefix
and scoring its final 39 cycles. Each selected local configuration then fits
the benchmark engine's observed prefix. TFT selects its architecture and
learning rate by mean validation quantile loss across seeds 42, 43 and 44,
using disjoint training and validation engines, with engine 1 excluded from
both. Preprocessing is fitted on the corresponding training histories.
Persistence repeats the last observed sensor reading.

The CPU experiment caps each TFT candidate at six epochs and 25 training
batches per epoch. The TFT comparison row scores the mean prediction across
three seeds. Individual seed metrics and tuning losses are exported alongside
the comparison. Deterministic local forecasts each produce one score per
sensor. The NASA asymmetric score is used for RUL errors in cycles. This
experiment scores sensor-level prediction errors in the original scale.

### Results on 25 November 2026

Scores below are for the same held-out cycles. RMSE and MAE use native sensor
units. MAPE uses percent.

| Model | Sensor 2 RMSE | Sensor 2 MAE | Sensor 2 MAPE | Sensor 3 RMSE | Sensor 3 MAE | Sensor 3 MAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence | 0.506 | 0.393 | 0.0611% | 8.734 | 7.397 | 0.4625% |
| ARCH | 0.428 | 0.332 | 0.0516% | 5.169 | 4.292 | 0.2684% |
| GARCH | 0.428 | 0.332 | 0.0516% | 5.167 | 4.291 | 0.2684% |
| Prophet | **0.399** | **0.307** | **0.0478%** | 5.087 | 4.239 | 0.2651% |
| TFT | 0.424 | 0.330 | 0.0512% | **4.524** | **3.936** | **0.2464%** |

- ARCH and GARCH selected a 60-cycle linear trend and a constant residual
  mean for both sensors. In the preliminary levels-only run
  `sensor-r6mnu220`, both models scored about 0.882 and 9.96 RMSE. Adding the
  trend reduced RMSE by about 52% and 48%, respectively. ARCH and GARCH
  produced nearly identical point forecasts after tuning.
- Prophet selected 10 changepoints with prior scale 0.20 for sensor 2, and
  5 changepoints with prior scale 0.05 for sensor 3. The original TFT
  architecture won validation with mean quantile loss 2.119 versus 2.372 for
  the smaller candidate. Selected TFT seed RMSE ranged from 0.385 to 0.479
  on sensor 2 and from 4.297 to 4.857 on sensor 3.
- The combined result is in [sensor-de8w21cf](../results/runs/sensor-de8w21cf).
  The initial invocation finished TFT and persistence while the other
  dependencies were being installed. A completed supplemental run,
  `sensor-dok4_689`, supplied the local model results. The manifests and
  held-out actuals were checked for exact agreement before combining reports.
  The run includes per-model predictions, trial records, seed scores,
  comparisons, residual plots and TFT checkpoints.

### Maintenance interpretation

- Use Prophet as the candidate trend forecast for the low-pressure compressor
  outlet temperature (T24, sensor 2). It reduced RMSE 21% versus persistence.
  Its MAE was 0.207 in the first 13 forecast cycles and 0.436 in the last 13.
  Test shorter refresh intervals before using that horizon for trend alerts.
  [C-MAPSS sensor mapping](https://www.mdpi.com/2227-9717/10/5/1012)
- Use TFT as the candidate trend forecast for the high-pressure compressor
  outlet temperature (T30, sensor 3). It reduced RMSE 48% versus persistence.
  Its final-cycle prediction was about 4.03 native units below the observed
  temperature. Check sustained positive residuals alongside related telemetry
  when triaging possible degradation. Calibrate alert thresholds with multiple
  engines, operating regimes and forecast origins before field use.
- Evaluate ARCH and GARCH for changing residual variability. Approximate
  Gaussian bands using forecast mean plus or minus 1.96 predictive standard
  deviations covered 85% of sensor 2 observations for both models. For sensor
  3 they covered 90% with ARCH and 72% with GARCH on this window. Calibrate
  coverage across engines before using these bands to trigger alerts.
  [ARCH forecast outputs](https://arch.readthedocs.io/en/latest/univariate/forecasting.html)
- Absolute sensor levels make MAPE small even when the model misses a
  meaningful change in temperature. Use native-unit errors, bias by lead
  time and alert performance for maintenance decisions. This experiment covers
  one simulated engine, two sensors and one forecast origin.

## XGBoost, LSTM, and GRU on FD001

The three supervised models use 48-cycle sensor histories to forecast the
next 39 cycles directly. Each forecast is a change relative to the final
eight-cycle mean. Feature scales are learned from training engines only.
Training and validation use disjoint engines, and engine 1 is excluded from
both. Candidate selection uses mean normalized validation RMSE across seeds
42, 43 and 44. Final scores use the mean forecast across those seeds.

XGBoost learns one booster per sensor from lag summaries and forecast lead.
The LSTM and GRU encode the observed sensor sequence and predict the entire
horizon jointly. We tuned the tree depth and learning rate for XGBoost, and
the recurrent width, dropout and learning rate for LSTM and GRU. Early
stopping uses validation data for all three models. The 39 scored cycles
are the same as the previous benchmark, cycles 154-192 of engine 1.

```powershell
python -m pip install -e ".[xgboost]"
python -m pdm.forecasting.benchmark --models xgboost lstm gru --reference-run results/runs/sensor-de8w21cf
```

The comparison verifies the reference manifest and the actual values at
every cycle before rescoring the saved prior forecasts. Standalone
launchers are `xgboost_model.py`, `lstm_model.py`, and `gru_model.py`.

### Results on 25 November 2026

| Model | Sensor 2 RMSE | Sensor 2 MAE | Sensor 2 MAPE | Sensor 3 RMSE | Sensor 3 MAE | Sensor 3 MAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost | **0.336** | **0.261** | **0.0406%** | 4.335 | 3.797 | 0.2378% |
| LSTM | 0.366 | 0.280 | 0.0435% | 4.340 | 3.764 | 0.2357% |
| GRU | 0.342 | 0.267 | 0.0415% | **4.296** | **3.750** | **0.2348%** |

- XGBoost selected depth 5 and learning rate 0.04. Mean normalized validation
  RMSE moved from 0.83466 to 0.83378, a 0.11% gain in this grid. Both recurrent
  models selected 24 hidden units, 0.1 dropout and learning rate 0.003.
  Their validation RMSE values were 0.85754 for LSTM and 0.83354 for GRU.
- XGBoost reduced sensor 2 RMSE 16% versus the previous Prophet winner.
  GRU reduced sensor 3 RMSE 5% versus the previous TFT winner. All three
  supervised models improved on both previous winning scores.
- For low-pressure compressor outlet temperature, XGBoost had near-zero mean
  bias over this window. Its MAE rose from 0.218 in the first 13 cycles to
  0.280 in the last 13. Test shorter refresh intervals before selecting an
  alert horizon.
- For high-pressure compressor outlet temperature, GRU had the lowest
  overall errors. Actuals averaged 0.431 native units above its forecast.
  Evaluate persistent forecast residuals alongside related sensors before
  using them to triage possible degradation. Validate across engines and
  origins before setting maintenance thresholds.
- The [combined run](../results/runs/sensor-myzytdtd) contains aligned
  forecasts, plots, per-seed scores, tuning diagnostics and saved models.
  Supervised training finished first, then the prior forecasts were imported,
  verified against the actuals and rescored against the same 39 cycles.
  Verification completed with 68 pytest tests passing.
