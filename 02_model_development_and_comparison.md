# Part 2 — ML Model Development & Comparison

## Scope
This document covers only the classification/regression model experiments completed so far.

## Baseline models

Baseline classification models:

```text
Logistic Regression
Random Forest Classifier
```

Baseline regression models:

```text
Linear Regression
Random Forest Regressor
```

Relevant code:

```text
src/models/baseline/
src/models/core/model_factory.py
src/models/baseline/train_baseline.py
```

The baselines establish a reference before class balancing and advanced modeling.

## Baseline classification results

### Defect

```text
Logistic Regression:
Accuracy  66.90%
Precision 82.35%
Recall     1.54%
F1         3.03%

Random Forest:
Accuracy  65.50%
Precision 44.05%
Recall    11.01%
F1        17.62%
```

### Porosity

```text
Logistic Regression:
Accuracy  64.43%
Precision 60.54%
Recall     8.94%
F1        15.59%

Random Forest:
Accuracy  64.06%
Precision 52.86%
Recall    19.50%
F1        28.49%
```

### Scrap

```text
Logistic Regression:
Accuracy  77.60%
Precision 0%
Recall    0%
F1        0%

Random Forest:
Accuracy  77.53%
Precision 40.00%
Recall     0.66%
F1         1.30%
```

These results demonstrate why accuracy alone is misleading for this problem.

## Balanced classification experiment

Balanced Logistic Regression was evaluated with 5-fold cross-validation.

Results:

| Target | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Defect | 55.6% | 38.7% | 55.8% | 45.7% |
| Porosity | 57.1% | 43.6% | 57.7% | 49.7% |
| Scrap | 53.1% | 24.4% | 52.0% | 33.2% |

This experiment improved positive-event detection substantially compared with the default baseline behavior.

## Advanced model: XGBoost

Current advanced implementation:

```text
src/models/advanced/xgboost_classification.py
```

Configuration currently used for CV:

```python
n_estimators = 300
max_depth = 4
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
objective = "binary:logistic"
eval_metric = "logloss"
random_state = 42
```

Class imbalance is handled through `scale_pos_weight`.

5-fold CV results:

| Target | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Defect | 58.4% | 40.2% | 49.5% | 44.4% |
| Porosity | 60.0% | 46.0% | 51.0% | 48.4% |
| Scrap | 59.3% | 24.4% | 38.8% | 30.0% |

XGBoost increased accuracy, but its recall/F1 trade-off means it cannot simply be declared "best" from accuracy alone.

## Yield regression

Baseline results:

```text
Linear Regression:
MAE  = 5.1960
RMSE = 6.1754
R²   = 0.0493

Random Forest:
MAE  = 5.0923
RMSE = 6.0754
R²   = 0.0798
```

The current Yield signal is weak, so regression results should be presented as a baseline/reference rather than as a highly accurate yield predictor.

## Important engineering lesson

The model-selection question is not:

> Which model has the highest accuracy?

The current project asks:

> Which model/threshold combination provides a useful engineering screening signal under an explicit error/alert policy?

This leads to the decision-policy stage in Part 3.

## Useful commands

```bash
python -m pytest src/models -v
```

```bash
python -m pytest src/models/baseline -v
```
