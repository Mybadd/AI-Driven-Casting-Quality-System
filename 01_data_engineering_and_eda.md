# Part 1 — Data Engineering, Schema & EDA

## Scope
This document covers only the data/EDA work completed so far. It does not describe the future frontend or unfinished backend work.

## Current dataset

Development dataset:

`data/raw/foundry_quality_dataset_13548.csv`

- Rows: **13,548**
- Columns: **10**
- Missing values: **0**
- Duplicate rows: **0**

Columns:

```text
Batch
Alloy
Pour_Temp
Mold_Moisture
Cooling_Time
Riser
Defect
Porosity
Scrap
Yield
```

## Approved ML inputs

The project currently uses exactly five raw ML inputs:

```python
INPUT_FEATURES = [
    "Alloy",
    "Pour_Temp",
    "Mold_Moisture",
    "Cooling_Time",
    "Riser",
]
```

`Batch` is an identifier and is not used as an ML feature.

Targets:

```python
CLASSIFICATION_TARGETS = [
    "Defect",
    "Porosity",
    "Scrap",
]

REGRESSION_TARGET = "Yield"
```

## Target distribution

```text
Defect:
No  = 9009
Yes = 4539

Porosity:
No  = 8574
Yes = 4974

Scrap:
No  = 10512
Yes = 3036
```

Positive-event rates are approximately:

```text
Defect   33.5%
Porosity 36.7%
Scrap    22.4%
```

This imbalance is important because accuracy alone can hide poor positive-event detection.

## Data loading

Current loader:

```text
src/data/load_data.py
```

The loader converts `Yield` values such as `60.90%` into numeric values.

Example:

```python
from pathlib import Path
from src.data.load_data import load_csv

df = load_csv(
    Path("data/raw/foundry_quality_dataset_13548.csv")
)

print(df.shape)
print(df.isna().sum().sum())
print(df.duplicated().sum())
```

## Preprocessing

Current preprocessing is in:

```text
src/features/preprocessing.py
```

Categorical feature:

```text
Alloy
```

Numerical features:

```text
Pour_Temp
Mold_Moisture
Cooling_Time
Riser
```

Tree-based preprocessing uses OneHotEncoder for Alloy and passes numerical features through.

All preprocessing is fitted on training data through pipelines to avoid train/test leakage.

## Train/test split

Current split:

```text
80% training
20% held-out test
random_state = 42
```

Classification targets are stratified.

The held-out test set is reserved for final evaluation and is not used for threshold tuning.

## Feature engineering experiment

The team tested derived features such as:

```text
Pour_Temp_sq
Mold_Moisture_sq
Cooling_Time_sq
Riser_sq

Pour_Temp_x_Mold_Moisture
Pour_Temp_x_Cooling_Time
Mold_Moisture_x_Cooling_Time
Riser_div_Cooling_Time
```

The experiment did not improve F1 consistently across targets, so the project retained the original five raw inputs as the cleaner reference representation.

## Useful commands

```bash
python -m pytest src/data -v
```

```bash
python -c "from src.data.load_data import load_csv; from pathlib import Path; df=load_csv(Path('data/raw/foundry_quality_dataset_13548.csv')); print(df.shape); print(df.isna().sum().sum()); print(df.duplicated().sum())"
```

## Current inference

The dataset is structurally clean, but the classification targets are imbalanced. The available five process variables contain useful but limited predictive signal. Therefore the later ML stages must evaluate recall, precision and F1 rather than relying only on accuracy.
