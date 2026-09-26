# Feature Engineering Design

## 1. Purpose

Feature engineering was introduced to investigate whether useful relationships
can be derived from the existing casting-process variables.

The purpose is not to artificially increase the number of variables, but to
test whether mathematically and engineering-motivated transformations can
improve model performance.

## 2. Approved Raw Inputs

The project uses the following five ML input features:

- Alloy
- Pour_Temp
- Mold_Moisture
- Cooling_Time
- Riser

`Batch` is retained only as an identifier and is excluded from the ML feature
set.

## 3. Engineered Features

The first feature-engineering experiment contains eight derived features:

| Feature | Description |
|---|---|
| Pour_Temp_sq | Square of pour temperature |
| Mold_Moisture_sq | Square of mold moisture |
| Cooling_Time_sq | Square of cooling time |
| Riser_sq | Square of riser value |
| Pour_Temp_x_Mold_Moisture | Interaction between temperature and moisture |
| Pour_Temp_x_Cooling_Time | Interaction between temperature and cooling time |
| Mold_Moisture_x_Cooling_Time | Interaction between moisture and cooling time |
| Riser_div_Cooling_Time | Riser-to-cooling-time ratio |

## 4. Design Principles

The engineered features are deterministic transformations of the approved
input variables.

No target variable is used in feature construction.

No additional external process measurements are introduced.

The objective is to investigate potential nonlinear and interaction effects
while preserving the original dataset structure.

## 5. Implementation

The transformations are implemented as a reusable function:

```python
add_engineered_features(dataframe)
```python
add_engineered_features(dataframe)

in:

src/features/engineering.py

A feature-name collection is also maintained so that the engineered feature
set can be referenced consistently by later experiments.

6. Verification

The feature-engineering module was successfully compiled and tested with
sample process data.

The verification confirmed that:

the original four numerical variables remain available,
eight engineered features are added,
the transformations produce numerical values,
the ratio calculation does not produce an error for valid cooling-time
values.
7. Experimental Plan

The engineered feature set will be compared against the original feature set
using cross-validation.

The primary comparison will use the class-balanced Logistic Regression model.

The evaluation will consider:

Accuracy
Precision
Recall
F1-score
Fold-to-fold variability

The engineered features will only be retained if they demonstrate useful and
stable validation performance.

8. Current Status

Feature engineering has been implemented and verified.

No conclusion has yet been made about whether the engineered features improve
the final classification models.

The next experiment will perform the controlled cross-validation comparison.