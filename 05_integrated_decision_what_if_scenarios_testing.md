# Part 5 — Integrated Engineering Decision Layer, What-If, Scenarios & QA

## Scope
This document covers the current integration of the analytical modules. Frontend development is intentionally excluded because it has not started yet.

## Unified engineering decision layer

Current implementation:

```text
src/analysis/engineering_decision_service.py
```

The service combines:

```text
Quality predictions
        +
Decision states
        +
Anomaly detection
        +
SHAP-informed engineering review
        +
Optional scenario exploration
```

Main function:

```python
build_engineering_decision(
    casting_input,
    scenario_grids=None,
)
```

The output is intended for later engineer-facing/dashboard use.

## Current live example

Input:

```python
casting_input = {
    "Alloy": "Al-Si",
    "Pour_Temp": 710,
    "Mold_Moisture": 3.0,
    "Cooling_Time": 330,
    "Riser": 1.25,
}
```

Current model probabilities:

```text
Defect   42.35%
Porosity 37.32%
Scrap    48.29%
```

Current anomaly result:

```text
Normal
score ≈ 0.1698
```

The quality outputs are close enough to their decision boundaries that the decision layer returns Review states.

## What-if analysis

Implementation:

```text
src/analysis/what_if.py
```

Main integration function:

```python
analyze_what_if(
    target,
    base_input,
    changes,
)
```

Example:

```python
result = analyze_what_if(
    "Defect",
    {
        "Alloy": "Al-Si",
        "Pour_Temp": 710,
        "Mold_Moisture": 3.0,
        "Cooling_Time": 330,
        "Riser": 1.25,
    },
    {"Pour_Temp": 720},
)
```

Observed model comparison:

```text
Baseline: 42.35%
Scenario: 47.72%
Change:   +5.37 percentage points
```

The correct interpretation is:

> The model predicts a higher Defect probability for the proposed scenario.

It is not:

> Increasing Pour_Temp causes defects.

## Scenario constraints

Implementation:

```text
src/analysis/scenario_constraints.py
```

Current prototype numeric bounds:

```text
Pour_Temp      680 to 740
Mold_Moisture  2.2 to 4.0
Cooling_Time   250 to 450
Riser          1.05 to 1.55
```

These constraints prevent the scenario explorer from evaluating values outside the current prototype operating ranges.

## Scenario exploration

Implementation:

```text
src/analysis/scenario_explorer.py
```

Current example grid:

```text
Pour_Temp:
700, 710, 720

Mold_Moisture:
2.8, 3.0, 3.2
```

This produced:

```text
9 feasible scenarios
```

The explorer evaluates model probability, compares each scenario with the baseline, and ranks the requested top scenarios.

Example current ranking:

```text
1. 700 / 2.8 -> 40.35%
2. 710 / 2.8 -> 41.31%
3. 700 / 3.0 -> 41.48%
4. 710 / 3.0 -> 42.35%
5. 720 / 2.8 -> 46.65%
```

These are model-based rankings inside the tested grid. They are not guaranteed optimal process settings.

## Automated testing

Current analysis test suite:

```bash
python -m pytest src/analysis -v
```

Current result:

```text
28 passed
```

The tests cover:

- metallurgical rules
- SHAP/engineering review integration
- prediction service
- engineering decision service
- what-if analysis
- scenario constraints
- scenario explorer

Anomaly tests also pass under:

```bash
python -m pytest src/models/anomaly -v
```

## Independent validation dataset

A separate validation dataset is available:

```text
RAT_Readings_Synthetic_Validation.csv
```

It contains:

```text
3,000 readings
same 10-column structure
no missing values
```

It has not been used for model training.

The intended next ML experiment is independent validation on these readings. It should not be used retrospectively to tune the current thresholds if independence is to be preserved.

## Current project position

The current analytical core is implemented:

```text
Data
  ↓
Preprocessing
  ↓
Baseline ML
  ↓
Balanced ML
  ↓
XGBoost
  ↓
Decision Policy
  ↓
SHAP
  ↓
Anomaly Detection
  ↓
Engineering Review
  ↓
What-If
  ↓
Constrained Scenario Exploration
  ↓
Automated Tests
  ↓
Independent Validation (next)
```

The frontend has not started and is deliberately outside this current-progress documentation.

## Useful demo commands

Prediction:

```bash
python -c "from src.analysis.prediction_service import analyze_casting_input; x={'Alloy':'Al-Si','Pour_Temp':710,'Mold_Moisture':3.0,'Cooling_Time':330,'Riser':1.25}; import pprint; pprint.pp(analyze_casting_input(x))"
```

What-if:

```bash
python -c "from src.analysis.what_if import analyze_what_if; x={'Alloy':'Al-Si','Pour_Temp':710,'Mold_Moisture':3.0,'Cooling_Time':330,'Riser':1.25}; print(analyze_what_if('Defect',x,{'Pour_Temp':720}))"
```

Tests:

```bash
python -m pytest src/analysis -v
```
