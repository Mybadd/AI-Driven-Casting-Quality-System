# Part 3 — Decision Policy, Threshold Selection & Final Evaluation

## Scope
This document covers the current decision-oriented classification layer. It does not describe frontend implementation.

## Why a decision layer was added

A classifier outputs a probability, but an engineering system needs a decision state.

For this project, a missed quality event is generally more undesirable than generating an additional investigation alert. However, excessive false alarms also make the system impractical.

Therefore the current prototype policy is:

```text
Minimum recall: 50%
Maximum alert rate: 50%
```

This is a project prototype policy, not an industry standard.

## Threshold selection

The default classification threshold of 0.50 was not treated as automatically correct.

Threshold candidates were selected using out-of-fold predictions from the training portion.

The held-out 20% test set remained untouched during threshold selection.

Relevant code:

```text
src/models/decision/oof_threshold_selection.py
src/models/decision/run_oof_threshold_selection.py
src/models/decision/evaluate_final_policy.py
src/models/decision/decision_policy.py
```

## Selected operating points

```text
Defect:
Model     = XGBoost
Threshold = 0.48

Porosity:
Model     = XGBoost
Threshold = 0.47

Scrap:
Model     = Balanced Logistic Regression
Threshold = 0.50
```

XGBoost did not produce a feasible Scrap operating point under the prototype recall/alert constraints, so Balanced Logistic Regression was retained for Scrap.

## Held-out test results

| Target | Model | Threshold | Precision | Recall | F1 | Alert rate |
|---|---|---:|---:|---:|---:|---:|
| Defect | XGBoost | 0.48 | 38.6% | 53.5% | 44.9% | 46.5% |
| Porosity | XGBoost | 0.47 | 44.2% | 59.0% | 50.6% | 49.0% |
| Scrap | Balanced Logistic | 0.50 | 24.2% | 51.9% | 33.0% | 48.0% |

The operating points satisfy the prototype constraints on the held-out test set.

## Decision states

The current policy also uses a review margin:

```python
REVIEW_MARGIN = 0.10
```

Conceptually:

```text
probability >= threshold + 0.10
    -> High-risk / further engineering investigation recommended

probability <= threshold - 0.10
    -> Low-risk / no high-risk signal

otherwise
    -> Review / close to decision boundary
```

This prevents the system from forcing every borderline case into a strong yes/no engineering conclusion.

## Example

For:

```text
Alloy          = Al-Si
Pour_Temp      = 710
Mold_Moisture  = 3.0
Cooling_Time   = 330
Riser          = 1.25
```

Current probabilities are approximately:

```text
Defect   = 42.35%
Porosity = 37.32%
Scrap    = 48.29%
```

The current decision policy classifies these near-boundary outputs as Review.

## Why this matters

This converts ML probability into an engineering-facing screening decision while preserving uncertainty.

The system is not designed to automatically reject a casting.

For example, Scrap precision is only about 24.2%, so an alert must be treated as an investigation signal rather than an automatic scrap command.

## Useful commands

```bash
python -m pytest src/models/decision -v
```

```bash
python -c "from src.analysis.prediction_service import analyze_casting_input; x={'Alloy':'Al-Si','Pour_Temp':710,'Mold_Moisture':3.0,'Cooling_Time':330,'Riser':1.25}; import pprint; pprint.pp(analyze_casting_input(x))"
```
