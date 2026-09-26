# Project Decision Log

## Decision 001

**Decision:**  
Start the project by creating the complete modular folder structure
before implementing the machine-learning components.

**Why:**  
The system contains multiple components including data processing,
machine learning, explainability, anomaly detection, analysis,
optimization, backend services, and a frontend dashboard. A modular
structure makes these components easier to develop, test, maintain,
and integrate.

---

## Decision 002

**Decision:**  
Use a dedicated Python virtual environment and install project
dependencies inside it.

**Why:**  
The virtual environment isolates this project's dependencies from the
system Python installation and helps ensure the project can be
reproduced consistently on another machine.

---

## Decision 003

**Decision:**  
Install only the initial dependencies required for the current phase.

**Why:**  
Dependencies will be added when their corresponding components are
implemented. This keeps the environment focused and reduces
unnecessary dependency conflicts.
## Decision 005

**Decision:**  
Use five primary casting parameters as the initial machine learning input features and treat Defect, Porosity, Scrap, and Yield as separate quality outputs.

**Why:**  
The dataset contains five meaningful process/material parameters suitable as predictive inputs. Batch is a unique identifier and should not be used as a feature because it may introduce meaningless patterns. The available quality outputs naturally support three classification tasks and one regression task.
## Decision 001

**Decision:**  
Maintain a structured project architecture separating data, source code, models, reports, documentation, backend, frontend, and tests.

**Why:**  
A structured architecture improves maintainability, testing, scalability, and clarity during project evaluation.

---

## Decision 002

**Decision:**  
Treat the raw dataset directory as read-only.

**Why:**  
The original dataset must remain unchanged to preserve data integrity and reproducibility.

---

## Decision 003

**Decision:**  
Create a dedicated data loading and validation layer before developing machine learning models.

**Why:**  
Dataset assumptions should be validated before preprocessing or model development.

---

## Decision 004

**Decision:**  
Convert the Yield column from percentage-formatted strings to numeric values during data loading.

**Why:**  
Yield values are stored in a format such as `60.90%`, which cannot be directly used for statistical analysis or regression models. The conversion is performed in memory so that the original raw dataset remains unchanged.

---

## Decision 005

**Decision:**  
Use five primary casting parameters as the initial machine learning input features and treat Defect, Porosity, Scrap, and Yield as separate quality outputs.

**Why:**  
The dataset contains five meaningful process and material parameters suitable as predictive inputs. Batch is an identifier and should not be used as a machine learning feature.

---

## Decision 006

**Decision:**  
Develop the project as a multi-model AI-driven casting quality system instead of a single defect prediction model.

**Why:**  
The available dataset supports three classification targets and one regression target. Using all available quality outputs provides a broader and more appropriate implementation for the project objective.
## Decision 007

**Decision:**  
Use stratified train/test splitting for the classification targets.

**Why:**  
The Defect, Porosity, and Scrap targets are not perfectly balanced. Stratification preserves the target class proportions in both training and testing datasets.

---

## Decision 008

**Decision:**  
Use separate preprocessing strategies for tree-based models and scale-sensitive models.

**Why:**  
Tree-based models generally do not require feature scaling, while models such as Logistic Regression and Linear Regression benefit from standardized numerical features. Separate reusable preprocessing pipelines allow appropriate preprocessing without duplicating code.
                                MISTAKE
# Baseline Model Evaluation

## Decision: Use Multiple Baseline Models

### Context

The project predicts three casting quality classification targets:

- Defect
- Porosity
- Scrap

It also predicts one regression target:

- Yield

Two baseline models were selected for each problem type.

### Classification Models

- Logistic Regression
- Random Forest Classifier

### Regression Models

- Linear Regression
- Random Forest Regressor

### Reason

The baseline models provide a simple and interpretable starting point before applying more advanced machine-learning techniques.

Logistic Regression and Linear Regression provide simple reference models.

Random Forest models can capture nonlinear relationships between the selected casting process parameters and the target variables.

---

# Decision: Use Recall and F1-Score as Important Classification Metrics

## Context

The classification targets are imbalanced.

The majority class for all three quality targets is `No`.

For example:

- Defect: approximately 66.5% `No`
- Porosity: approximately 63.3% `No`
- Scrap: approximately 77.6% `No`

A model could therefore achieve relatively high accuracy while failing to identify defective or scrap castings.

## Decision

Classification performance should not be judged using accuracy alone.

The project will also evaluate:

- Precision
- Recall
- F1-score
- Confusion Matrix

## Reason

In a casting quality system, failing to identify an actual defect can be more important than correctly predicting the majority `No` class.

Recall and F1-score provide better information about positive-class detection.

---

# Decision: Preserve the Original Five Input Features for Baseline Models

## Selected Features

- Alloy
- Pour_Temp
- Mold_Moisture
- Cooling_Time
- Riser

## Reason

These five variables represent the selected manufacturing inputs for the initial machine-learning pipeline.

The baseline experiments intentionally use only these features so that later improvements can be compared fairly.

---

# Baseline Evaluation Results

## Defect

### Logistic Regression

- Accuracy: 0.6690
- Precision: 0.8235
- Recall: 0.0154
- F1-score: 0.0303

### Random Forest Classifier

- Accuracy: 0.6550
- Precision: 0.4405
- Recall: 0.1101
- F1-score: 0.1762

### Observation

Random Forest achieved better recall and F1-score than Logistic Regression.

However, both models showed limited ability to identify the positive Defect class.

---

## Porosity

### Logistic Regression

- Accuracy: 0.6443
- Precision: 0.6054
- Recall: 0.0894
- F1-score: 0.1559

### Random Forest Classifier

- Accuracy: 0.6406
- Precision: 0.5286
- Recall: 0.1950
- F1-score: 0.2849

### Observation

Random Forest produced the best baseline F1-score and recall for Porosity.

However, recall remained relatively low.

---

## Scrap

### Logistic Regression

- Accuracy: 0.7760
- Precision: 0.0000
- Recall: 0.0000
- F1-score: 0.0000

### Random Forest Classifier

- Accuracy: 0.7753
- Precision: 0.4000
- Recall: 0.0066
- F1-score: 0.0130

### Observation

Both baseline models performed poorly at detecting the positive Scrap class.

The relatively high accuracy is misleading because the dataset contains a large majority of `No` observations.

This target requires class imbalance investigation.

---

## Yield

### Linear Regression

- MAE: 5.1960
- RMSE: 6.1754
- R² Score: 0.0493

### Random Forest Regressor

- MAE: 5.0923
- RMSE: 6.0754
- R² Score: 0.0798

### Observation

Random Forest Regressor performed slightly better than Linear Regression.

However, both models achieved low R² scores.

This suggests that the five selected input features have limited predictive power for Yield in the current baseline configuration.

---

# Decision: Improve the Pipeline Incrementally

## Planned Improvements

The following improvements will be tested separately:

1. Class imbalance handling for classification targets.
2. Cross-validation for more reliable performance estimates.
3. Feature engineering using the original five manufacturing inputs.
4. Advanced machine-learning models after the above experiments.

## Reason

Changing multiple parts of the pipeline simultaneously would make it difficult to determine which improvement caused a change in performance.

The project will therefore evaluate improvements incrementally.
                                MISTAKE

Decision: Keep the completed baseline models unchanged as the reference point for future experiments.

Why: The baseline provides a fixed comparison against which class balancing, cross-validation, feature engineering, and later model improvements can be evaluated fairly.

## Decision: Preserve Original Baseline and Add Class-Balanced Experiment

**Decision:** The original baseline models will remain unchanged and will be
used as the fixed reference point for future experiments.

A separate class-balanced classification experiment was introduced using
`class_weight="balanced"` for Logistic Regression and Random Forest.

**Reason:** The original baseline models showed poor minority-class recall,
especially for the Scrap target. Class balancing substantially improved
recall and F1-score for the `"Yes"` class.

**Trade-off:** Overall accuracy decreased because the balanced models give
greater importance to minority quality events.

**Implication:** Model improvement will be evaluated using precision, recall,
and F1-score in addition to accuracy. The next experiment will investigate
classification probability-threshold tuning.

**Date:** 2026-09-03

## Decision: Introduce Reusable Feature Engineering

**Decision:** Introduce a dedicated feature-engineering module before
evaluating more advanced machine-learning models.

**Reason:** The dataset currently provides only five approved ML input
features, and initial model results indicate limited predictive strength.
Derived relationships may allow the models to capture nonlinear effects and
interactions already implicit in the available process variables.

**Implementation:** Create `src/features/engineering.py` containing
engineering-motivated transformations based only on:
- Pour_Temp
- Mold_Moisture
- Cooling_Time
- Riser

**Constraints:**
- No external raw variables.
- `Batch` remains an identifier and is not used as an ML feature.
- Target variables are never used to construct features.
- Engineered features must be evaluated empirically.
- Features will not be retained solely because they are mathematically
  possible; validation performance must justify their use.

**Validation approach:** Compare the original feature set against the
expanded feature set using stratified cross-validation before deciding
whether the engineered features should be retained.

**Date:** 2026-09-03

## Decision: Retain Balanced Logistic Regression as Current Classification Reference

**Decision:** Balanced Logistic Regression remains the current reference
candidate for quality-event detection. XGBoost is retained as an advanced
comparison model.

**Reason:** XGBoost improved accuracy for Defect, Porosity, and Scrap, but
consistently reduced recall and F1-score relative to the balanced Logistic
Regression model.

**Engineering interpretation:** Accuracy alone is not sufficient for the
casting-quality use case. The system must consider the trade-off between
missing potential quality events and generating false alarms.

**Current status:** No final production model or threshold has been selected.

**Next step:** SHAP-based explainability followed by metallurgical
interpretation and definition of the engineering decision policy.

**Date:** 2026-09-03

## Decision: Engineering Decision Policy and Threshold Selection

### Date
2026-09-06

### Context
Initial threshold selection based only on maximum F1 produced impractically high alert rates, often close to 100%. Therefore, threshold selection was changed from an F1-only approach to an engineering decision-policy approach.

### Decision Policy
A prototype project-level decision policy was defined:

- Minimum recall: 50%
- Maximum alert rate: 50%
- No hard precision constraint at this stage

The purpose is to require the model to identify at least half of the historical positive quality events while limiting the investigation workload to approximately half of the batches.

This is a configurable project decision policy and is not claimed to be an industry-standard foundry threshold.

### Threshold Selection Method
Thresholds are selected using out-of-fold (OOF) predictions generated from the training portion of the dataset.

The held-out 20% test set is not used for threshold selection.

The ranking of feasible thresholds is:

1. Higher recall
2. Higher precision
3. Lower alert rate
4. Higher threshold

If no threshold satisfies the policy constraints, no threshold is selected.

### Selected Operating Points

| Target | Model | Selected Threshold |
|---|---|---:|
| Defect | XGBoost | 0.48 |
| Porosity | XGBoost | 0.47 |
| Scrap | Balanced Logistic Regression | 0.50 |

XGBoost was selected for Defect and Porosity because it provided slightly better recall, precision, F1, and alert-rate performance at the policy operating point.

For Scrap, XGBoost did not provide a feasible threshold satisfying both the minimum recall and maximum alert-rate constraints. Therefore, Balanced Logistic Regression remains the current candidate.

### Important Limitation
These operating points are model-selection results, not evidence that the models provide production-ready quality decisions.

The selected thresholds must be evaluated once on the untouched 20% test set before being considered final.

The system should also support an uncertain/review state rather than forcing a binary engineering decision when model evidence is weak.