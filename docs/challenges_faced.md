# Challenges Faced

## Challenge 1 — Classification Pipeline Construction Error

Error:
`AttributeError: 'DataFrame' object has no attribute '_validate_params'`

Brief:
The classification pipeline function was returning the `Pipeline` class instead of a properly instantiated pipeline object.

Approach Used:
Inspected the pipeline creation function and verified the returned object's type. The function was corrected to instantiate `Pipeline` with the preprocessing and model steps. During the correction, the model instance was also explicitly created using `create_classification_model()`.

Resolution:
The function now returns a valid `sklearn.pipeline.Pipeline` object.

Challenge 2 — Regression Model Factory Error

Error: None returned for linear_regression and random_forest_regressor, eventually causing the pipeline prediction error.
Cause: The regression factory contained classification-model branches and did not contain the required regression models.
Approach: Checked the model factory and compared the supported regression model names with the configuration.
Resolution: Corrected the factory to return LinearRegression() and RandomForestRegressor() for the appropriate model names.
Status: Resolved.

## Challenge 2: Regression Model Factory Configuration

### Problem
During regression pipeline testing, the regression model factory returned `None`
instead of creating the requested regression model.

### Cause
The regression factory contained classification-model branches, while the
required regression branches for `LinearRegression` and
`RandomForestRegressor` were missing.

### Approach
The supported regression model names were compared with the model
configuration and the factory implementation was corrected accordingly.

### Resolution
The regression factory was updated to explicitly create:
- `LinearRegression()` for `linear_regression`
- `RandomForestRegressor(...)` for `random_forest_regressor`

The regression pipelines were then tested successfully.

---

## Challenge 3: Class Imbalance in Quality Classification

### Problem
The original classification baselines showed very low recall for the
minority `"Yes"` quality-event class. This was particularly severe for the
Scrap target, where the baseline Logistic Regression model had zero recall.

### Approach
A separate class-balanced experiment was created without modifying the
original baseline models. `class_weight="balanced"` was applied to Logistic
Regression and Random Forest classification models.

### Results
Class balancing substantially improved minority-class detection.

For Defect, balanced Logistic Regression improved recall from 0.0154 to
0.5584 and F1-score from 0.0303 to 0.4597.

For Porosity, balanced Logistic Regression improved recall from 0.0894 to
0.5879 and F1-score from 0.1559 to 0.4966.

For Scrap, balanced Logistic Regression improved recall from 0.0000 to
0.5189 and F1-score from 0.0000 to 0.3304.

### Resolution
The original baseline models were retained unchanged as the fixed reference
point. Class-balanced models were kept as a separate experimental stage.

The results indicate that class imbalance is an important modeling
consideration for this dataset. The next classification experiment will
investigate probability-threshold tuning to study the precision-recall
trade-off.

## Challenge 4: Designing Meaningful Engineered Features

### Problem
The current dataset contains only five approved ML input features. The initial
classification experiments showed that the available variables provide
limited predictive strength.

### Approach
A reusable feature-engineering module was introduced so that additional
features can be derived systematically from the existing process variables
without introducing external or unsupported raw variables.

The initial engineered features include:

- Pour_Temp squared
- Mold_Moisture squared
- Cooling_Time squared
- Riser squared
- Pour_Temp × Mold_Moisture
- Pour_Temp × Cooling_Time
- Mold_Moisture × Cooling_Time
- Riser / Cooling_Time

### Design Principle
Feature engineering is being treated as an experimental step rather than
assuming that every derived feature will improve model performance.

The transformations use only the approved process variables and do not use
the `Batch` identifier or any target information.

### Resolution
The reusable transformation was implemented in:

`src/features/engineering.py`

The module was successfully compiled and verified using sample data. The
result contained the original numerical inputs plus the eight engineered
features.

The next step is to evaluate whether these engineered features provide a
stable improvement through cross-validation.

## Challenge 5: Integrating XGBoost With String Quality Labels

### Challenge 6: Threshold Selection and Decision Policy

Initial threshold tuning based only on maximum F1-score produced extremely
high alert rates for all three classification targets. Although these
thresholds improved recall, they would make the system impractical as an
engineering investigation tool because too many batches would be flagged.

To address this, threshold selection was reframed as an engineering
decision-policy problem rather than a pure metric-optimization problem.

The prototype policy currently uses:

- Minimum recall = 50%
- Maximum alert rate = 50%

Threshold selection is performed using out-of-fold predictions generated
from the training portion of the dataset. The held-out 20% test set remains
untouched until final evaluation.

Under this policy, the current operating-point candidates are:

- Defect → XGBoost, threshold 0.48
- Porosity → XGBoost, threshold 0.47
- Scrap → Balanced Logistic Regression, threshold 0.50

XGBoost did not provide a feasible Scrap threshold satisfying both policy
constraints.

These thresholds are not yet considered final production settings. They
must be evaluated once on the untouched test set before the final decision
policy is accepted.
## Challenge 7: Combining ML Explanations with Engineering Interpretation

### Problem

Machine-learning explanations such as SHAP feature importance identify which input variables influenced a model prediction. However, a raw SHAP ranking is not directly equivalent to a metallurgical explanation or a proven physical cause.

Presenting model-important features as causes could lead to an incorrect engineering interpretation.

### Solution

A separate metallurgical interpretation layer was implemented in:

`src/analysis/shap_to_engineer.py`

The layer maps important model features to engineering-readable interpretations and review actions.

The system uses cautious language such as:

- influential factor
- likely contributing factor
- engineering review
- process condition to review

rather than claiming direct causality.

### Result

The project now converts SHAP outputs into structured engineering review information while maintaining a clear distinction between model behaviour and physical causality.

The engineering review service is implemented in:

`src/analysis/engineering_review_service.py`

It provides engineering reviews for Defect, Porosity, and Scrap.

---

## Challenge 8: Adding Anomaly Detection Without Replacing Quality Prediction

### Problem

A casting process condition can be unusual without necessarily resulting in a defective casting. Therefore, anomaly detection should not be treated as a direct substitute for defect, porosity, or scrap prediction.

### Solution

Isolation Forest was implemented as a complementary analytical component using the approved process inputs:

- Alloy
- Pour Temperature
- Mold Moisture
- Cooling Time
- Riser

The anomaly detector produces an anomaly status and anomaly score.

These outputs are intended to be considered alongside the classification predictions and engineering review.

### Result

The system can distinguish between:

1. Predicted quality risk
2. Unusual process conditions
3. Model-influential process factors
4. Engineering review recommendations

This creates a more complete quality-analysis workflow without treating anomaly detection as a direct quality classifier.

### Limitation

The current anomaly model artifact was trained on the available historical dataset for application inference. A production deployment would require a carefully defined historical reference population and appropriate validation methodology.