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