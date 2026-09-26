## 5. Model Selection Decision

The initial cross-validation comparison retained Balanced Logistic
Regression as the reference model because it provided better recall and
F1-score than XGBoost at the default classification threshold.

A subsequent engineering decision-policy experiment was introduced because
default-threshold metrics and maximum-F1 threshold selection did not
represent a practical investigation workflow.

The prototype decision policy uses:

- minimum recall = 50%
- maximum alert rate = 50%

Thresholds were selected using out-of-fold predictions from the training
portion of the dataset. The held-out 20% test set was not used for threshold
selection.

At the resulting policy operating points, the current model candidates are:

- Defect → XGBoost, threshold 0.48
- Porosity → XGBoost, threshold 0.47
- Scrap → Balanced Logistic Regression, threshold 0.50

For Defect and Porosity, XGBoost provided slightly better operating-point
performance than Balanced Logistic Regression.

For Scrap, XGBoost did not provide a feasible threshold satisfying both
the minimum-recall and maximum-alert-rate constraints.

These are current model-selection candidates, not final production models.
They must be evaluated on the untouched test set before finalizing the
decision policy.
## 7. Explainability and Engineering Review Integration

To make the machine-learning results useful for engineering decision support, the project integrates SHAP-based explainability with a rule-based metallurgical interpretation layer.

### 7.1 SHAP-Based Feature Importance

SHAP (SHapley Additive exPlanations) is used to identify which input features have the greatest influence on the predictions of the trained XGBoost classification models.

The analysis is performed separately for:

- Defect
- Porosity
- Scrap

The global SHAP results provide feature importance based on mean absolute SHAP value. A larger value indicates that the feature has a greater influence on the model's predictions.

The results showed that the four numerical process variables generally had greater influence than Alloy:

- Mold Moisture
- Riser
- Pour Temperature
- Cooling Time

The relative ranking differs between quality targets.

### 7.2 Metallurgical Interpretation Layer

SHAP values describe model behaviour but do not by themselves provide a metallurgical explanation or establish causality.

Therefore, a rule-based metallurgical interpretation layer was developed in:

`src/analysis/shap_to_engineer.py`

This layer converts model-important features into engineering-readable interpretations and review actions.

For example:

- Mold Moisture is interpreted as a process condition requiring review of mold condition and related gas behaviour.
- Riser is interpreted in relation to feeding and solidification behaviour.
- Pour Temperature is interpreted as a thermal process condition affecting filling and solidification conditions.
- Cooling Time is interpreted in relation to the thermal history and solidification behaviour.

These interpretations are deliberately framed as engineering review guidance rather than proven causal relationships.

### 7.3 Engineering Review Service

The integration is exposed through:

`src/analysis/engineering_review_service.py`

The service provides:

- `get_engineering_review(target)`
- `get_all_engineering_reviews()`

The service loads the appropriate SHAP importance report for each quality target and generates a structured engineering review.

The resulting structure contains:

- Target
- Engineering statement
- Influential factors
- SHAP importance
- Engineering interpretation
- Recommended review action

### 7.4 Current Engineering Review Results

For **Defect**, the highest-ranked engineering factors are:

1. Mold Moisture
2. Riser
3. Pour Temperature
4. Cooling Time

For **Porosity**, the highest-ranked engineering factors are:

1. Riser
2. Mold Moisture
3. Pour Temperature
4. Cooling Time

For **Scrap**, the highest-ranked engineering factors are:

1. Mold Moisture
2. Cooling Time
3. Pour Temperature
4. Riser

These rankings are model-based and should be used to prioritize engineering investigation rather than to claim that a particular factor directly causes the observed quality outcome.

### 7.5 Engineering Interpretation Limitation

The system explicitly distinguishes between model explanation and physical causality.

The engineering review therefore uses the following principle:

> Influential model feature ≠ proven physical cause.

The purpose of this layer is to help engineers identify process conditions that deserve further investigation. Final engineering conclusions should be based on process knowledge, operating limits, inspection results, and other relevant manufacturing evidence.

### 7.6 Role in the Overall System

The current workflow is:

`Casting Input`
→ `ML Prediction`
→ `SHAP Feature Importance`
→ `Metallurgical Interpretation`
→ `Engineering Review`

This establishes the foundation for the next stage of the project: integrating predictions, anomaly detection, engineering review, and what-if analysis into a unified engineering decision-support workflow.
## 8. Anomaly Detection

Anomaly detection was added as a complementary quality-analysis component.

The project uses an Isolation Forest model to identify casting process conditions that are unusual relative to the dataset.

The anomaly detector uses the same approved process inputs:

- Alloy
- Pour Temperature
- Mold Moisture
- Cooling Time
- Riser

`Batch` is not used as an ML feature because it is an identifier.

### 8.1 Isolation Forest

The anomaly detection implementation is located in:

`src/models/anomaly/isolation_forest.py`

The model uses:

- Isolation Forest
- 200 estimators
- 5% contamination
- Random state 42

The model produces:

- Anomaly status
- Anomaly score

The current exploratory analysis identified approximately 5% of observations as anomalies.

### 8.2 Relationship with Observed Quality

The anomaly results were compared with the observed quality outcomes.

Anomalous observations showed:

- Higher observed Defect proportion than normal observations.
- Higher observed Porosity proportion than normal observations.
- Lower average Yield than normal observations.

The separation for Scrap was comparatively weak.

These results indicate that anomaly detection may provide useful complementary information for quality analysis. However, the observed association does not establish that anomalous process conditions cause the corresponding quality outcomes.

### 8.3 Role in the System

Anomaly detection is not used as a replacement for the classification models.

Instead, it provides an additional signal:

`Process Input`
→ `Anomaly Detection`
→ `Normal / Anomaly`

This signal can be combined with:

- ML quality predictions
- Decision thresholds
- SHAP explanations
- Metallurgical interpretation

to provide a more complete engineering review.

### 8.4 Deployment Consideration

The current anomaly model artifact is trained on the available historical dataset for application inference.

For a production deployment, the fitting strategy should be reviewed to ensure that the anomaly detector is trained using an appropriate historical/training reference population and that evaluation methodology is consistent with the final deployment design.

Therefore, the current anomaly results are treated as a project-level analytical component rather than evidence of production-level anomaly detection performance.
## 9. Unified Engineering Decision Layer

The individual ML, anomaly detection, explainability, and metallurgical interpretation components were integrated into a unified engineering decision service.

The service accepts the five approved casting input features:

- Alloy
- Pour_Temp
- Mold_Moisture
- Cooling_Time
- Riser

It combines:

1. Quality prediction results for Defect, Porosity, and Scrap.
2. Engineering decision states based on the selected thresholds.
3. Anomaly detection results from the Isolation Forest model.
4. SHAP-based influential process factors.
5. Metallurgical interpretation and recommended engineering review points.

The resulting structure is designed to provide a single engineer-facing analysis that can later be consumed by the application dashboard.

The service was validated using an automated pytest test. The test confirmed that all three quality targets, anomaly information, and engineering review information are returned correctly.

The engineering decision layer is intended as decision-support rather than automatic process control. Model predictions and SHAP factors are treated as model-based signals and review support, not as proof of causal relationships.
## 10. What-If Scenario Analysis

A model-based what-if analysis layer was implemented to allow comparison of a baseline casting condition with a proposed process scenario.

The analysis uses only the approved input features:

- Alloy
- Pour_Temp
- Mold_Moisture
- Cooling_Time
- Riser

A scenario can modify one or multiple process parameters. The system then obtains the model probability for the baseline and scenario and reports the change in predicted quality risk.

For example, an experimental scenario changing Pour_Temp from 710 to 720 produced a change in the model-based Defect probability from 42.35% to 47.72%, corresponding to an increase of 5.37 percentage points.

A multi-parameter Porosity scenario changing Pour_Temp from 710 to 720, Mold_Moisture from 3.0 to 3.4, and Cooling_Time from 330 to 350 produced an increase in predicted Porosity probability from 37.32% to 57.17%.

These results demonstrate scenario comparison rather than causal analysis. The system explicitly communicates that a model-based probability change does not establish a causal effect or guarantee the resulting casting quality.

The what-if integration was validated using six automated tests covering scenario validation, scenario creation, invalid feature handling, probability comparison, scenario summary generation, and actual model-inference integration.