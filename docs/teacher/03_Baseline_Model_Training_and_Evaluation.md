# 03 — Baseline Model Training and Evaluation

## 1. Purpose

This stage establishes the first machine-learning baseline for the AI-Driven Casting Quality System.

The purpose of baseline modeling is to create an initial performance reference before attempting model improvements.

The system predicts four quality-related targets:

### Classification Targets

- Defect
- Porosity
- Scrap

### Regression Target

- Yield

---

# 2. Input Features

The baseline models use five selected manufacturing input features:

- Alloy
- Pour_Temp
- Mold_Moisture
- Cooling_Time
- Riser

These features were selected earlier in the project and remain unchanged during baseline evaluation.

---

# 3. Train-Test Split

The dataset contains 13,548 observations.

The data was divided into:

- Training data: 10,838 observations
- Testing data: 2,710 observations

The approximate split is:

- 80% training
- 20% testing

For classification targets, stratification was used so that the class distribution remained similar between training and testing data.

---

# 4. Baseline Classification Models

Two classification algorithms were evaluated.

## Logistic Regression

Logistic Regression provides a simple baseline model.

It is useful because it establishes how well a relatively simple linear decision boundary can predict casting quality outcomes.

## Random Forest Classifier

Random Forest can model nonlinear relationships and interactions between manufacturing variables.

This makes it useful for comparing a more flexible machine-learning model against Logistic Regression.

---

# 5. Baseline Regression Models

Two regression algorithms were evaluated for Yield prediction.

## Linear Regression

Linear Regression provides a simple baseline for predicting Yield from the selected process parameters.

## Random Forest Regressor

Random Forest Regressor can capture nonlinear relationships between manufacturing inputs and Yield.

---

# 6. Evaluation Metrics

## Classification Metrics

The following metrics were calculated:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Accuracy alone was not considered sufficient because the classification targets contain more `No` observations than `Yes` observations.

Recall and F1-score are particularly important because they measure how effectively the system identifies positive quality events.

## Regression Metrics

The following metrics were calculated:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

# 7. Baseline Results

## Defect

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.6690 | 0.8235 | 0.0154 | 0.0303 |
| Random Forest | 0.6550 | 0.4405 | 0.1101 | 0.1762 |

Random Forest achieved better Recall and F1-score.

However, both models had difficulty detecting positive Defect cases.

---

## Porosity

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.6443 | 0.6054 | 0.0894 | 0.1559 |
| Random Forest | 0.6406 | 0.5286 | 0.1950 | 0.2849 |

Random Forest achieved the strongest baseline Recall and F1-score for Porosity.

However, positive-class detection remained limited.

---

## Scrap

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7760 | 0.0000 | 0.0000 | 0.0000 |
| Random Forest | 0.7753 | 0.4000 | 0.0066 | 0.0130 |

The Scrap target demonstrates why accuracy cannot be used alone.

Despite approximately 77% accuracy, both models almost completely failed to identify positive Scrap cases.

This indicates that class imbalance requires further investigation.

---

## Yield

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 5.1960 | 6.1754 | 0.0493 |
| Random Forest Regressor | 5.0923 | 6.0754 | 0.0798 |

Random Forest Regressor performed slightly better than Linear Regression.

However, both R² scores were low.

This indicates that the five selected process inputs explain only a limited portion of Yield variation in the baseline experiment.

---

# 8. Key Findings

The baseline pipeline successfully:

- Loaded the dataset.
- Split the dataset into training and testing data.
- Trained classification models.
- Trained regression models.
- Generated predictions.
- Calculated evaluation metrics.
- Generated confusion matrices.
- Saved evaluation reports.

The main performance findings were:

1. Positive-class Recall was low for all classification targets.
2. Scrap detection was particularly poor.
3. Random Forest generally achieved better Recall and F1-score than Logistic Regression.
4. Yield prediction performance was weak for both baseline regression models.
5. The selected five manufacturing inputs may contain limited predictive information for some targets.

---

# 9. Next Improvement Stage

The next stage will investigate three improvements.

## Class Imbalance Handling

Balanced model configurations will be tested to improve detection of positive quality events.

## Cross-Validation

Cross-validation will be used on the training data to obtain more reliable performance estimates.

## Feature Engineering

Additional derived features will be created using the original five manufacturing input variables.

These experiments will be performed separately so that the effect of each improvement can be evaluated clearly.

---

# 10. Conclusion

The baseline stage successfully established a reproducible machine-learning reference point.

The results show that the technical pipeline is functioning correctly, but predictive performance requires improvement.

The baseline metrics will therefore serve as the comparison point for future experiments involving class balancing, cross-validation, feature engineering, and more advanced machine-learning models.