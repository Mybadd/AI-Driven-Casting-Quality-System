# Class-Balanced Classification Experiment

## 1. Purpose

The initial classification baseline showed that the quality-event targets
were imbalanced, with the `"Yes"` class representing the minority class.

This experiment investigates whether assigning greater importance to the
minority class improves the ability of the classification models to detect
quality events.

The original baseline models were not modified. Class-balanced models were
implemented as a separate experimental stage so that their results could be
compared fairly against the original baseline.

---

## 2. Experimental Setup

The same dataset and approved ML input features were used as the original
baseline experiment.

### Input Features

- Alloy
- Pour_Temp
- Mold_Moisture
- Cooling_Time
- Riser

`Batch` was retained as an identifier and was not used as an ML feature.

### Classification Targets

- Defect
- Porosity
- Scrap

### Models

Two class-balanced models were evaluated for each target:

1. Balanced Logistic Regression
2. Balanced Random Forest Classifier

Class balancing was implemented using:

```text
class_weight="balanced"
This allows the models to give greater importance to the minority class.

The same train/test split methodology used for the baseline experiment was
retained.

3. Why Class Balancing Was Tested

The original baseline results showed very low recall for the positive
quality-event class.

For example, the original Scrap Logistic Regression model achieved:

Recall: 0.0000
F1-score: 0.0000

This indicates that the model was effectively failing to identify the
minority "Yes" class.

Because a casting-quality system should not evaluate classification
performance using accuracy alone, a separate class-balanced experiment was
performed.

The main metrics considered were:

Accuracy
Precision
Recall
F1-score

Particular attention was given to recall and F1-score for the minority
quality-event class.

4. Results
Defect
Model	Accuracy	Precision	Recall	F1-score
Balanced Logistic Regression	0.5601	0.3906	0.5584	0.4597
Balanced Random Forest	0.5841	0.3930	0.4427	0.4164

The balanced Logistic Regression model achieved the highest F1-score for
Defect and detected a substantially larger proportion of positive defect
cases than the original baseline models.

Porosity
Model	Accuracy	Precision	Recall	F1-score
Balanced Logistic Regression	0.5624	0.4298	0.5879	0.4966
Balanced Random Forest	0.6125	0.4718	0.4623	0.4670

Balanced Logistic Regression achieved the highest recall and F1-score for
Porosity.

Scrap
Model	Accuracy	Precision	Recall	F1-score
Balanced Logistic Regression	0.5288	0.2423	0.5189	0.3304
Balanced Random Forest	0.6513	0.2529	0.2850	0.2680

The balanced Logistic Regression model substantially improved Scrap
detection compared with the original baseline, although its precision
remained relatively low.

5. Comparison With Original Baseline

The class-balanced experiment produced a clear improvement in minority-class
recall and F1-score.

Defect

Original Logistic Regression:

Recall = 0.0154
F1 = 0.0303

Balanced Logistic Regression:

Recall = 0.5584
F1 = 0.4597
Porosity

Original Logistic Regression:

Recall = 0.0894
F1 = 0.1559

Balanced Logistic Regression:

Recall = 0.5879
F1 = 0.4966
Scrap

Original Logistic Regression:

Recall = 0.0000
F1 = 0.0000

Balanced Logistic Regression:

Recall = 0.5189
F1 = 0.3304

These results demonstrate that class imbalance was significantly affecting
the original classification results.

6. Accuracy Trade-Off

The improvement in minority-class detection came with a reduction in
overall accuracy.

This behavior is expected because the class-balanced models assign greater
importance to minority quality events instead of favoring the majority
"No" class.

Therefore, accuracy alone is not sufficient for selecting a quality
classification model.

For this project, recall and F1-score are particularly useful because the
system is intended to support identification of potentially problematic
casting outcomes.

However, increasing recall can also increase false positive predictions.
Therefore, precision must also be considered when selecting a model for the
final system.

7. Engineering Interpretation

The experiment provides an important modeling finding for the
AI-Driven Casting Quality System.

The classification problem cannot be evaluated only by asking how many
predictions are correct overall.

A model that predicts the majority "No" class frequently can obtain
reasonable accuracy while missing a large number of actual quality events.

The class-balanced experiment reduces this problem by giving greater
importance to the minority class.

This supports the use of multiple evaluation metrics in the final system,
particularly:

Precision
Recall
F1-score
Accuracy
Confusion matrix

The results should be interpreted as model behavior rather than as proof of
causal metallurgical relationships.

8. Decision

The original baseline models will remain unchanged.

They provide the fixed reference point against which future model
experiments can be compared.

The class-balanced models are retained as a separate experimental stage.

The experiment demonstrates that class balancing can substantially improve
minority-class detection for Defect, Porosity, and Scrap.

The balanced models are therefore candidates for further investigation, but
they are not automatically selected as the final production models.

9. Limitations

The experiment uses the same five approved input features as the baseline:

Alloy
Pour_Temp
Mold_Moisture
Cooling_Time
Riser

No additional process variables were introduced.

The results therefore reflect the information available in the current
dataset.

The class-balanced experiment also does not establish whether the available
features are sufficient for high-quality prediction in a real foundry
environment.

Further experiments are required before selecting the final classification
models.

10. Next Step

The next classification experiment will investigate probability-threshold
tuning.

The purpose will be to determine whether changing the classification
decision threshold can produce a more useful precision-recall trade-off
than the default threshold.

This will be evaluated without changing the original baseline models.

The overall development sequence remains:

Baseline Models
      ↓
Class-Balanced Models
      ↓
Threshold Tuning
      ↓
Cross-Validation
      ↓
Feature Engineering
      ↓
Advanced Models
      ↓
Explainability and Quality Analysis