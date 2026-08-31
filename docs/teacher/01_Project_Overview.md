# AI-Driven Casting Quality System

## 1. Project Overview

The AI-Driven Casting Quality System is a machine-learning-based engineering decision-support system designed to analyse casting process parameters and estimate important casting quality outcomes.

The purpose of the system is not to replace metallurgical simulation software or foundry engineers. Instead, the system provides a data-driven method for analysing historical casting data and generating quality-related predictions and insights.

The system is intended to demonstrate the integration of:

- Machine Learning
- Data Engineering
- Metallurgy and Casting Knowledge
- Explainable Artificial Intelligence
- Software Engineering
- Engineering Decision Support

---

## 2. Problem Statement

Casting quality is influenced by multiple process and material-related parameters.

Examples include:

- Alloy
- Pouring temperature
- Mold moisture
- Cooling time
- Riser configuration

These parameters may influence casting quality outcomes such as:

- Defects
- Porosity
- Scrap
- Yield

In many practical situations, analysing relationships between multiple parameters and quality outcomes can require examination of a large number of production records.

The objective of this project is to develop an AI-driven system capable of learning patterns from historical casting data and providing quality-related predictions and analysis.

---

## 3. Project Objective

The primary objective of the project is to develop a machine-learning-based casting quality analysis system using available casting process data.

The system will analyse five primary input parameters:

1. Alloy
2. Pouring Temperature
3. Mold Moisture
4. Cooling Time
5. Riser

The system will generate predictions for four quality-related outputs:

1. Defect
2. Porosity
3. Scrap
4. Yield

---

## 4. Machine Learning Problem Structure

The project contains four machine learning prediction tasks.

### Classification Tasks

The following outputs are treated as classification problems:

- Defect
- Porosity
- Scrap

### Regression Task

The following output is treated as a regression problem:

- Yield

The same primary casting input parameters are used to develop models for the different quality outputs.

---

## 5. System Architecture

The high-level architecture of the project is:

Casting Dataset
        ↓
Data Loading
        ↓
Data Validation
        ↓
Exploratory Data Analysis
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Machine Learning Models
        ↓
Quality Predictions
        ↓
Explainability and Analysis
        ↓
Dashboard and Decision Support

---

## 6. Current Dataset

The current dataset contains casting records with the following columns:

- Batch
- Alloy
- Pour_Temp
- Mold_Moisture
- Cooling_Time
- Riser
- Defect
- Porosity
- Scrap
- Yield

The Batch parameter is treated as an identifier and is not used as a primary machine learning feature.

---

## 7. Primary Input Features

The machine learning system initially uses five primary input features:

- Alloy
- Pour_Temp
- Mold_Moisture
- Cooling_Time
- Riser

Additional engineered features may later be derived from these parameters during feature engineering.

---

## 8. Quality Outputs

The system provides analysis for:

### Defect Prediction

Estimates the probability or predicted class of a casting defect based on available process parameters.

### Porosity Prediction

Estimates the probability or predicted class of porosity.

### Scrap Prediction

Estimates whether the casting is likely to be classified as scrap.

### Yield Prediction

Estimates the expected yield based on the available input parameters.

---

## 9. Project Scope

The project focuses on data-driven analysis of the available dataset.

The system does not claim to perform complete physical simulation of:

- Molten metal flow
- Heat transfer
- Solidification
- Stress development
- Fluid dynamics

Therefore, the system should be considered an AI-based quality analysis and decision-support system rather than a replacement for specialised casting simulation software.

---

## 10. Expected Final System

The completed project is expected to include:

- Automated data validation
- Data preprocessing pipeline
- Feature engineering
- Multiple machine learning models
- Model comparison
- Classification evaluation
- Regression evaluation
- Explainable AI
- Quality insight generation
- Scenario or what-if analysis
- Interactive dashboard
- Teacher-facing technical documentation

---

## 11. Evaluation Strategy

Classification models will be evaluated using appropriate metrics such as:

- Confusion Matrix
- Precision
- Recall
- F1 Score
- ROC-AUC where appropriate

The Yield regression model will be evaluated using metrics such as:

- MAE
- RMSE
- R² Score

Model evaluation will not rely on accuracy alone.

---

## 12. Current Development Status

Completed components:

- Project structure
- Python environment
- Dependency installation
- Dataset loading
- Dataset validation
- Data quality reporting
- Yield percentage conversion
- Exploratory Data Analysis

Upcoming components:

- EDA result interpretation
- Preprocessing pipeline
- Feature engineering
- Baseline model development
- Advanced model development
- Model evaluation
- Explainability
- Decision-support logic
- Dashboard development

---

## 13. Final Project Vision

The final system will demonstrate how historical casting process data can be converted into an AI-assisted quality analysis platform.

The system combines machine learning predictions with engineering-oriented interpretation and visualization to provide a structured view of multiple casting quality outcomes.