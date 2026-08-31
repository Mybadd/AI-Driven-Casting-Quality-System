AI-Driven Casting Quality System
1. Purpose of This Stage

The purpose of this stage is to understand and evaluate the available casting dataset before developing machine learning models.

The dataset is analysed to determine:

Available casting process parameters
Available quality outputs
Data types and data quality
Missing values and duplicate records
Distribution of target variables
Relationships between numerical parameters
Suitability of the dataset for machine learning

No machine learning model is trained during this stage.
2. Dataset Used

The current dataset used for the initial development of the system is:

foundry_quality_dataset_13548.csv
The dataset contains 13,548 casting records and 10 columns.

The dataset is stored in:

data/raw/

The raw dataset is treated as read-only and is never modified directly by the application.

3. Dataset Structure

The dataset contains the following parameters.

Parameter	Type	Role
Batch	Identifier	Record identification
Alloy	Categorical	ML input feature
Pour_Temp	Numerical	ML input feature
Mold_Moisture	Numerical	ML input feature
Cooling_Time	Numerical	ML input feature
Riser	Numerical	ML input feature
Defect	Categorical	Classification target
Porosity	Categorical	Classification target
Scrap	Categorical	Classification target
Yield	Numerical	Regression target
4. Machine Learning Input Features

Only the following five parameters are used as primary machine learning input features:

Alloy
Pour_Temp
Mold_Moisture
Cooling_Time
Riser
Reason

These parameters represent meaningful casting-related information that can be used to analyse relationships with casting quality outcomes.

The Batch column is not used as a machine learning feature because it acts as an identifier.

5. Machine Learning Outputs

The system uses four quality-related outputs.

Classification Outputs
Defect
Porosity
Scrap

These outputs are treated as classification problems.

Regression Output
Yield

Yield is treated as a regression problem because it represents a numerical casting performance value.

6. Yield Data Conversion

During initial analysis, the Yield column was found to contain percentage-formatted values.

Example:

60.90%

Machine learning and statistical analysis require numerical values.

Therefore, the system converts the values during data loading:

60.90%
    ↓
60.90

This conversion is performed only in memory.

The original raw dataset remains unchanged.

7. Data Quality Validation

The dataset validation module checks:

Dataset availability
File format
Empty dataset condition
Number of rows
Number of columns
Data types
Missing values
Duplicate rows
Unique values
Numerical statistics
Categorical value distributions

A data quality report is automatically generated after validation.

8. Exploratory Data Analysis

Exploratory Data Analysis is performed before machine learning development.

The following analyses are generated.

8.1 Target Distribution Analysis

The system analyses the distribution of:

Defect
Porosity
Scrap
Yield

This analysis helps determine whether classification targets are balanced or imbalanced.

8.2 Feature Analysis

The following process parameters are analysed:

Pour_Temp
Mold_Moisture
Cooling_Time
Riser

The analysis includes:

Minimum value
Maximum value
Mean
Median
Standard deviation

The categorical feature Alloy is analysed based on value frequency.

8.3 Correlation Analysis

A numerical correlation matrix is generated to analyse relationships between:

Pour_Temp
Mold_Moisture
Cooling_Time
Riser
Yield

A correlation heatmap is also generated for visualization.

Correlation analysis is used as an exploratory tool and does not independently prove a causal relationship between casting parameters and quality outcomes.

9. Generated Reports

The EDA process automatically generates:

reports/eda/
├── correlations.csv
├── ml_readiness_report.md
└── target_distribution.json
10. Generated Visualizations

The following visualizations are generated:

reports/figures/
├── defect_distribution.png
├── porosity_distribution.png
├── scrap_distribution.png
├── yield_distribution.png
└── correlation_heatmap.png

These visualizations can later be used for project demonstration and teacher evaluation.

11. Current ML Architecture

The current planned architecture is:

                 CASTING INPUT PARAMETERS

        Alloy
        Pour Temperature
        Mold Moisture
        Cooling Time
        Riser
                 │
                 ▼
          DATA PROCESSING
                 │
                 ▼
         FEATURE PREPARATION
                 │
                 ▼
        ┌────────┼─────────┬──────────┐
        │        │         │          │
        ▼        ▼         ▼          ▼
      Defect  Porosity    Scrap      Yield
        │        │         │          │
 Classification Classification Classification Regression
        │        │         │          │
        └────────┴─────────┴──────────┘
                         │
                         ▼
             AI QUALITY INTELLIGENCE
                         │
                         ▼
                   DASHBOARD
12. Current Project Status

The following components have been completed:

Component	Status
Project structure	Completed
Python environment	Completed
Dependency setup	Completed
Dataset integration	Completed
Data loading	Completed
Data validation	Completed
Data quality reporting	Completed
Yield format handling	Completed
Exploratory Data Analysis	Completed
Target distribution analysis	Completed
Correlation analysis	Completed
ML model development	Not started
Dashboard development	Not started
13. Conclusion of the Current Stage

The available dataset has been successfully integrated into the AI-Driven Casting Quality System.

The dataset provides five meaningful casting-related input parameters and four quality-related outputs.

The system is therefore being designed as a multi-output quality analysis platform consisting of:

Defect classification
Porosity classification
Scrap classification
Yield prediction

The next development stage is to inspect the EDA results and design the preprocessing and machine learning pipeline based on the observed dataset characteristics.