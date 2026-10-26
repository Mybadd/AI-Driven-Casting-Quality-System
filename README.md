# AI-Driven Casting Quality System

## Project Overview

The **AI-Driven Casting Quality System (CastingIQ)** is a machine-learning-based engineering decision-support system for analyzing casting process parameters and estimating quality-related outcomes.

The system combines machine learning, explainable AI, anomaly detection, metallurgical rules, and scenario exploration to help engineers investigate casting quality.

Key capabilities include:

- Casting defect prediction
- Porosity prediction
- Scrap-risk prediction
- Yield-related analysis
- Explainable AI-based parameter analysis
- Process anomaly detection
- Quality analysis and likely contributing-factor support
- What-if scenario analysis
- Model-based scenario exploration
- Interactive dashboard visualization
- Persistent analysis history
- Backend API for analysis and scenario requests

## Project Scope

This project is designed as an **engineering decision-support system**, not an automated casting-process controller.

Predictions are model estimates based on the available dataset. Explanations and metallurgical rules provide supporting evidence for engineering investigation; they do not establish causation. Scenario exploration ranks model-based scenarios and does not guarantee improved real-world casting quality.

The system is intended to support engineering investigation and decision-making. Final process decisions require appropriate engineering review and independent validation.

## Current Development Stage

**Integrated prototype — model development, backend integration, dashboard, and testing.**

The project has progressed beyond the initial data-understanding phase. The current implementation includes trained prediction pipelines, a decision-policy prototype, explainability and anomaly-analysis components, a backend API, a React dashboard, and persistent analysis history.

Further work is required before the system can be considered production-ready.

## Main Features

### 1. Quality Prediction

The system estimates the likelihood of:

- Casting defects
- Porosity
- Scrap

It also provides quality-related analysis using the available process parameters.

### 2. Explainability and Engineering Review

Explainability tools help identify which input parameters contribute most to a model's predictions. Metallurgical rules and engineering-review logic provide additional context.

These outputs indicate model associations and possible contributing factors; they do not prove physical root causes.

### 3. Anomaly Detection

An anomaly-detection component identifies process records that differ from patterns learned by the detector. An anomaly should be treated as a signal for further investigation, not proof of a defective casting.

### 4. What-If Analysis and Scenario Exploration

Users can explore changes to selected process parameters and compare model-based outcomes. Scenario exploration is constrained by the implemented parameter ranges and should be interpreted as exploratory decision support rather than guaranteed process optimization.

### 5. Interactive Dashboard

The frontend provides the following screens:

- Overview
- New Analysis
- Quality Results
- Engineering Review
- What-if Explorer
- Analysis History
- Help & Methodology

### 6. Backend API and Analysis History

The FastAPI backend exposes endpoints for analysis, prediction-related workflows, scenario exploration, and persistent history. Analysis records are stored in a local SQLite database.

## Technology Stack

- **Language:** Python, TypeScript
- **Machine learning:** scikit-learn, XGBoost
- **Explainability:** SHAP
- **Anomaly detection:** Isolation Forest
- **Backend:** FastAPI
- **Frontend:** React, Vite
- **Database:** SQLite
- **Testing:** pytest
- **Version control:** Git

## Dataset

The main development dataset is:

`data/raw/foundry_quality_dataset_13548.csv`

It contains 13,548 records with casting process parameters and quality-related target variables.

The model input features include:

- `Alloy`
- `Pour_Temp`
- `Mold_Moisture`
- `Cooling_Time`
- `Riser`

`Batch` is treated as an identifier rather than a model input.

The dataset and its target relationships have limitations. Model performance, especially for scrap prediction, is not sufficient to justify automatic rejection decisions.

**Independent validation data must remain separate from model training, model selection, and decision-threshold tuning.** Do not use the independent validation file to tune the pipeline.

## Getting Started

### Prerequisites

Install Python and Node.js with npm. Run the following commands from the project root unless otherwise specified.

### 1. Set up the Python environment

Activate the project's virtual environment if it already exists.

For Git Bash on Windows:

```bash
source .venv/Scripts/activate
```

If the environment has not yet been created:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Start the backend

From the project root:

```bash
python -m uvicorn backend.app.main:app --reload
```

The API will normally be available at:

- API base: `http://127.0.0.1:8000`
- Interactive API documentation: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

Keep the backend terminal running.

### 3. Start the frontend

Open a second terminal and run:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, normally:

`http://localhost:5173/`

The dashboard requires the backend to be running for API-backed features.

## Running Tests

From the project root, activate the Python virtual environment and run:

```bash
python -m pytest backend/tests/test_api.py -q
```

Run the frontend production build from the frontend directory:

```bash
cd frontend
npm run build
```

The frontend production build has completed successfully during development. Re-run the commands above after making changes to verify the current working tree.

## API Overview

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API root information |
| GET | `/health` | Health check |
| POST | `/api/predict` | Prediction workflow |
| POST | `/api/analyze` | Integrated quality analysis |
| POST | `/api/scenarios` | What-if scenario exploration |
| POST | `/api/history` | Save an analysis record |
| GET | `/api/history?limit=50&offset=0` | List analysis history |
| GET | `/api/history/{record_id}` | Retrieve an individual history record |

See [`docs/API_DOCUMENTATION.md`](docs/API_DOCUMENTATION.md) for request examples, response details, and error handling.

## Project Structure

```text
AI-DRIVEN-CASTING-QUALITY-SYSTEM/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   └── history_store.py
│   ├── data/
│   └── tests/
├── data/
│   └── raw/
├── frontend/
│   └── src/
├── models/
│   ├── defect/
│   ├── porosity/
│   ├── scrap/
│   └── anomaly/
├── reports/
├── src/
│   ├── analysis/
│   ├── data/
│   ├── features/
│   └── models/
├── docs/
├── tests/
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## Documentation

- [API Documentation](docs/API_DOCUMENTATION.md) — API endpoints, requests, and responses.
- [User Guide](docs/USER_GUIDE.md) — Dashboard usage and setup instructions.
- [Project Status](docs/PROJECT_STATUS.md) — Implemented components, known limitations, and remaining work.
- [Challenges Faced](docs/challenges_faced.md) — Development challenges and observations.
- [Decision Log](docs/DECISION_LOG.md) — Important project decisions.
- [Data Engineering and EDA](01_data_engineering_and_eda.md)
- [Model Development and Comparison](02_model_development_and_comparison.md)
- [Decision Policy and Evaluation](03_decision_policy_and_evaluation.md)
- [Explainability, Anomaly Detection and Metallurgy](04_explainability_anomaly_and_metallurgy.md)
- [Integrated Decision, What-If Analysis, Scenarios and Testing](05_integrated_decision_what_if_scenarios_testing.md)

Documentation links should be checked against the actual files in the repository before committing.

## Limitations and Future Work

Known limitations and further work include:

- Prediction quality is limited by the available dataset and its learned relationships.
- Scrap prediction has substantial false-positive risk and must not be used for automatic rejection.
- Decision thresholds are a prototype policy, not validated industrial operating limits.
- Probability calibration and additional independent validation remain future work.
- Anomaly-detection training and evaluation separation needs further review before deployment.
- Scenario rankings are model-based estimates, not guaranteed real-world improvements.
- Industrial deployment would require broader validation, monitoring, and engineering approval.

## Development Status

The project currently has an integrated prototype covering data processing, machine learning, engineering-oriented analysis, backend services, frontend visualization, and analysis history.

The next stage is to strengthen validation, verify the complete user workflow, review the remaining limitations, finalize the documentation, and prepare the project for demonstration and academic evaluation.
