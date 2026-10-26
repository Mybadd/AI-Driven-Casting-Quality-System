# CastingIQ User Guide

## Overview

CastingIQ is a browser-based dashboard for exploring casting quality estimates and engineering review information. The React frontend communicates with a FastAPI backend.

## Starting the application

### Backend

From the project root, activate the project's Python virtual environment if it is not already active, then run:

```bash
python -m uvicorn backend.app.main:app --reload
```

The backend normally runs at `http://127.0.0.1:8000`.

### Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, normally `http://localhost:5173/`.

## Dashboard screens

### 1. Overview

Provides the dashboard landing page and shortcuts to the main workspace screens.

### 2. New Analysis

Choose an alloy and enter pouring temperature, mold moisture, cooling time, and riser values. Submit the form to request an analysis.

The interface displays dataset ranges as input guidance. These ranges should not be interpreted as universal operating recommendations.

### 3. Quality Results

Reviews model-estimated probabilities for casting defects, porosity, and scrap. It also displays the available anomaly-pattern result.

A probability is a model estimate, not a guarantee that a defect exists or will occur.

### 4. Engineering Review

Shows model-identified factors and suggested review actions. Feature importance is not proof of causality and does not, by itself, establish the direction in which a process parameter should be changed.

### 5. What-if Explorer

Compares model estimates across a constrained grid of process settings. Results are model-based rankings, not guaranteed optimal settings. Confirm any proposed changes against actual process requirements and engineering judgement.

### 6. Analysis History

Loads previously saved analyses from the backend history API. Select a record to inspect its stored result and open its results view.

Completed analyses are saved by the backend to local SQLite history.

### 7. Help & Methodology

Summarizes the intended use and limitations of the prototype.

## Recommended workflow

1. Enter a supported set of casting parameters.
2. Run the analysis.
3. Review all quality estimates and any anomaly signal.
4. Examine the engineering review information.
5. Explore alternative scenarios only as a comparison exercise.
6. Use inspection evidence, process records, and engineering knowledge before drawing conclusions.

## Important limitations

CastingIQ is a student-project decision-support prototype. It should not be used as an automatic reject decision, a substitute for physical inspection, or a guarantee of defect-free production. Model quality is limited by the available data and validation evidence.