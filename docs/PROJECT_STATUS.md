\# CastingIQ Project Status



\## Implemented components



\- Dataset loading, schema validation, and exploratory analysis.

\- Baseline and advanced classification experiments.

\- Decision-policy and threshold-selection experiments.

\- Model inference pipelines for the supported quality targets.

\- SHAP-based model explainability and anomaly-pattern analysis.

\- Metallurgical rule and engineering review layers.

\- Integrated casting analysis API.

\- Constrained what-if scenario exploration.

\- Persistent SQLite analysis history endpoints.

\- React and TypeScript dashboard with seven navigation screens.

\- Backend API tests and frontend production-build verification.



\## Current application capabilities



The frontend supports process input, model-result review, engineering review, scenario exploration, analysis history, and methodology guidance. The backend exposes health, analysis, scenario, and history endpoints.



\## Known limitations



\- Predictions are estimates and must not be treated as confirmed defects.

\- Scrap prediction performance is weak and false alerts are a concern.

\- SHAP importance does not establish causality or effect direction.

\- Scenario rankings are model-based comparisons, not guaranteed process optimization.

\- Independent validation data must remain separate from training and threshold selection.

\- Calibration and further validation should be treated as incomplete until separately implemented and verified.

\- The anomaly detector's training and deployment separation should be reviewed before making deployment claims.

\- The application has not been established as production-ready process-control software.



\## Verification status



\- Frontend production build: passed in the latest reported build.

\- Backend API tests: previously reported as passing; rerun after the latest backend changes before committing.

\- Runtime verification: check the complete user workflow, including analysis submission, scenario comparison, and history retrieval.



\## Next recommended work



1\. Re-run backend tests and frontend build.

2\. Exercise all dashboard screens against the running backend.

3\. Review dependency and ignore rules.

4\. Keep generated databases and build artifacts out of source control.

5\. Review independent validation protocol before using the held-out validation dataset.

6\. Record any additional test results and limitations before the final academic demonstration.

