\# CastingIQ API Documentation



\## Overview



CastingIQ uses a FastAPI backend to expose model-based casting quality analysis, engineering review, scenario exploration, and persistent analysis history.



Default local API address: `http://127.0.0.1:8000`



Interactive API documentation: `http://127.0.0.1:8000/docs`



\## Health endpoints



\### `GET /`



Returns the API root response.



\### `GET /health`



Checks whether the backend is responding.



\## Casting analysis



\### `POST /api/analyze`



Accepts casting process parameters, runs the integrated analysis, and saves the completed analysis to persistent history.



Example request:



```json

{

&#x20; "Alloy": "Al-Si",

&#x20; "Pour\_Temp": 710,

&#x20; "Mold\_Moisture": 3.0,

&#x20; "Cooling\_Time": 330,

&#x20; "Riser": 1.25

}

```



Required input fields:



| Field | Meaning | Dataset range |

|---|---|---|

| `Alloy` | Casting alloy category | One of the supported alloy categories |

| `Pour\_Temp` | Pouring temperature | 680–740 |

| `Mold\_Moisture` | Mold moisture parameter | 2.2–4.0 |

| `Cooling\_Time` | Cooling duration | 250–450 |

| `Riser` | Riser parameter | 1.05–1.55 |



The analysis response includes the submitted input, quality predictions, anomaly information, engineering reviews, scenario-related information where available, and a decision note.



Exact response fields should be verified against the current implementation and the live API schema.



\## Scenario exploration



\### `POST /api/scenarios`



Evaluates a constrained set of alternative process settings for a selected target.



Example request:



```json

{

&#x20; "target": "Defect",

&#x20; "base\_input": {

&#x20;   "Alloy": "Al-Si",

&#x20;   "Pour\_Temp": 710,

&#x20;   "Mold\_Moisture": 3.0,

&#x20;   "Cooling\_Time": 330,

&#x20;   "Riser": 1.25

&#x20; },

&#x20; "parameter\_grid": {

&#x20;   "Pour\_Temp": \[700, 710, 720],

&#x20;   "Mold\_Moisture": \[2.8, 3.0, 3.2]

&#x20; },

&#x20; "top\_n": 5

}

```



The response ranks evaluated scenarios using model estimates. It does not guarantee optimal or causal process settings.



Invalid scenario constraints may return HTTP `422`.



\## Persistent analysis history



\### `POST /api/history`



Saves an analysis history record. The request contains `input` and `result` objects.



\### `GET /api/history`



Returns saved history records. Supports `limit` and `offset` query parameters.



Example:



`GET /api/history?limit=50\&offset=0`



\### `GET /api/history/{record\_id}`



Retrieves a saved record by its ID. A missing record returns HTTP `404`.



The backend uses SQLite for local persistent history. The database is application data and should not normally be committed to source control.



\## Errors and limitations



\- HTTP error responses should be inspected for their `detail` field.

\- Input values should remain within supported dataset ranges.

\- Predictions and anomaly results are decision-support signals, not confirmed defect findings.

\- The API is a student-project prototype, not an automatic process-control or rejection system.

