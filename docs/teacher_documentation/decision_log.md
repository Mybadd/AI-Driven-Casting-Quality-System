# Project Decision Log

## Decision 001

**Decision:**  
Start the project by creating the complete modular folder structure
before implementing the machine-learning components.

**Why:**  
The system contains multiple components including data processing,
machine learning, explainability, anomaly detection, analysis,
optimization, backend services, and a frontend dashboard. A modular
structure makes these components easier to develop, test, maintain,
and integrate.

---

## Decision 002

**Decision:**  
Use a dedicated Python virtual environment and install project
dependencies inside it.

**Why:**  
The virtual environment isolates this project's dependencies from the
system Python installation and helps ensure the project can be
reproduced consistently on another machine.

---

## Decision 003

**Decision:**  
Install only the initial dependencies required for the current phase.

**Why:**  
Dependencies will be added when their corresponding components are
implemented. This keeps the environment focused and reduces
unnecessary dependency conflicts.