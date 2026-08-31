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
## Decision 005

**Decision:**  
Use five primary casting parameters as the initial machine learning input features and treat Defect, Porosity, Scrap, and Yield as separate quality outputs.

**Why:**  
The dataset contains five meaningful process/material parameters suitable as predictive inputs. Batch is a unique identifier and should not be used as a feature because it may introduce meaningless patterns. The available quality outputs naturally support three classification tasks and one regression task.
## Decision 001

**Decision:**  
Maintain a structured project architecture separating data, source code, models, reports, documentation, backend, frontend, and tests.

**Why:**  
A structured architecture improves maintainability, testing, scalability, and clarity during project evaluation.

---

## Decision 002

**Decision:**  
Treat the raw dataset directory as read-only.

**Why:**  
The original dataset must remain unchanged to preserve data integrity and reproducibility.

---

## Decision 003

**Decision:**  
Create a dedicated data loading and validation layer before developing machine learning models.

**Why:**  
Dataset assumptions should be validated before preprocessing or model development.

---

## Decision 004

**Decision:**  
Convert the Yield column from percentage-formatted strings to numeric values during data loading.

**Why:**  
Yield values are stored in a format such as `60.90%`, which cannot be directly used for statistical analysis or regression models. The conversion is performed in memory so that the original raw dataset remains unchanged.

---

## Decision 005

**Decision:**  
Use five primary casting parameters as the initial machine learning input features and treat Defect, Porosity, Scrap, and Yield as separate quality outputs.

**Why:**  
The dataset contains five meaningful process and material parameters suitable as predictive inputs. Batch is an identifier and should not be used as a machine learning feature.

---

## Decision 006

**Decision:**  
Develop the project as a multi-model AI-driven casting quality system instead of a single defect prediction model.

**Why:**  
The available dataset supports three classification targets and one regression target. Using all available quality outputs provides a broader and more appropriate implementation for the project objective.
## Decision 007

**Decision:**  
Use stratified train/test splitting for the classification targets.

**Why:**  
The Defect, Porosity, and Scrap targets are not perfectly balanced. Stratification preserves the target class proportions in both training and testing datasets.

---

## Decision 008

**Decision:**  
Use separate preprocessing strategies for tree-based models and scale-sensitive models.

**Why:**  
Tree-based models generally do not require feature scaling, while models such as Logistic Regression and Linear Regression benefit from standardized numerical features. Separate reusable preprocessing pipelines allow appropriate preprocessing without duplicating code.