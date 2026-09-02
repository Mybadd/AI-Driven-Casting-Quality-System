# Challenges Faced

## Challenge 1 — Classification Pipeline Construction Error

Error:
`AttributeError: 'DataFrame' object has no attribute '_validate_params'`

Brief:
The classification pipeline function was returning the `Pipeline` class instead of a properly instantiated pipeline object.

Approach Used:
Inspected the pipeline creation function and verified the returned object's type. The function was corrected to instantiate `Pipeline` with the preprocessing and model steps. During the correction, the model instance was also explicitly created using `create_classification_model()`.

Resolution:
The function now returns a valid `sklearn.pipeline.Pipeline` object.