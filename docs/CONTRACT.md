# JSON request/response contract - schema 1.0.0

## Request

```json
{
  "schema_version": "1.0.0",
  "request_id": "demo-ml-001",
  "features": {
    "palm_width": 8.0,
    "palm_length": 18.5,
    "finger_ratio": 0.95,
    "face_width": 14.0,
    "face_height": 19.0,
    "eye_distance": 6.2
  }
}
```

This example illustrates valid structure/ranges, not a promised category.
All three top-level fields are required; extras are rejected. request_id must be
a string of 1-80 ASCII letters, digits, periods, underscores, or hyphens. Feature
objects must contain exactly the six named numeric fields in docs/DATASET.md.
Feature order is fixed internally and never inferred from JSON insertion order.

Booleans, strings (even numeric strings), nulls, arrays/objects as measurements,
nonfinite numbers, and out-of-range values are rejected. Bounds are inclusive.
Duplicate JSON keys, malformed JSON, NaN, and Infinity tokens are rejected by
the file/stdin JSON reader. The Python API also rejects nonfinite float values.

CLI: predict.py --request FILE [--backend sklearn|portable] [--output FILE].
--stdin is an alternative to --request. The CLI default is sklearn. The
ArchetypeAgent Python class defaults to its lightweight portable implementation;
pass backend="sklearn" explicitly to load the full pipeline. Model paths are
resolved relative to the project files; request-file paths are relative to the
current shell directory or absolute. No training occurs during either backend.

## Response fields

| Field | Meaning |
| --- | --- |
| schema_version | 1.0.0 |
| request_id | Echoed request string; null when unavailable/not a string |
| status | ok, uncertain, invalid_input, or unavailable |
| category | Accepted fictional label, otherwise null |
| confidence | Maximum model class probability, 0-1; null before a prediction |
| confidence_type | uncalibrated_model_probability |
| model_version | archetype-lr-1.1.0 |
| acceptance_threshold | Saved threshold 0.60, or null when no model was used |
| probabilities | Object keyed in fitted classifier class order; null before prediction |
| explanation | Accepted-class local score explanation; null for other statuses |
| reason | below_acceptance_threshold for uncertain; otherwise null |
| error | {code,message} for invalid_input/unavailable; otherwise null |
| warnings | Fictional-label, synthetic-data, and noncausal/uncalibrated limitations |
| processing_ms | Time inside the prediction operation; excludes CLI startup/model loading |

Classes are creator, explorer, guardian, leader in the fitted classifier's
probability-column order. Labels in the dataset card are listed differently;
model classes_ determines evaluation and runtime order.

## Acceptance and explanation

The acceptance rule is max_probability >= 0.60; below that, category is null and
status is uncertain. It is an abstention policy fixed before final test evaluation,
not a calibrated confidence guarantee or a reliable out-of-distribution detector.
Decisions use unrounded numbers; printed probabilities use 12 decimal places.

For an accepted class k, each local term is c_jk=z_j*beta_kj, with z computed
using the saved fitted scaler. The response returns the three terms of largest
absolute magnitude, including feature, standardized_value, contribution, and
direction (supports, opposes, or neutral). Terms act on a class's linear score,
not directly on its probability or on personal characteristics.

The explanation also reports intercept, sum_all_contributions (all six terms),
and linear_score. Intercept + all six terms reconstructs that class score.
The top three terms alone are not the full sum. Softmax probabilities also
depend on competing class scores. None of these are causal effects.

## Failure codes and return codes

Validation codes: invalid_request, missing_field, unexpected_field,
unsupported_schema_version, invalid_request_id, invalid_features,
missing_feature, unexpected_feature, nonnumeric_feature, nonfinite_feature,
out_of_range. File/JSON reader codes: request_file_unreadable, malformed_json.
The first applicable validation error is returned, with category/confidence null.

Valid requests with missing/broken model files or missing/mismatched native
dependencies return unavailable/model_unavailable. This extra operational status
does not replace the required invalid_input/uncertain behavior. Package hashes
detect accidental corruption; they are not a cryptographic trust signature.
Only trusted joblib packages should be loaded, because joblib is pickle-based.

CLI codes: 0 ok, 1 uncertain, 2 invalid_input, 3 unavailable, 4 output-write failure.
Incorrect command syntax uses argparse usage text/code 2 rather than JSON.
Batch mode treats expected uncertainty/rejections as passed checks. For custom
request folders, completion is not an accuracy or accepted-input claim.
