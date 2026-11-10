"""Validate synthetic feature requests and use an already fitted model.

The default portable backend uses only Python's standard library. Training and
model selection live in train.py and never run during a prediction request.
"""

import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
from time import perf_counter

from config import (
    CONFIDENCE_TYPE,
    EXPECTED_COUNTS,
    FEATURES,
    LIMITATIONS,
    MODEL_VERSION,
    RANGES,
    ROOT,
    SCHEMA_VERSION,
)


class RequestError(ValueError):
    def __init__(self, code, message):
        self.code, self.message = code, message
        super().__init__(message)


class ModelError(ValueError):
    pass


def validate_request(request):
    # Reject extra fields instead of accidentally treating a label or ID as a predictor.
    if not isinstance(request, dict):
        raise RequestError("invalid_request", "Request must be a JSON object.")
    if not all(isinstance(key, str) for key in request):
        raise RequestError("invalid_request", "Request object keys must be strings.")
    expected = {"schema_version", "request_id", "features"}
    missing = expected - request.keys()
    if missing:
        raise RequestError(
            "missing_field", f"Required field {sorted(missing)[0]} is missing."
        )
    if request.keys() - expected:
        raise RequestError(
            "unexpected_field",
            f"Unexpected top-level field: {sorted(request.keys() - expected)[0]}.",
        )
    if request["schema_version"] != SCHEMA_VERSION:
        raise RequestError(
            "unsupported_schema_version", f"schema_version must be {SCHEMA_VERSION}."
        )
    request_id = request["request_id"]
    if not isinstance(request_id, str) or not re.fullmatch(
        r"[A-Za-z0-9_.-]{1,80}", request_id
    ):
        raise RequestError(
            "invalid_request_id",
            "request_id must contain 1-80 letters, digits, dots, underscores, or hyphens.",
        )
    features = request["features"]
    if not isinstance(features, dict):
        raise RequestError("invalid_features", "features must be a JSON object.")
    if not all(isinstance(key, str) for key in features):
        raise RequestError("invalid_features", "Feature names must be strings.")
    for name in FEATURES:
        if name not in features:
            raise RequestError(
                "missing_feature", f"Required feature {name} is missing."
            )
    extra = features.keys() - set(FEATURES)
    if extra:
        raise RequestError(
            "unexpected_feature", f"Unexpected feature: {sorted(extra)[0]}."
        )
    vector = []
    for name in FEATURES:
        value = features[name]
        # Python treats bool as a number, so it needs a separate rejection here.
        if isinstance(value, bool) or not isinstance(value, Real):
            raise RequestError(
                "nonnumeric_feature",
                f"{name} must be a numeric measurement, not a Boolean/string/null.",
            )
        try:
            value = float(value)
        except (OverflowError, ValueError):
            raise RequestError("nonfinite_feature", f"{name} must be a finite number.")
        if not math.isfinite(value):
            raise RequestError("nonfinite_feature", f"{name} must be finite.")
        low, high = RANGES[name]
        if not low <= value <= high:
            raise RequestError(
                "out_of_range", f"{name} must be between {low} and {high}, inclusive."
            )
        vector.append(value)
    return vector


def response_base(request, model_version=MODEL_VERSION):
    request_id = request.get("request_id") if isinstance(request, dict) else None
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id if isinstance(request_id, str) else None,
        "status": "invalid_input",
        "category": None,
        "confidence": None,
        "confidence_type": CONFIDENCE_TYPE,
        "model_version": model_version,
        "acceptance_threshold": None,
        "probabilities": None,
        "explanation": None,
        "reason": None,
        "error": None,
        "warnings": list(LIMITATIONS),
        "processing_ms": 0.0,
    }


def invalid_response(request, code, message):
    result = response_base(request)
    result["error"] = {"code": code, "message": message}
    return result


class ArchetypeAgent:
    def __init__(self, model_dir=None, backend="portable", threshold_override=None):
        self.folder = Path(model_dir) if model_dir is not None else ROOT / "models"
        self.backend = backend
        try:
            self.metadata = json.loads(
                (self.folder / "metadata.json").read_text(encoding="utf-8")
            )
            self.model = json.loads(
                (self.folder / "portable_model.json").read_text(encoding="utf-8")
            )
            self._check_package()
            self.threshold = (
                self.metadata["acceptance_threshold"]
                if threshold_override is None
                else float(threshold_override)
            )
            if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
                raise ModelError("Invalid acceptance threshold")
            self.pipeline = None
            if backend == "sklearn":
                # This optional backend uses the saved complete sklearn Pipeline.
                # Only load a joblib model package you trust; it is a pickle-based format.
                import joblib
                from importlib.metadata import version

                for package in ["scikit-learn", "numpy", "scipy", "joblib"]:
                    if version(package) != self.metadata["dependencies"][package]:
                        raise ModelError(
                            f"Install the pinned {package} version before using the sklearn backend."
                        )
                path = self.folder / "pipeline.joblib"
                if (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    != self.metadata["pipeline_sha256"]
                ):
                    raise ModelError("Saved pipeline hash differs from its metadata")
                self.pipeline = joblib.load(path)
                if (
                    list(self.pipeline.named_steps["classifier"].classes_)
                    != self.model["classes"]
                ):
                    raise ModelError(
                        "Saved pipeline class order differs from portable metadata"
                    )
            elif backend != "portable":
                raise ModelError("Unsupported backend")
        except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
            raise ModelError(str(error)) from error

    def _check_package(self):
        # The JSON export holds the fitted scaler and classifier, not a second trained model.
        model = self.model
        if (
            model["feature_order"] != FEATURES
            or self.metadata["feature_order"] != FEATURES
        ):
            raise ModelError("Unexpected saved feature order")
        if (
            model["classes"] != self.metadata["class_order"]
            or len(model["classes"]) != 4
        ):
            raise ModelError("Unexpected class order")
        if set(model["classes"]) != set(EXPECTED_COUNTS):
            raise ModelError("Unexpected fictional class labels")
        if model["model_version"] != self.metadata["model_version"]:
            raise ModelError("Model versions do not agree")
        for name in ["mean", "scale"]:
            if len(model[name]) != 6 or not all(math.isfinite(v) for v in model[name]):
                raise ModelError(f"Invalid scaler {name}")
        if not all(value > 0 for value in model["scale"]):
            raise ModelError("Scaler scales must be positive")
        if len(model["coefficients"]) != 4 or any(
            len(row) != 6 for row in model["coefficients"]
        ):
            raise ModelError("Invalid coefficient shape")
        if len(model["intercepts"]) != 4:
            raise ModelError("Invalid intercept shape")
        if not all(math.isfinite(v) for row in model["coefficients"] for v in row):
            raise ModelError("Coefficients must be finite")
        if not all(math.isfinite(v) for v in model["intercepts"]):
            raise ModelError("Intercepts must be finite")

    def predict_vector(self, vector):
        # Reuse the training means/scales. Nothing gets fitted to this request.
        z = [
            (value - mean) / scale
            for value, mean, scale in zip(
                vector, self.model["mean"], self.model["scale"]
            )
        ]
        scores = [
            bias + sum(value * coefficient for value, coefficient in zip(z, row))
            for bias, row in zip(self.model["intercepts"], self.model["coefficients"])
        ]
        if not all(math.isfinite(score) for score in scores):
            raise ModelError("Nonfinite prediction score")
        if self.pipeline is not None:
            probabilities = self.pipeline.predict_proba([vector])[0].tolist()
            z = self.pipeline.named_steps["scaler"].transform([vector])[0].tolist()
            scores = self.pipeline.decision_function([vector])[0].tolist()
        else:
            # Subtract the maximum so exp() stays stable for large linear scores.
            exponentials = [math.exp(score - max(scores)) for score in scores]
            total = sum(exponentials)
            probabilities = [value / total for value in exponentials]
        return z, scores, probabilities

    def predict(self, request):
        started = perf_counter()
        result = response_base(request, self.metadata["model_version"])
        result["acceptance_threshold"] = self.threshold
        try:
            vector = validate_request(request)
        except RequestError as error:
            result["error"] = {"code": error.code, "message": error.message}
            result["processing_ms"] = round((perf_counter() - started) * 1000, 3)
            return result
        z, scores, probabilities = self.predict_vector(vector)
        winner = max(range(4), key=lambda index: probabilities[index])
        confidence = probabilities[winner]
        result["confidence"] = round(confidence, 12)
        result["probabilities"] = {
            label: round(probability, 12)
            for label, probability in zip(self.model["classes"], probabilities)
        }
        if confidence < self.threshold:
            result["status"] = "uncertain"
            result["reason"] = "below_acceptance_threshold"
        else:
            label = self.model["classes"][winner]
            result["status"], result["category"] = "ok", label
            contributions = [
                value * coefficient
                for value, coefficient in zip(z, self.model["coefficients"][winner])
            ]
            strongest = sorted(
                range(6), key=lambda index: abs(contributions[index]), reverse=True
            )[:3]
            result["explanation"] = {
                "method": "standardized_linear_contributions",
                "target_class": label,
                "intercept": round(self.model["intercepts"][winner], 8),
                "sum_all_contributions": round(sum(contributions), 8),
                "linear_score": round(scores[winner], 8),
                "top_features": [
                    {
                        "feature": FEATURES[index],
                        "standardized_value": round(z[index], 8),
                        "contribution": round(contributions[index], 8),
                        "direction": (
                            "supports"
                            if contributions[index] > 0
                            else "opposes" if contributions[index] < 0 else "neutral"
                        ),
                    }
                    for index in strongest
                ],
            }
        result["processing_ms"] = round((perf_counter() - started) * 1000, 3)
        return result


def predict_request(
    request, model_dir=None, backend="portable", threshold_override=None
):
    try:
        validate_request(request)
    except RequestError as error:
        return invalid_response(request, error.code, error.message)
    try:
        return ArchetypeAgent(model_dir, backend, threshold_override).predict(request)
    except ModelError:
        result = response_base(request)
        result["status"] = "unavailable"
        result["error"] = {
            "code": "model_unavailable",
            "message": "Restore the saved model package and check backend dependencies.",
        }
        return result
