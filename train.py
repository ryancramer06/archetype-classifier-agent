"""Required leakage-safe training workflow. Run once, then use predict.py."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import warnings

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer, recall_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    ACCEPTANCE_THRESHOLD,
    CONFIDENCE_TYPE,
    FEATURES,
    MODEL_VERSION,
    RANGES,
    ROOT,
    SCHEMA_VERSION,
    SEED,
)
from data_utils import load_dataset
from evaluation_utils import evaluate_predictions, make_figures, write_report


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def train():
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    model_dir, reports, evidence = ROOT / "models", ROOT / "reports", ROOT / "evidence"
    for folder in [model_dir, reports, evidence]:
        folder.mkdir(exist_ok=True)
    vectors, targets, dataset = load_dataset()
    X, y = np.asarray(vectors, dtype=float), np.asarray(targets)
    indices = np.arange(len(y))

    # Reserve the test rows before any scaler is fitted or hyperparameter is chosen.
    train_ids, test_ids = train_test_split(
        indices, test_size=0.20, stratify=y, random_state=SEED
    )
    X_train, X_test, y_train, y_test = (
        X[train_ids],
        X[test_ids],
        y[train_ids],
        y[test_ids],
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(cv.split(X_train, y_train))
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED),
            ),
        ]
    )
    grid = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__class_weight": [None, "balanced"],
    }
    # Macro F1 chooses the candidate; recall scorers record each class on the same folds.
    scoring = {"f1_macro": "f1_macro"}
    for label in sorted(Counter(y_train)):
        scoring[f"recall_{label}"] = make_scorer(
            recall_score, labels=[label], average="macro", zero_division=0
        )
    search = GridSearchCV(
        pipeline,
        grid,
        scoring=scoring,
        refit="f1_macro",
        cv=cv,
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
    )
    print(
        "Fitting 8 candidates x 5 training folds (test rows remain reserved).",
        flush=True,
    )
    with warnings.catch_warnings(record=True) as observed_warnings:
        warnings.simplefilter("always")
        search.fit(X_train, y_train)
    convergence = [
        str(item.message)
        for item in observed_warnings
        if issubclass(item.category, ConvergenceWarning)
    ]
    if convergence:
        raise RuntimeError(
            "Convergence warnings occurred; inspect training before final evaluation: "
            + "; ".join(convergence)
        )
    best = search.best_estimator_
    classifier = best.named_steps["classifier"]
    classes = classifier.classes_.tolist()
    candidates = []
    for i, params in enumerate(search.cv_results_["params"]):
        candidates.append(
            {
                "C": params["classifier__C"],
                "class_weight": params["classifier__class_weight"],
                "mean_f1_macro": float(search.cv_results_["mean_test_f1_macro"][i]),
                "std_f1_macro": float(search.cv_results_["std_test_f1_macro"][i]),
                "fold_f1_macro": [
                    float(search.cv_results_[f"split{fold}_test_f1_macro"][i])
                    for fold in range(5)
                ],
                "per_class_recall": {
                    label: float(search.cv_results_[f"mean_test_recall_{label}"][i])
                    for label in classes
                },
                "rank": int(search.cv_results_["rank_test_f1_macro"][i]),
            }
        )
    baseline = DummyClassifier(strategy="most_frequent")
    baseline_cv = cross_val_score(baseline, X_train, y_train, cv=cv, scoring="f1_macro")
    selection = {
        "candidates": candidates,
        "selected": candidates[search.best_index_],
        "best_unweighted": max(
            (r for r in candidates if r["class_weight"] is None),
            key=lambda r: r["mean_f1_macro"],
        ),
        "best_balanced": max(
            (r for r in candidates if r["class_weight"] == "balanced"),
            key=lambda r: r["mean_f1_macro"],
        ),
        "baseline_cv_fold_scores": baseline_cv.tolist(),
        "baseline_cv_mean": float(baseline_cv.mean()),
        "convergence_warning_count": len(convergence),
        "corrective_action": "None needed; no convergence warnings.",
        "other_warnings": [str(item.message) for item in observed_warnings],
    }

    # Refit only the chosen configuration on each training fold to audit scaler isolation.
    audits = []
    for fold, (fit_ids, validation_ids) in enumerate(folds):
        fitted = clone(best).fit(X_train[fit_ids], y_train[fit_ids])
        scaler = fitted.named_steps["scaler"]
        assert np.allclose(scaler.mean_, X_train[fit_ids].mean(axis=0), atol=1e-12)
        audits.append(
            {
                "fold": fold + 1,
                "fit_row_indices": train_ids[fit_ids].tolist(),
                "validation_row_indices": train_ids[validation_ids].tolist(),
                "scaler_samples_seen": int(scaler.n_samples_seen_),
                "scaler_mean": scaler.mean_.tolist(),
                "validation_macro_f1": float(
                    f1_score(
                        y_train[validation_ids],
                        fitted.predict(X_train[validation_ids]),
                        average="macro",
                    )
                ),
            }
        )
    save_json(reports / "fold_audit.json", audits)
    save_json(reports / "cv_results.json", selection)
    split = {
        "train_indices": train_ids.tolist(),
        "test_indices": test_ids.tolist(),
        "random_state": SEED,
        "train_counts": dict(Counter(y_train)),
        "test_counts": dict(Counter(y_test)),
    }
    save_json(model_dir / "split.json", split)
    joblib.dump(best, model_dir / "pipeline.joblib", compress=3)
    export = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "feature_order": FEATURES,
        "classes": classes,
        "mean": best.named_steps["scaler"].mean_.tolist(),
        "scale": best.named_steps["scaler"].scale_.tolist(),
        "coefficients": classifier.coef_.tolist(),
        "intercepts": classifier.intercept_.tolist(),
        "probability_method": "multinomial_softmax",
    }
    save_json(model_dir / "portable_model.json", export)
    metadata = {
        "model_version": MODEL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "feature_order": FEATURES,
        "feature_schema": {
            name: {
                "min": RANGES[name][0],
                "max": RANGES[name][1],
                "unit": dataset["card"]["feature_schema"][name]["unit"],
            }
            for name in FEATURES
        },
        "class_order": classes,
        "acceptance_threshold": ACCEPTANCE_THRESHOLD,
        "threshold_selection": "Fixed at 0.60 in config.py before fitting or final test evaluation; no test-based tuning.",
        "confidence_type": CONFIDENCE_TYPE,
        "seed": SEED,
        "split": split,
        "cv": {"folds": 5, "shuffle": True, "random_state": SEED},
        "selected_hyperparameters": search.best_params_,
        "best_cv_macro_f1": float(search.best_score_),
        "scaler_samples_seen": int(best.named_steps["scaler"].n_samples_seen_),
        "classifier_iterations": classifier.n_iter_.tolist(),
        "majority_class": Counter(y_train).most_common(1)[0][0],
        "dataset": {key: value for key, value in dataset.items() if key != "card"},
        "pipeline_sha256": hashlib.sha256(
            (model_dir / "pipeline.joblib").read_bytes()
        ).hexdigest(),
        "dependencies": {
            package: version(package)
            for package in [
                "scikit-learn",
                "numpy",
                "scipy",
                "joblib",
                "threadpoolctl",
                "matplotlib",
            ]
        },
        "training_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "event_order": [
            "validate supplied data",
            "reserve stratified test set",
            "training-only CV/search and scaler audits",
            "select/refit and save model with predeclared threshold",
            "evaluate held-out test rows",
        ],
    }
    save_json(model_dir / "metadata.json", metadata)

    # Selection and the threshold are now fixed. This is the first test evaluation.
    probabilities = best.predict_proba(X_test)
    predicted = best.predict(X_test)
    metrics = evaluate_predictions(
        y_test, predicted, probabilities, classes, ACCEPTANCE_THRESHOLD
    )
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)
    baseline_metrics = {
        "accuracy": float(accuracy_score(y_test, baseline_predictions)),
        "macro_f1": float(
            f1_score(y_test, baseline_predictions, average="macro", zero_division=0)
        ),
    }
    save_json(
        reports / "metrics.json",
        {"selected_model": metrics, "baseline": baseline_metrics},
    )
    coefficients = {}
    for label, row, intercept in zip(classes, classifier.coef_, classifier.intercept_):
        positive = [
            (name, float(value)) for name, value in zip(FEATURES, row) if value > 0
        ]
        negative = [
            (name, float(value)) for name, value in zip(FEATURES, row) if value < 0
        ]
        coefficients[label] = {
            "coefficients": dict(zip(FEATURES, row.tolist())),
            "intercept": float(intercept),
            "strongest_positive": (
                max(positive, key=lambda item: item[1]) if positive else None
            ),
            "strongest_negative": (
                min(negative, key=lambda item: item[1]) if negative else None
            ),
        }
    save_json(reports / "global_coefficients.json", coefficients)

    from agent import ArchetypeAgent

    portable = ArchetypeAgent()
    native = ArchetypeAgent(backend="sklearn")
    all_probabilities = best.predict_proba(X)
    reloaded = joblib.load(model_dir / "pipeline.joblib")
    portable_probabilities = np.array(
        [portable.predict_vector(row)[2] for row in vectors]
    )
    consistency = {
        "rows_checked": len(y),
        "joblib_probabilities_identical": bool(
            np.array_equal(all_probabilities, reloaded.predict_proba(X))
        ),
        "max_portable_probability_error": float(
            np.max(np.abs(all_probabilities - portable_probabilities))
        ),
        "portable_predicted_classes_identical": bool(
            np.array_equal(
                np.argmax(all_probabilities, axis=1),
                np.argmax(portable_probabilities, axis=1),
            )
        ),
    }
    assert (
        consistency["joblib_probabilities_identical"]
        and consistency["max_portable_probability_error"] < 1e-10
    )
    save_json(evidence / "model_consistency.json", consistency)
    save_json(
        model_dir / "consistency_vectors.json",
        [
            {"features": row, "probabilities": prob.tolist()}
            for row, prob in zip(vectors, all_probabilities)
        ],
    )

    def request(vector, request_id):
        return {
            "schema_version": SCHEMA_VERSION,
            "request_id": request_id,
            "features": dict(zip(FEATURES, [float(v) for v in vector])),
        }

    mistakes = []
    for local in np.flatnonzero(predicted != y_test)[:5]:
        mistakes.append(
            {
                "row_index": int(test_ids[local]),
                "features": dict(zip(FEATURES, X_test[local].tolist())),
                "true_label": str(y_test[local]),
                "predicted_label": str(predicted[local]),
                "confidence": float(probabilities[local].max()),
                "accepted": bool(probabilities[local].max() >= ACCEPTANCE_THRESHOLD),
            }
        )
    save_json(reports / "misclassifications.json", mistakes)
    local_examples = []
    for local in np.flatnonzero(probabilities.max(axis=1) >= ACCEPTANCE_THRESHOLD)[:2]:
        local_examples.append(
            {
                "row_index": int(test_ids[local]),
                "true_label": str(y_test[local]),
                "request": request(X_test[local], f"explain-test-{test_ids[local]}"),
                "response": portable.predict(
                    request(X_test[local], f"explain-test-{test_ids[local]}")
                ),
            }
        )
    save_json(reports / "local_explanations.json", local_examples)
    make_figures(
        reports, metrics, y_test, probabilities, classifier.coef_, FEATURES, candidates
    )
    write_report(
        reports,
        metadata,
        metrics,
        baseline_metrics,
        selection,
        coefficients,
        local_examples,
        mistakes,
    )
    from scripts.make_requests import make_requests

    make_requests(vectors, targets, train_ids.tolist(), portable)
    print(f"Selected {search.best_params_}; CV macro F1={search.best_score_:.6f}")
    print(
        f"Test accuracy={metrics['accuracy']:.6f}; macro F1={metrics['macro_f1']:.6f}; macro OvR AUC={metrics['macro_ovr_auc']:.6f}"
    )
    print(
        f"Acceptance coverage={metrics['coverage']:.6f}; accepted accuracy={metrics['accepted_accuracy']}"
    )
    print(
        "Saved pipeline, portable parameters, reports, figures, requests, and consistency checks."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    train()
