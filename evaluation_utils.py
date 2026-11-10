"""Classification metrics and figures; used only for evaluation, not inference."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve


def evaluate_predictions(y_true, predictions, probabilities, classes, threshold):
    report = classification_report(y_true, predictions, labels=classes, output_dict=True, zero_division=0)
    accepted = np.max(probabilities, axis=1) >= threshold
    return {"sample_count": len(y_true), "class_order": list(classes),
            "accuracy": float(accuracy_score(y_true, predictions)),
            "macro_f1": float(f1_score(y_true, predictions, labels=classes, average="macro", zero_division=0)),
            "per_class": {label: report[label] for label in classes},
            "confusion_matrix": confusion_matrix(y_true, predictions, labels=classes).tolist(),
            "auc_ovr": {label: float(roc_auc_score(np.asarray(y_true) == label, probabilities[:, i]))
                        for i, label in enumerate(classes)},
            "macro_ovr_auc": float(roc_auc_score(y_true, probabilities, labels=classes, multi_class="ovr", average="macro")),
            "acceptance_threshold": threshold, "accepted_count": int(accepted.sum()),
            "coverage": float(accepted.mean()),
            "accepted_accuracy": float(accuracy_score(np.asarray(y_true)[accepted], np.asarray(predictions)[accepted])) if accepted.any() else None}


def make_figures(folder, metrics, y_test, probabilities, coefficients, features, candidates):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(folder)
    classes = metrics["class_order"]
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(6.5, 5))
    matrix = np.array(metrics["confusion_matrix"])
    image = ax.imshow(matrix, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i,j] > matrix.max()/2 else "black")
    ax.set(xticks=range(4), yticks=range(4), xticklabels=classes, yticklabels=classes,
           xlabel="Predicted fictional category", ylabel="True synthetic label", title="Held-out confusion matrix (all 200 rows)")
    fig.colorbar(image, ax=ax)
    fig.tight_layout(); fig.savefig(folder / "confusion_matrix.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, label in enumerate(classes):
        false_positive, true_positive, _ = roc_curve(np.asarray(y_test) == label, probabilities[:, i])
        ax.plot(false_positive, true_positive, label=f"{label}: AUC={metrics['auc_ovr'][label]:.3f}")
    ax.plot([0,1],[0,1],"k--",alpha=.5,label="Chance ranking")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"Synthetic test ROC: macro OvR AUC={metrics['macro_ovr_auc']:.3f}", xlim=(0,1),ylim=(0,1.02))
    ax.legend(loc="lower right"); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(folder / "roc_curves.png"); plt.close(fig)

    fig, axes = plt.subplots(2,2,figsize=(11,7))
    for ax, label, row in zip(axes.flat, classes, coefficients):
        ax.barh(features, row, color=["#17807b" if v >= 0 else "#c75c48" for v in row])
        ax.axvline(0,color="black",linewidth=.7); ax.set_title(label)
        ax.set_xlabel("Coefficient per standardized feature unit")
    fig.suptitle("Class-specific linear-score coefficients (not causal effects)")
    fig.tight_layout(); fig.savefig(folder / "global_coefficients.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,4.5))
    for weight in [None,"balanced"]:
        rows=[row for row in candidates if row["class_weight"] == weight]
        ax.errorbar([r["C"] for r in rows],[r["mean_f1_macro"] for r in rows],
                    yerr=[r["std_f1_macro"] for r in rows],marker="o",capsize=3,label=str(weight))
    ax.set_xscale("log"); ax.set(xlabel="C (inverse regularization strength)",ylabel="Mean five-fold macro F1",title="Training-only cross-validation (error bars: fold SD)")
    ax.legend(title="class_weight"); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(folder / "cv_comparison.png"); plt.close(fig)


def write_report(folder, metadata, metrics, baseline, selection, coefficients, local_examples, mistakes):
    from html import escape
    folder=Path(folder)
    classes=metrics["class_order"]
    best=selection["selected"]
    lines=["# Cramer ML Archetype Agent - evaluation report", "",
           "All records and labels are fictional. Results describe this synthetic task only.", "",
           "## Dataset and split", "",
           "Source: Dr. V's instructor-provided dataset; coursework/educational demonstration only.",
           "All 1,000 original rows and labels are retained. The generator used make_classification",
           "with seed 317, four informative and one redundant feature, one cluster per class,",
           "flip_y=0.04 and class_sep=1.1. Class IDs map to fictional narrative names.",
           "Tanh-based bounded range mapping does not replace fold-fitted StandardScaler.", "",
           f"Dataset SHA-256: `{metadata['dataset']['csv_sha256']}`.",
           f"Card SHA-256: `{metadata['dataset']['card_sha256']}`.",
           "Split: stratified 80/20, random_state=317; train=800, test=200.",
           "Internal row positions audit the split and are never predictors.", "",
           "| Class | All | Train | Test |", "| --- | ---: | ---: | ---: |"]
    for label in classes:
        lines.append(f"| {label} | {metadata['dataset']['class_counts'][label]} | {metadata['split']['train_counts'][label]} | {metadata['split']['test_counts'][label]} |")
    lines += ["", "Feature definitions, units, and ranges are in data/dataset_card.json and docs/DATASET.md.",
              "The CSV has no missing values or duplicate feature rows. Invalid inference values are rejected.", "",
              "## Leakage prevention and model selection", "",
              "The test rows are reserved before preprocessing/model selection. GridSearchCV fits",
              "StandardScaler + LogisticRegression independently inside each training fold. The",
              "same five shuffled stratified folds (seed 317) are used for all eight candidates.",
              "C is [0.01, 0.1, 1.0, 10.0]; class_weight is [None, balanced].",
              "Selection maximizes mean macro F1. No test results choose features, parameters, or threshold.", "",
              "| C | Class weight | Mean CV macro F1 | Fold SD | Creator recall | Leader recall |", "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in selection["candidates"]:
        lines.append(f"| {row['C']} | {row['class_weight']} | {row['mean_f1_macro']:.6f} | {row['std_f1_macro']:.6f} | {row['per_class_recall']['creator']:.6f} | {row['per_class_recall']['leader']:.6f} |")
    lines += ["", f"Selected C={best['C']}, class_weight={best['class_weight']}, mean CV macro F1={best['mean_f1_macro']:.6f} (SD={best['std_f1_macro']:.6f}).",
              "Selected candidate's five fold macro F1 scores: " + ", ".join(f"{value:.6f}" for value in best["fold_f1_macro"]) + ".",
              f"Baseline CV macro F1: {selection['baseline_cv_mean']:.6f}.",
              f"Convergence warnings: {selection['convergence_warning_count']}. Corrective action: {selection['corrective_action']}.",
              "CV means support selection; they are not an independent final performance estimate.", "",
              "### Class-weight experiment", "",
              "For each weighting choice, compare its best C on the same training folds:", "",
              "| Class | Unweighted CV recall | Balanced CV recall | Difference (balanced - unweighted) |",
              "| --- | ---: | ---: | ---: |"]
    for label in classes:
        a=selection["best_unweighted"]["per_class_recall"][label]; b=selection["best_balanced"]["per_class_recall"][label]
        lines.append(f"| {label} | {a:.6f} | {b:.6f} | {b-a:+.6f} |")
    lines += ["", "Balanced weighting uses n_samples/(n_classes*class_count) inside the training fold.",
              "It changes the loss, not the held-out class distribution. Minority recalls can improve",
              "or worsen; weighting is an experiment, not an automatic winning choice. The selected",
              "configuration wins on mean training-CV macro F1. Equal-score ties use GridSearchCV's first candidate.", "",
              "## Held-out results before abstention", "",
              f"Explicit probability/matrix class order: {classes}.", "",
              "| Model | Accuracy | Macro F1 |", "| --- | ---: | ---: |",
              f"| Majority-class baseline ({metadata['majority_class']}) | {baseline['accuracy']:.6f} | {baseline['macro_f1']:.6f} |",
              f"| Selected logistic regression | {metrics['accuracy']:.6f} | {metrics['macro_f1']:.6f} |", "",
              "The baseline always returns the training set's most frequent class. Macro F1 weights",
              "all classes equally, exposing minority errors hidden by an imbalanced accuracy score.", "",
              "| Class | Precision | Recall | F1 | Support |", "| --- | ---: | ---: | ---: | ---: |"]
    for label in classes:
        row=metrics["per_class"][label]
        lines.append(f"| {label} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1-score']:.6f} | {int(row['support'])} |")
    lines += ["", "Confusion matrix: rows=true labels, columns=predicted labels, in the class order above.", "", "```text"]
    lines += [str(row) for row in metrics["confusion_matrix"]]
    lines += ["```", "", "![Confusion matrix](confusion_matrix.png)", "", "### ROC/AUC", "",
              "Each class is positive versus the other three. Curves use predict_proba columns in",
              "classifier.classes_ order. AUC measures synthetic discrimination, not classification",
              "accuracy, calibration, or evidence of actual personality traits.", "",
              "| Class | One-vs-rest AUC |", "| --- | ---: |"]
    for label in classes:
        lines.append(f"| {label} | {metrics['auc_ovr'][label]:.6f} |")
    lines += [f"| Macro OvR | {metrics['macro_ovr_auc']:.6f} |", "", "![ROC curves](roc_curves.png)", "",
              "## Acceptance and uncertainty", "",
              f"Threshold={metrics['acceptance_threshold']:.2f}, fixed in config.py before training/test evaluation.",
              "Accept when max predicted probability >= threshold. Otherwise category is null and",
              "status is uncertain. This fixed classroom rule was not optimized against test results.",
              f"Accepted {metrics['accepted_count']}/{metrics['sample_count']} test rows; coverage={metrics['coverage']:.6f}.",
              f"Accepted-case accuracy: {metrics['accepted_accuracy'] if metrics['accepted_accuracy'] is not None else 'undefined (no accepted rows)'}.",
              "Probabilities are uncalibrated model probabilities. Abstention does not reliably detect out-of-distribution inputs.", "",
              "## Global and local explanations", "",
              "Coefficients are per standardized feature unit, not per raw synthetic centimeter.",
              "Positive/negative signs affect the class's linear score and are not causal effects.",
              "The intercept plus all six z_j*beta_kj contributions reconstructs that score.",
              "The response lists the three largest absolute terms. They do not sum to a probability",
              "change, and softmax also depends on competing class scores.", ""]
    for label in classes:
        row=coefficients[label]
        lines.append(f"- {label}: strongest positive={row['strongest_positive']}; strongest negative={row['strongest_negative']}.")
    lines += ["", "Full coefficient table: global_coefficients.json. Two actual local examples follow:", ""]
    for example in local_examples:
        response=example["response"]
        lines += [f"### Test row {example['row_index']} (synthetic true label {example['true_label']})", "", "```json",json.dumps(response,indent=2),"```",""]
    lines += ["![Global coefficients](global_coefficients.png)", "", "## Misclassifications", ""]
    if mistakes:
        for mistake in mistakes:
            lines.append(f"- Test row {mistake['row_index']}: true={mistake['true_label']}, predicted={mistake['predicted_label']}, max probability={mistake['confidence']:.6f}, accepted={mistake['accepted']}.")
        lines.append("Full synthetic vectors and outcomes: misclassifications.json.")
    else:
        lines.append("No held-out misclassifications occurred; therefore two error examples are unavailable.")
    lines += ["", "## Limits and portability", "",
              "The data were generated, bounded, and jointly labeled as fictional clusters; they are not",
              "measurements of people. Arbitrary teaching ranges and noisy labels do not validate biological",
              "relationships. A linear boundary may miss nonlinear structure and performance can shift",
              "on new distributions. No personal, demographic, health, or ability claims are justified.", "",
              "The complete fitted sklearn pipeline is saved. A JSON export of the same scaler, coefficients,",
              "intercepts, and class order supports standard-library inference on Python 3.13.5. Desktop",
              "save/load and portable-equivalence checks are recorded in evidence. Physical Raspberry Pi",
              "runtime results must be generated on the user's hardware; none are claimed here.", ""]
    (folder / "evaluation.md").write_text("\n".join(lines),encoding="utf-8")
    (folder / "index.html").write_text("<!doctype html><html><head><meta charset='utf-8'><title>Cramer archetype results</title><style>body{max-width:1000px;margin:30px auto;font:17px system-ui;color:#18262e;padding:0 20px}img{max-width:100%;height:auto}p{line-height:1.5}section{border-top:1px solid #ccc;margin-top:25px}code{background:#eee;padding:3px}</style></head><body>"
        + "<h1>Cramer ML Archetype Agent</h1><p>Fictional synthetic categories; no personal assessment.</p>"
        + f"<p>Held-out accuracy: <b>{metrics['accuracy']:.3f}</b>; macro F1: <b>{metrics['macro_f1']:.3f}</b>; macro OvR AUC: <b>{metrics['macro_ovr_auc']:.3f}</b>.</p>"
        + f"<p>Majority baseline accuracy: {baseline['accuracy']:.3f}; macro F1: {baseline['macro_f1']:.3f}. Acceptance coverage: {metrics['coverage']:.3f} at threshold {metrics['acceptance_threshold']:.2f}.</p>"
        + "<p>Read <a href='evaluation.md'>the full evaluation report</a> for all tables, assumptions, and limitations.</p>"
        + "".join(f"<section><h2>{escape(title)}</h2><img src='{name}' alt='{escape(title)}'></section>" for title,name in [("Held-out confusion matrix","confusion_matrix.png"),("ROC/AUC","roc_curves.png"),("Training-only model selection","cv_comparison.png"),("Standardized coefficients","global_coefficients.png")])
        + "</body></html>",encoding="utf-8")
