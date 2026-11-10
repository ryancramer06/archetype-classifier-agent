# Assignment 3 and requested deployment coverage

| Requirement | Implementation / evidence |
| --- | --- |
| Supplied unchanged data/card, 1000 rows and exact counts | data/, data_utils.py validation, metadata hashes, docs/DATASET.md |
| Six approved predictors; target/IDs excluded | config.FEATURES, data_utils.py, closed inference schema |
| Stratified 80/20 split with seed 317 | train.py and models/split.json |
| Fold-fitted StandardScaler + logistic regression | Pipeline inside GridSearchCV; reports/fold_audit.json |
| Most-frequent baseline | DummyClassifier trained on training rows, reports/metrics.json |
| Weighted/unweighted comparison | All 8 candidates, identical 5 folds; per-class recalls and macro F1 |
| Required C grid, CV fold scores, mean/SD | reports/cv_results.json and evaluation.md |
| Selection based on training CV; no test-based threshold | config.ACCEPTANCE_THRESHOLD fixed 0.60; training event order in metadata |
| Held-out confusion matrix, per-class metrics, macro F1, baseline | reports/metrics.json and confusion_matrix.png |
| Per-class ROC/AUC and macro OvR AUC | roc_curves.png, explicit classes_ order, report tables |
| At least two misclassifications where available | Five examples in misclassifications.json; report summary |
| Global coefficients and two local explanations | global_coefficients.json/png, local_explanations.json and report |
| Correct signed z*beta terms | agent.py, score-reconstruction tests; no probability-point or causal claim |
| Invalid/uncertain/ok, required feature checks | 22 supplied JSON cases; finite/bool/range/extra/schema validation |
| Model packaging and prediction consistency | Complete pipeline.joblib, metadata, portable export; all 1000 probability comparisons |
| Runnable commands, dependencies, tests | README, pinned requirements, 31 full / 23 portable tests |
| Raspberry Pi OS 64-bit, Python 3.13.5 | Pi core wheelhouse, explicit Desktop commands, hardware-check script |
| Unique student folder | cramer-archetype-agent at /home/sju/Desktop/cramer-archetype-agent |
| Classroom oral questions | presentation/ORAL_EXAM_GUIDE.md; actual metrics included |
| Folder and ZIP, no GitHub requirement | Complete local project and verified submission ZIP |
| Responsible framing and citations | Supplied synthetic data only, no images/personal measurements; SOURCES.md |

The Pi hardware-check script records actual platform/device-tree information.
The supplied verification ran on desktop Python environments and is labeled as
such. No physical Pi execution or successful SD-card transfer is fabricated.
