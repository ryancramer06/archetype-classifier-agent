# Design and class connections

The required workflow is supervised classification on the supplied fictional
labels. It uses a simple most-frequent baseline and regularized multinomial
logistic regression, not image processing, a neural network, or clustering.

## Covered course concepts

Page references use PDF page order, beginning at 1.

| Project concept | Class source |
| --- | --- |
| Features, labels, predictions, probabilities | Week 7 slides 5-7 |
| Baselines and logistic regression | Week 7 slides 8-9 |
| Reserve test data, overfitting, leakage | Week 7 slides 10-12; Week_9 pages 5, 12-16 |
| Saved scaling/classifier pipeline | Week 7 slide 13; Week_9 pages 6-8 |
| Imbalance, class weighting, per-class recall, macro F1 | Week_9 pages 9-11 |
| Five-fold stratified CV and fold variability | Week_9 pages 12-13 |
| C, regularization, complete-pipeline grid search | Week_9 pages 14-16 |
| Confusion matrix, precision, recall, F1 | Week 7 slides 14-15; Week_9 page 10 |
| Per-class ROC, OvR macro AUC, class order | Week_9 pages 17-20 |
| Standardized class-specific coefficients, softmax | Week_9 pages 21-23 |
| Abstention and explicit uncertainty | Week 7 slide 16; Assignment 3 section 5.8 |
| Pi constraints, versioned artifacts, contracts | Week 7 slides 17-18 |
| Agent interfaces and safe failures | Weeks 1-2 contracts/agent boundaries |

## Where the assignment differs from slideshow examples

- The supplied assignment specifies seed 317 and C=[0.01,0.1,1,10]. Week_9's
  Penguins examples use seed 42 and also show C=100. This project follows the
  assignment, giving eight candidates and 40 CV fits, plus the winner refit.
- Penguins needs an imputer for missing values. This supplied CSV has no missing
  values and the assignment requires rejecting missing inference features, so
  this pipeline has StandardScaler and LogisticRegression only.
- Assignment 3 section 5.7 requests each predicted class's own z_j*beta_kj terms.
  Week_9 page 23 instead illustrates a predicted-versus-runner-up coefficient
  contrast. This project implements the assignment's own-class score explanation,
  with a zero-standardized-feature reference. It does not label it a log-ratio contrast.
- Week_9 pages 24-37 explore unsupervised profiles. Those are a separate learning
  question, not a replacement for this assignment's supervised fictional labels.

## Leakage prevention and selection

The CSV/card are validated and the 80/20 stratified split is made before fitting.
Only the six approved predictors reach the estimator. Neither archetype nor row
positions are model inputs. The pipeline is passed into GridSearchCV so each fold
fits its own scaler on 640 training rows and validates on 160 other training rows.
All candidates use identical shuffled stratified folds. CV macro F1 chooses the
winner; per-class recall is an additional diagnostic on those same folds.

The final scaler sees 800 rows, not the entire 1,000. reports/fold_audit.json records
separate refits of the selected configuration to audit fold scaler means and
sample counts. Those audit fits still use training rows only. The final saved
pipeline is the winner refitted on all 800 training rows.

The threshold is the predeclared constant 0.60. It is not selected to improve the
held-out results. Held-out classification/AUC are computed on all 200 test rows
before abstention; coverage and accepted-case accuracy are separate results.

## Runtime and Pi deployment

The normal CLI backend loads models/pipeline.joblib and calls predict_proba and
the saved scaler's transform. No fit/search/CV function is called by inference.
The model package records its version, dependencies, schema, class order, scaler
parameters, random seeds, threshold, and data/model hashes.

The optional portable backend exports the same fitted mean, scale, coefficients,
intercepts, and class order to JSON. It computes z=(x-mean)/scale, class linear
scores, and a numerically stable softmax using math.exp. This is a deployment
addition, not a newly fitted model or a new scaler. All 1,000 dataset probabilities
are compared with the sklearn reference at tolerance 1e-10; joblib reload outputs
must also match. The standard class-demo commands use the complete sklearn pipeline.

Pi target: 64-bit Raspberry Pi OS, Python 3.13.5, project folder
/home/sju/Desktop/cramer-archetype-agent. The core ARM64 CPython 3.13 wheels allow
offline inference installation. Matplotlib is optional on the Pi unless retraining
and regenerating figures. Desktop verification is not physical Pi evidence.

## Supporting implementation details and references

JSON/file CLI handling (argparse, pathlib, json), persistence (joblib), input/error
tests (unittest/mock/tempfile), and checksum checks (hashlib) are supporting APIs.
The portable parameter export, offline-wheel packaging, and hardware-check script
are deployment additions beyond the lecture examples. Official learning references
are in SOURCES.md. No extra ML model family, resampling, SHAP, calibration, or
real-person data collection is introduced.
