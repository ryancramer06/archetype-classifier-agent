# Cramer ML Archetype Agent

CSCI 317H - Assignment 3

This project uses the instructor's **unchanged synthetic dataset** to learn four
fictional demonstration categories. It accepts a JSON request containing six
synthetic numeric features and returns `ok`, `uncertain`, or `invalid_input`, a
model probability, and signed local feature contributions. Model/package failures
return `unavailable`. It does not accept photos or assess real people.

The required training model is a saved scikit-learn `StandardScaler` +
`LogisticRegression` pipeline. The project also includes a checked JSON export of
the same fitted parameters for optional standard-library inference. No prediction
request retrains a model or fits a new scaler.

**Your Pi setup:** 64-bit Raspberry Pi OS, Python 3.13.5.
**Project folder on the Pi:** `/home/sju/Desktop/cramer-archetype-agent`.

Start with `presentation/ORAL_EXAM_GUIDE.md` for all ten questions and a demo plan.
The full measured report is `reports/evaluation.md`; `reports/index.html` displays
the summary and plots offline. No GitHub repository or web server is required.

## Put the project on the SD card / Pi

Extract `cramer-archetype-agent.zip`. Copy the **entire resulting folder** to the
Pi's Desktop, so `predict.py` is at:

```text
/home/sju/Desktop/cramer-archetype-agent/predict.py
```

Use the SD-card/file-transfer method that already works with your Pi setup.
Preserve `models/`, `data/`, `samples/`, `reports/`, and `pi_wheels/` beside the
Python files. Copy this folder, rather than a Windows virtual environment.

On the running Pi, verify the folder before setup:

```bash
cd /home/sju/Desktop/cramer-archetype-agent
python3 --version
ls predict.py models/pipeline.joblib models/portable_model.json
```

If you copied the ZIP to the Desktop instead, extract it there:

```bash
cd /home/sju/Desktop
python3 -m zipfile -e cramer-archetype-agent.zip .
cd /home/sju/Desktop/cramer-archetype-agent
```

## Setup on your Raspberry Pi (standard saved-pipeline method)

The five core dependency wheels for **CPython 3.13, Linux ARM64** are included.
For your specified setup, the following installation reads them locally and does
not need a network connection or compile scientific libraries:

```bash
cd /home/sju/Desktop/cramer-archetype-agent
python3 -m venv .venv
.venv/bin/python -m pip install --no-index --find-links pi_wheels -r requirements-pi.txt
.venv/bin/python predict.py --request samples/requests/valid_explorer.json
.venv/bin/python batch.py
.venv/bin/python -m unittest discover -s tests -v
```

Expected batch result: **22/22 checks passed**. Expected tests: **31 tests, OK**.
The default CLI backend is `sklearn`, using the complete saved pipeline. Activation
is optional because these commands use the virtual environment's Python directly.
The bundled wheels target your stated 64-bit/Python 3.13 setup, not Windows/32-bit Pi OS.

The equivalent setup helper is:

```bash
cd /home/sju/Desktop/cramer-archetype-agent
bash scripts/setup_pi.sh --core
```

This also checks the exported and sklearn predictions. It creates no system service
and does not run training. The operating system must provide Python's venv module.

## Immediate package-free fallback

These commands use Python's standard library and the exported fitted parameters.
They work without venv, pip, NumPy, or scikit-learn installation:

```bash
cd /home/sju/Desktop/cramer-archetype-agent
python3 predict.py --backend portable --request samples/requests/valid_explorer.json
python3 batch.py --backend portable
python3 -m unittest discover -s tests -p 'test_portable.py' -v
python3 demo.py --backend portable
```

Expected: **22/22 batch cases** and **23 portable tests**. This is an optional
deployment export, not a separately trained model. Its scaler means/scales,
coefficients, intercepts, and class order come from the same sklearn pipeline.
Full-model tests are separate and require the core packages.

## Run normal, uncertain, and invalid requests

Run these from the project folder after the standard setup:

```bash
.venv/bin/python predict.py --request samples/requests/valid_explorer.json
.venv/bin/python predict.py --request samples/requests/uncertain.json
.venv/bin/python predict.py --request samples/requests/missing_feature.json
.venv/bin/python predict.py --request samples/requests/boolean_feature.json
.venv/bin/python predict.py --request samples/requests/above_range.json
```

The valid example returns `ok`, a fictional category, the top uncalibrated model
probability, and three signed feature contributions. The actual uncertain example
is a valid training row below the fixed 0.60 acceptance threshold; it returns
`category: null`. Invalid inputs also return `category: null` with an error code.

Single-request exit codes: 0 `ok`, 1 `uncertain`, 2 `invalid_input`, 3 `unavailable`,
4 optional output-write failure. Nonzero codes are expected for rejection demos.
Batch mode returns zero when the expected rejections occur correctly.

## Try every included request individually

The four accepted examples come from the training partition; they are not claims
that every request with those feature names should have those labels.

```bash
.venv/bin/python predict.py --request samples/requests/valid_explorer.json
.venv/bin/python predict.py --request samples/requests/valid_guardian.json
.venv/bin/python predict.py --request samples/requests/valid_creator.json
.venv/bin/python predict.py --request samples/requests/valid_leader.json
.venv/bin/python predict.py --request samples/requests/uncertain.json
.venv/bin/python predict.py --request samples/requests/missing_feature.json
.venv/bin/python predict.py --request samples/requests/boolean_feature.json
.venv/bin/python predict.py --request samples/requests/numeric_string.json
.venv/bin/python predict.py --request samples/requests/null_feature.json
.venv/bin/python predict.py --request samples/requests/below_range.json
.venv/bin/python predict.py --request samples/requests/above_range.json
.venv/bin/python predict.py --request samples/requests/extra_feature.json
.venv/bin/python predict.py --request samples/requests/extra_field.json
.venv/bin/python predict.py --request samples/requests/wrong_schema.json
.venv/bin/python predict.py --request samples/requests/bad_request_id.json
.venv/bin/python predict.py --request samples/requests/missing_request_id.json
.venv/bin/python predict.py --request samples/requests/features_not_object.json
.venv/bin/python predict.py --request samples/requests/request_not_object.json
.venv/bin/python predict.py --request samples/requests/malformed.json
.venv/bin/python predict.py --request samples/requests/nan.json
.venv/bin/python predict.py --request samples/requests/infinity.json
.venv/bin/python predict.py --request samples/requests/duplicate_key.json
```

`valid_*.json` should return `ok`; `uncertain.json` should return `uncertain`;
the remaining requests should return `invalid_input`. The exact expected error
codes are in `samples/expected.json`. All six features are required, numeric,
finite, non-Boolean, and in the supplied ranges. Extra fields/features are rejected.

Use `--backend portable` with `python3` on any of these commands for the package-free method.

## Test lots of requests in one command

```bash
cd /home/sju/Desktop/cramer-archetype-agent
.venv/bin/python batch.py
.venv/bin/python batch.py --backend portable --output-dir local_results/portable_batch
```

Each command processes all 22 bundled cases and compares expected status/error
codes. PASS for invalid input means the rejection was correct. Results and the
summary JSON go into `local_results/`. The standard batch also exercises actual
saved sklearn predictions, not hard-coded categories.

To process your own folder of synthetic JSON requests:

```bash
.venv/bin/python batch.py --folder samples/requests --output-dir local_results/my_batch
```

Replace `samples/requests` with your folder path, in quotes if it contains spaces.
Custom folders have no expected labels: completion means responses were produced,
not that every input was accepted or every fictional label is correct.

## Save a response or demonstrate unavailable behavior

```bash
.venv/bin/python predict.py --request samples/requests/valid_explorer.json --output local_results/single_response.json
.venv/bin/python predict.py --request samples/requests/valid_explorer.json --model-dir models/missing
```

The second command deliberately uses a nonexistent model folder and returns
`unavailable`. It does not alter the real saved model package.

## Oral-exam demonstration and startup commands

```bash
cd /home/sju/Desktop/cramer-archetype-agent
.venv/bin/python demo.py
```

Press Enter between the valid prediction, actual uncertainty, Boolean rejection,
extra target-feature rejection, and out-of-range example. To run without pauses:

```bash
.venv/bin/python demo.py --auto
```

The folder-aware launcher also works:

```bash
bash /home/sju/Desktop/cramer-archetype-agent/run_demo.sh
```

It uses `.venv/bin/python` if installed; otherwise it runs the portable demo with
`python3`. Running it with `bash` avoids relying on executable permissions surviving
an SD-card/ZIP transfer. This is a manual startup command, not an autostart service.

## Record checks on your actual Pi

```bash
cd /home/sju/Desktop/cramer-archetype-agent
.venv/bin/python scripts/pi_check.py --with-sklearn
```

Or, for the package-free method:

```bash
python3 scripts/pi_check.py
```

This records Python, machine architecture, the Pi's device-tree model (when
available), timings, all 1,000 golden-vector probability comparisons, and request
batch results in `local_results/pi_check.json`. The `raspberry_pi_hardware_detected`
field distinguishes a real Pi run from desktop compatibility checks.
Physical Pi results are not claimed in the supplied desktop evidence.

## Training and held-out evaluation

The saved model is already supplied; prediction and the demo do not need training.
To reproduce training and regenerate plots, install the full requirements first
(plotting packages are additional downloads; only core Pi inference wheels are bundled):

```bash
cd /home/sju/Desktop/cramer-archetype-agent
.venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
.venv/bin/python train.py
.venv/bin/python evaluate.py
.venv/bin/python -m unittest discover -s tests -v
```

`train.py` validates the original data, reserves the stratified 80/20 split with
seed 317, performs five-fold training-only search, refits the winner, and evaluates
the untouched test partition. It writes models, reports/figures, request examples,
and save/load/portable consistency evidence. `evaluate.py` reloads that fixed model
and rechecks held-out metrics without fitting or selecting anything.
The predeclared acceptance threshold remains 0.60; it is not chosen from test results.

On the Pi Desktop, open `reports/index.html` to view the saved report summary and
plots. The complete report with every required table and explanations is
`reports/evaluation.md`. No network or plotting installation is needed to view them.

## Windows development commands

The workspace project is currently at the path below. If you extract it elsewhere,
open a terminal inside the extracted folder instead. Use your installed Python 3.12
or 3.13; the explicit path here is the verified runtime on this computer:

```powershell
Set-Location -LiteralPath "C:\Users\ryry5\Documents\Codex\2026-10-05\re\outputs\cramer-archetype-agent"
& "C:\Users\ryry5\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe predict.py --request samples/requests/valid_explorer.json
.\.venv\Scripts\python.exe batch.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Do not install `pi_wheels` on Windows; use requirements.txt through the package
index. Windows wheels and virtual environments are not included in the ZIP.

## Pipeline and folder contents

Training: original CSV/card -> validate -> reserve test rows -> training-fold
scaling and logistic regression -> eight-candidate/five-fold search -> refit winner
-> held-out metrics/ROC and acceptance results -> save complete pipeline and export.

Inference: JSON -> closed schema/range validation -> saved fitted model ->
probabilities -> fixed acceptance rule -> fictional category or null -> local
contributions for accepted predictions -> versioned response and limitations.

```text
cramer-archetype-agent/
  agent.py, predict.py          Validation and saved-model JSON inference
  train.py, evaluate.py         Required model selection and held-out evaluation
  batch.py, demo.py             Many-request tests and oral demonstration
  config.py, data_utils.py      Feature contract, fixed rule, dataset checks
  evaluation_utils.py           Metrics, report, and scientific figures
  data/                        Unchanged supplied CSV and dataset card
  models/                      Full fitted pipeline, portable export, metadata, split
  pi_wheels/                   Five pinned CPython 3.13 Linux ARM64 dependency wheels
  samples/                     Valid/uncertain/invalid JSONs, expected outcomes, provenance
  reports/                     Actual metrics, CV results, explanations, plots, report
  tests/                       31 full tests; 23 work with the standard library
  scripts/                     Pi setup/checks and request regeneration
  presentation/                Answers to all ten guiding questions
  evidence/                    Captured desktop verification results
```

## Known limitations and responsible use

All records, measurement ranges, and labels are fictional teaching examples.
Synthetic accuracy is not evidence of personality, identity, health, demographics,
ability, or future behavior. The fixed threshold is an abstention rule, not an
out-of-distribution detector or calibration guarantee. Correlated features and
linear model assumptions limit interpretation; contributions are score terms,
not causal effects or probability-point changes.

The complete trusted joblib package is included for sklearn use; only load joblib
packages you trust because its persistence format is pickle-based. The portable
backend reads JSON instead. No personal data, photos, credentials, or network API
keys are required. Dependencies and AI implementation assistance are credited in
`docs/SOURCES.md`. The deployment target is your Pi, but physical-device validation
must be recorded there with the provided check command.
