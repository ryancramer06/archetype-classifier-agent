# Cramer Archetype Agent

A small Python classification project with strict JSON input validation,
logistic regression, uncertainty handling, and feature-level explanations.
It learns four fictional labels from a supplied synthetic dataset. The labels
are not assessments of real people.

## Try the saved model

From the project directory, with Python 3.11-3.13:

```bash
python demo.py --backend portable --auto
python predict.py --backend portable --request samples/requests/valid_explorer.json
python batch.py --backend portable
```

The portable backend uses the saved pipeline's fitted parameters and needs only
the standard library. The sklearn backend is available after installing the
pinned dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\python.exe -m pip install -r requirements.txt
# Linux/macOS: .venv/bin/python -m pip install -r requirements.txt
```

Run `train.py` to refit the model, and `evaluate.py` to recheck held-out results.
The existing report is in [reports/evaluation.md](reports/evaluation.md).

## Tests

```bash
python -m unittest discover -s tests -p 'test_portable.py' -v
```

With the pinned scientific dependencies installed, omit `-p` to run the full suite.

## Project notes

- [JSON contract](docs/CONTRACT.md)
- [Dataset description](docs/DATASET.md)
- [Development log](docs/development-log.md)
- [Data and dependency notices](THIRD_PARTY_NOTICES.md)

The code uses the MIT license. The supplied dataset retains its separate
educational-use terms; see the notices before reusing data or fitted artifacts.
