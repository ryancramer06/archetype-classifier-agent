# Sources, permissions, and acknowledgments

## Course materials

Dr. Srikanth Vemula, CSCI 317H Artificial Intelligence, Fall 2026, College of
Saint Benedict and Saint John's University:

- Assignment 3: Machine Learning Archetype Agent, supplied 12-page PDF.
- CSCI 317H Synthetic Archetype Dataset v1.0.0: archetype_synthetic.csv and
  dataset_card.json. Source: Dr. V-generated synthetic data. Permission in the
  card: coursework and educational demonstration only.
- Week 7: Classification Models, Raspberry Pi Imaging, and Evaluation Metrics.
- Week_9.pdf: Machine Learning Fundamentals with Penguins, 38 pages.
- Weeks 1-6: agent contracts, bounded features, validation, and deployment context.
- The ten oral-exam questions supplied by the student.

Dataset Canvas link supplied by the student:
https://canvas.csbsju.edu/courses/27877/files/5393518/download?download_frd=1

Assignment Canvas link supplied by the student:
https://canvas.csbsju.edu/courses/27877/files/5393533?wrap=1

The supplied local files are used rather than replacing or regenerating data.
The dataset and card are included with their educational-use restriction. Course
slide PDFs are referenced rather than redistributed. DESIGN.md maps code to
the relevant slide/page numbers and notes differences from illustrative examples.

## Algorithms and implementation references

- LogisticRegression, class weights, regularization, class order:
  https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html
- StandardScaler: https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.StandardScaler.html
- Pipeline/leakage prevention: https://scikit-learn.org/1.7/common_pitfalls.html
- GridSearchCV: https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.GridSearchCV.html
- StratifiedKFold: https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.StratifiedKFold.html
- train_test_split: https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.train_test_split.html
- DummyClassifier: https://scikit-learn.org/1.7/modules/generated/sklearn.dummy.DummyClassifier.html
- ROC/AUC: https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.roc_auc_score.html
- Persistence and portability limits: https://scikit-learn.org/1.7/model_persistence.html
- Joblib persistence: https://joblib.readthedocs.io/en/stable/persistence.html
- NumPy: https://numpy.org/doc/2.2/
- SciPy: https://docs.scipy.org/doc/scipy-1.15.3/
- Matplotlib: https://matplotlib.org/stable/users/index.html
- Python standard library (argparse/json/csv/pathlib/unittest/hashlib/math):
  https://docs.python.org/3.13/library/index.html
- Raspberry Pi remote file-transfer context:
  https://www.raspberrypi.com/documentation/computers/remote-access.html
- Pinned sklearn wheel release: https://pypi.org/project/scikit-learn/1.7.2/
- Pinned NumPy wheel release: https://pypi.org/project/numpy/2.2.6/
- Pinned SciPy wheel release: https://pypi.org/project/scipy/1.15.3/

The core model follows the assignment and course examples. The portable JSON
export, strict duplicate-key JSON parsing, checksum checks, offline-wheel
packaging, and Pi device-tree inspection are supporting/deployment additions;
they are not claimed as lecture-taught APIs or new learning algorithms.

The wheelhouse retains upstream wheel files unmodified, including their internal
license notices. Version pins and a checksum manifest document their intended
CPython 3.13 ARM64 target. No claim of upstream endorsement is made.

## Assistance and data boundaries

OpenAI Codex assisted with implementation, tests, documentation, evaluation
reporting, and deployment packaging. The student should review and understand
the code and actual results before the oral exam.

All features/labels are synthetic. No face/palm photographs, real personal
measurements, identity models, API keys, or external personal datasets are used.
The previous computer-vision projects remain separate. This project does not
establish scientific validity for archetype themes or personal characteristics.
