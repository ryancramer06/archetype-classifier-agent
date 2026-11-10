# Sources and acknowledgments

## Dataset

Dr. Srikanth Vemula supplied the CSCI 317H Synthetic Archetype Dataset v1.0.0.
The original CSV and dataset card are retained unchanged. Their permitted-use
statement is "Coursework and educational demonstration only." See
[the data notices](../THIRD_PARTY_NOTICES.md) and [dataset description](DATASET.md).
The source PDF and slide decks are not redistributed in this repository.

## Implementation references

- [Logistic regression](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [StandardScaler](https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [Pipelines and data leakage](https://scikit-learn.org/1.7/common_pitfalls.html)
- [GridSearchCV](https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [StratifiedKFold](https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.StratifiedKFold.html)
- [ROC/AUC](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.roc_auc_score.html)
- [Model persistence](https://scikit-learn.org/1.7/model_persistence.html)
- [Python standard library](https://docs.python.org/3.13/library/index.html)
- [Matplotlib](https://matplotlib.org/stable/users/index.html)
- [GitHub Actions for Python](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)

Codex assisted with implementation, tests, documentation, and repository
preparation. The import and subsequent changes are distinguished in the
[development log](development-log.md).
