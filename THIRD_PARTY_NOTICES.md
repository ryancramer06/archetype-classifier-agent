# Data and dependency notices

The MIT license covers the project code and project-authored documentation.
It does not relicense the supplied dataset or third-party packages.

## Synthetic dataset and fitted artifacts

The source is Dr. Srikanth Vemula's **CSCI 317H Synthetic Archetype Dataset
v1.0.0**. The unchanged `data/dataset_card.json` states:

> Coursework and educational demonstration only.

The CSV, dataset card, fitted artifacts in `models/`, and dataset-derived
examples/results retain that educational-use boundary. No broader data license
or permission for commercial reuse is asserted by this repository. The dataset
contains generated records and fictional labels, not measurements of people.

## Dependencies

NumPy, SciPy, scikit-learn, joblib, threadpoolctl, Matplotlib, and their
dependencies retain their upstream licenses. See installed package metadata
and the references in [sources](docs/SOURCES.md). Optional local wheel files
are not tracked by this repository; the original archive preserves its wheel
cache and checksum manifest.
