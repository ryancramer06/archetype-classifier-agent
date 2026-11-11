# Development log

This log records repository preparation beginning November 10, 2026.
Git records the actual commit times. Earlier work is represented by one import;
no earlier milestones or dates have been reconstructed.

## Starting point

The supplied ZIP already contained the six-feature classifier, unchanged
synthetic dataset, leakage-safe training, weighted/unweighted model search,
saved sklearn pipeline, portable export, evaluation reports, 22 requests,
31 tests, and Raspberry Pi helpers. These are existing functionality.
The source archive has no Git metadata. Its checksum is recorded in
`source-snapshot.json`; the archive is preserved outside this working tree.
Codex assisted with the original project and this repository preparation.

Baseline verification before changes: all 31 existing tests passed on Windows
with Python 3.12.14 and the pinned scientific packages. The five original Pi
wheel hashes also matched their supplied manifest. Neither check is a physical
Raspberry Pi run.

## Repository organization and readability

- Added ignore rules for local environments, generated results, credentials,
  editor files, and optional vendor wheels.
- Preserved exact dataset bytes through Git attributes because the loader
  verifies checksums.
- Kept source modules at the root: nine small application modules do not need
  a packaging layer just to run from a clone.
- Separated code licensing from the supplied dataset's educational-use terms.
- Removed presentation notes, grading logistics, and redundant captured
  responses. The import commit still preserves the original files.
- Added a pinned formatter and applied a consistent Python layout.

Validation: Black 26.10.1 formatted all 14 Python files. AST comparisons for
every file confirmed that formatting did not change executable structure.
All 31 existing tests passed again after formatting (Python 3.12.14, Windows).
The formatter's parallel CLI stalled in this restricted Windows environment;
its Python API was used sequentially instead. No stalled run is counted as a
successful check.

## Saved-model validation

Added nine regression/CLI tests. Before the fix, this test run reported eight
failed assertions (including subtests) and two errors: oversized integers and
an empty pipeline could escape the normal unavailable response. Boolean model
parameters, Boolean thresholds, incompatible schemas, and unsupported
probability methods were also accepted incorrectly.

The loader now checks numeric types, dimensions, finite values, schema versions,
and the supported softmax export format. Truncated/unreadable native model
packages are converted to the existing `unavailable` contract. This does not
make loading arbitrary pickle files safe; joblib artifacts must still be trusted.
Added CLI checks for stdin input, missing model files, and output-write failure.
All 40 tests passed after the fix on Windows/Python 3.12.14.

## Readable example runner

The example runner now shows the status, category, model probability, threshold,
signed contributions, and useful errors before technical metadata. `--format
json` retains the complete responses. Interactive pauses remain optional through
`--auto`. EOF, cancellation, and missing model packages have clear exit behavior.
Five new tests passed, covering summary/JSON output, five interactive pauses,
end-of-input, and missing artifacts. The original five inputs and model decisions
are unchanged.

## Installation from a clone

Vendor wheels are intentionally outside Git. The Pi setup helper now installs
the pinned packages from the package index when a clone has no wheel cache;
an existing offline cache is still checked before use. The checksum checker
also reports missing files clearly and rejects paths outside its cache folder.
Four new tests cover valid, corrupt, missing, and unsafe cache entries.

## Inspecting the split without training

Added `train.py --preview-split` to show row counts, per-class counts, and overlap
without fitting or writing model files. Previewing and training share one split
function, preserving the existing seed and row allocation. Two integration tests
compare its output with the saved split and check that model files remain intact.

## Continuous integration

Added GitHub Actions checks for standard-library inference, full saved-model
tests on Windows and Ubuntu/Python 3.11 and 3.13, and consistent formatting.
The Ubuntu/Python 3.13 job also retrains and checks the newly written artifacts.
Workflow results are reported separately from the local tests once GitHub runs
them; adding a workflow alone is not a passing CI result.

## Software documentation and model naming

Replaced machine-specific instructions with clone-based Windows/Linux commands
and documented the request contract, result interpretation, architecture,
installation, tests, Pi setup, and remaining limitations. Removed grading and
class-presentation material while retaining dataset attribution and usage terms.
The software/model identifier is now `archetype-lr-1.1.0`; the model family,
features, split, grid, labels, and fixed threshold are unchanged. Regenerating
artifacts verifies that these preparation changes preserve the measured task.

Retraining completed with all eight candidates and five folds. The held-out
results remained accuracy=0.94, macro F1=0.9125411715, and macro OvR
AUC=0.9710434165. All 51 tests then passed, and evaluate.py reproduced the
saved held-out metrics without fitting. The native pipeline bytes and portable
numeric parameters remain equivalent to the imported fitted model.

The first CI submission was rejected because an unquoted install command
contained a YAML-significant colon. The command is now quoted; the cache-path
expression also lives at step scope. PyYAML 6.0.3 successfully parsed the
corrected workflow before it was pushed. This was a real preparation bug,
not an invented earlier development event.

## Published clone verification

A fresh clone at `6031c7c` installed the pinned requirements on Windows with
Python 3.13.5. All 51 tests passed, both batches passed 22/22 cases, and the
held-out evaluation reproduced the saved metrics. Both example modes and the
split preview ran successfully. Python 3.11 standard-library checks passed 40
tests with 11 native checks skipped. The dataset and fitted parameters matched
the original archive. All five jobs in GitHub Actions run 42893835531 passed,
including the Ubuntu training run and the Windows test job. Details are in
`evidence/portfolio-verification.json`.
