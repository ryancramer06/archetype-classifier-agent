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
