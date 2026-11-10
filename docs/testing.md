# Testing

## Commands

Run `python -S -m unittest discover -s tests -v` for standard-library checks.
The native-model tests are explicitly skipped when dependencies are unavailable.
Run the same command without `-S`, using a Python environment with the pinned
requirements, for all 51 tests. `python batch.py --backend portable` checks all
22 bundled request outcomes; omit the backend flag for sklearn.

## Coverage

- Required features, inclusive bounds, fixed feature order, IDs, and schemas.
- Boolean/string/null/nonfinite inputs, duplicate keys, and extra fields.
- Real uncertainty and the deterministic 0.60 acceptance boundary.
- Signed local contributions and reconstruction of the complete linear score.
- Missing/corrupt model files, bad numeric types, schema/format mismatches,
  truncated joblib data, CLI stdin, and output-write failures.
- Interactive/automatic readable output and optional JSON output.
- Train/test/fold disjointness, fold scaler means, and unchanged split preview.
- Complete pipeline persistence and portable/sklearn agreement on 1,000 rows.
- Optional wheel-cache checksums, missing files, and unsafe paths.

## Evidence and environments

Before changes, all 31 imported tests passed on Windows/Python 3.12.14.
The saved-model fix raised the suite to 40 passing tests; the readable runner
added five, wheel-cache handling added four, and split preview added two.
The final verification record is `../evidence/portfolio-verification.json`.
It records executed checks and interpreter versions rather than assumed results.

GitHub Actions checks Ubuntu/Python 3.11 and 3.13, Windows/Python 3.13, a
standard-library-only run, and formatting. One job also retrains and tests the
new artifacts. See the repository's Actions page for actual run outcomes.
Configuration alone is not evidence that a workflow passed.

Physical Raspberry Pi testing and macOS execution are not claimed. On the Pi,
run `scripts/pi_check.py --with-sklearn` using the installed environment to
record its real board and both backends.

## Reproducibility limits

Rows, labels, hashes, split seed, folds, search grid, and dependency pins are
fixed. Training timestamps and timing values naturally change on each run.
Different BLAS/platform implementations can produce tiny coefficient differences;
portable/native comparison uses an explicit 1e-10 tolerance. A consistency check
proves matching implementations on those vectors, not real-world validity.

Saved joblib models require trusted input and matching recorded native versions.
The portable path has no third-party imports. Git attributes preserve the exact
source CSV/card bytes so their checksums survive clones on different platforms.
