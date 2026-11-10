"""Run many JSON requests in one command, without retraining."""

import argparse
import json
from pathlib import Path
import platform
import sys

from agent import ArchetypeAgent, ModelError, predict_request
from config import ROOT
from predict import read_request


def run_batch(folder, backend="portable", output_dir=None):
    folder = Path(folder).resolve()
    paths = sorted(folder.glob("*.json"))
    if not paths:
        raise ValueError("No JSON request files found in the selected folder.")
    canonical = folder == (ROOT / "samples/requests").resolve()
    expected = (
        json.loads((ROOT / "samples/expected.json").read_text()) if canonical else {}
    )
    agent = ArchetypeAgent(backend=backend)
    output = (
        Path(output_dir)
        if output_dir is not None
        else ROOT / "local_results" / f"batch_{backend}"
    )
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in paths:
        request, response = read_request(path)
        if response is None:
            response = agent.predict(request)
        expectation = expected.get(path.name)
        passed = expectation is None or (
            response["status"] == expectation["status"]
            and (
                expectation["error_code"] is None
                or (
                    response["error"]
                    and response["error"]["code"] == expectation["error_code"]
                )
            )
        )
        (output / path.name).write_text(
            json.dumps(response, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        rows.append(
            {
                "input": path.name,
                "expected": expectation,
                "passed": bool(passed),
                "response": response,
            }
        )
        verdict = ("PASS" if passed else "FAIL") if canonical else "DONE"
        print(
            f"{verdict:4} {path.name:28} {response['status']:14} category={response['category']} confidence={response['confidence']}"
        )
    summary = {
        "backend": backend,
        "cases": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "expectations_available": canonical,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "results": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        f"\nBatch complete: {summary['passed']}/{summary['cases']} checks passed. Responses saved to {output}"
    )
    if not canonical:
        print(
            "Custom requests have no expected labels; completion does not mean accepted or correct classification."
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=ROOT / "samples/requests")
    parser.add_argument("--backend", choices=["portable", "sklearn"], default="sklearn")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        result = run_batch(args.folder, args.backend, args.output_dir)
    except (OSError, ValueError, ModelError) as error:
        print(f"Batch could not start: {error}", file=sys.stderr)
        return 2
    return 0 if result["passed"] == result["cases"] else 1


if __name__ == "__main__":
    sys.exit(main())
