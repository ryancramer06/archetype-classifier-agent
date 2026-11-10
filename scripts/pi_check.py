"""Record actual runtime/platform and check the saved portable predictor."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import ArchetypeAgent
from batch import run_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "local_results/pi_check.json"
    )
    parser.add_argument(
        "--with-sklearn",
        action="store_true",
        help="Also verify the complete joblib pipeline; requires the pinned core packages",
    )
    args = parser.parse_args()
    agent = ArchetypeAgent()
    vectors = json.loads((ROOT / "models/consistency_vectors.json").read_text())
    started = perf_counter()
    error = 0.0
    for case in vectors:
        probabilities = agent.predict_vector(case["features"])[2]
        error = max(
            error, max(abs(a - b) for a, b in zip(probabilities, case["probabilities"]))
        )
    elapsed = (perf_counter() - started) * 1000
    board_path = Path("/proc/device-tree/model")
    board = board_path.read_text().strip("\x00\n") if board_path.is_file() else None
    batch = run_batch(
        ROOT / "samples/requests", output_dir=ROOT / "local_results/pi_requests"
    )
    native = None
    if args.with_sklearn:
        reference = ArchetypeAgent(backend="sklearn")
        native_error = 0.0
        for case in vectors:
            actual = reference.predict_vector(case["features"])[2]
            native_error = max(
                native_error,
                max(abs(a - b) for a, b in zip(actual, case["probabilities"])),
            )
        native_batch = run_batch(
            ROOT / "samples/requests",
            backend="sklearn",
            output_dir=ROOT / "local_results/pi_requests_sklearn",
        )
        native = {
            "max_probability_error": native_error,
            "consistency_passed": native_error < 1e-10,
            "batch_passed": native_batch["passed"],
            "batch_cases": native_batch["cases"],
        }
    report = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "board_model": board,
        "raspberry_pi_hardware_detected": bool(board and "Raspberry Pi" in board),
        "requested_deployment": {
            "folder": str(ROOT),
            "python": "3.13.5",
            "os": "64-bit Raspberry Pi OS",
        },
        "model_version": agent.metadata["model_version"],
        "backend": "portable",
        "consistency_rows": len(vectors),
        "max_probability_error": error,
        "consistency_passed": error < 1e-10,
        "total_probability_check_ms": round(elapsed, 3),
        "mean_ms_per_vector": round(elapsed / len(vectors), 6),
        "batch_passed": batch["passed"],
        "batch_cases": batch["cases"],
        "sklearn_check": native,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["raspberry_pi_hardware_detected"]:
        print(
            "This is a compatibility check on this computer, not physical Raspberry Pi evidence."
        )
    passed = report["consistency_passed"] and batch["passed"] == batch["cases"]
    if native is not None:
        passed = (
            passed
            and native["consistency_passed"]
            and native["batch_passed"] == native["batch_cases"]
        )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
