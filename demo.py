"""Short oral-exam demonstration using only the standard-library predictor."""

import argparse
import json

from agent import ArchetypeAgent
from config import ROOT
from predict import read_request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auto", action="store_true", help="Skip pauses")
    parser.add_argument("--backend", choices=["portable", "sklearn"], default="sklearn")
    args = parser.parse_args()
    agent = ArchetypeAgent(backend=args.backend)
    print(
        "CRAMER ARCHETYPE AGENT - fictional synthetic categories, never a personal assessment.\n"
    )
    steps = [
        (
            "valid_explorer.json",
            "Valid synthetic input: inspect category, probability, and signed score contributions.",
        ),
        (
            "uncertain.json",
            "An actual valid training-row example falls below the fixed acceptance threshold.",
        ),
        ("boolean_feature.json", "A Boolean is not accepted as a numeric measurement."),
        (
            "extra_feature.json",
            "The target label cannot be smuggled into the feature vector.",
        ),
        ("above_range.json", "Out-of-range values are rejected before prediction."),
    ]
    for name, note in steps:
        if not args.auto:
            input(f"Press Enter for {name}.")
        request, result = read_request(ROOT / "samples/requests" / name)
        result = result if result is not None else agent.predict(request)
        print(note)
        print(json.dumps(result, indent=2))
    print(
        "The separate batch command checks all requests. Reports and plots are already saved in reports/."
    )


if __name__ == "__main__":
    main()
