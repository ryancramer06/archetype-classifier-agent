"""Inspect accepted predictions, uncertainty, and input validation."""

import argparse
import json
from pathlib import Path
import sys

from agent import ArchetypeAgent, ModelError
from config import ROOT
from predict import read_request

EXAMPLES = [
    ("valid_explorer.json", "Valid synthetic measurements"),
    ("uncertain.json", "Valid input below the acceptance threshold"),
    ("boolean_feature.json", "A Boolean supplied as a measurement"),
    ("extra_feature.json", "An unexpected target feature"),
    ("above_range.json", "A measurement outside the allowed range"),
]


def format_summary(response):
    """Show the decision first; leave the complete machine response to JSON mode."""
    lines = [
        f"Status: {response['status']}",
        f"Category: {response['category'] or '(none)'}",
    ]
    if response["confidence"] is not None:
        lines.append(
            f"Model probability: {response['confidence']:.2%} "
            f"(acceptance threshold: {response['acceptance_threshold']:.0%})"
        )
    if response["explanation"]:
        lines.append("Largest contributions to the class score:")
        for term in response["explanation"]["top_features"]:
            lines.append(
                f"  {term['feature']}: {term['contribution']:+.3f} ({term['direction']})"
            )
    if response["reason"]:
        lines.append("Reason: " + response["reason"].replace("_", " "))
    if response["error"]:
        lines.append(
            f"Error ({response['error']['code']}): {response['error']['message']}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auto", action="store_true", help="Run without pauses")
    parser.add_argument("--backend", choices=["portable", "sklearn"], default="sklearn")
    parser.add_argument("--format", choices=["summary", "json"], default="summary")
    parser.add_argument("--model-dir", type=Path, help="Optional saved-model directory")
    args = parser.parse_args()
    try:
        agent = ArchetypeAgent(model_dir=args.model_dir, backend=args.backend)
        print("ARCHETYPE CLASSIFIER AGENT")
        print("Synthetic data; fictional categories. Probabilities are uncalibrated.\n")
        for index, (name, description) in enumerate(EXAMPLES, start=1):
            if not args.auto:
                try:
                    input(f"Press Enter for {name}.")
                except EOFError:
                    print(
                        "\nStopped before the next example. Use --auto for noninteractive runs."
                    )
                    return 0
            print(f"[{index}/{len(EXAMPLES)}] {description} ({name})")
            request, response = read_request(ROOT / "samples/requests" / name)
            if response is None:
                response = agent.predict(request)
            print(
                json.dumps(response, indent=2)
                if args.format == "json"
                else format_summary(response)
            )
            print()
    except ModelError:
        print(
            "Demo unavailable: check the saved model and pinned dependencies. "
            "Use --backend portable for package-free prediction.",
            file=sys.stderr,
        )
        return 3
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    print(
        "Run batch.py to check every included request. Results and plots are in reports/."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
