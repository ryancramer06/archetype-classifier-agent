"""Read a JSON feature request and return a JSON response. No training here."""

import argparse
import json
from pathlib import Path
import sys

from agent import invalid_response, predict_request


def strict_json(text):
    def no_constant(value):
        raise ValueError(f"Nonstandard JSON constant: {value}")

    def unique_keys(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(text, parse_constant=no_constant, object_pairs_hook=unique_keys)


def read_request(path):
    try:
        return strict_json(Path(path).read_text(encoding="utf-8-sig")), None
    except OSError:
        return None, invalid_response(None, "request_file_unreadable", "Could not read the JSON request file.")
    except (ValueError, UnicodeError):
        return None, invalid_response(None, "malformed_json", "Request must be valid JSON with unique keys and finite numeric constants.")


def main():
    parser = argparse.ArgumentParser(description="Predict a fictional archetype from six synthetic features.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--request", type=Path, help="JSON request file")
    source.add_argument("--stdin", action="store_true", help="Read a request from standard input")
    parser.add_argument("--backend", choices=["portable", "sklearn"], default="sklearn")
    parser.add_argument("--model-dir", type=Path, help="Optional saved-model directory")
    parser.add_argument("--output", type=Path, help="Optional JSON response destination")
    args = parser.parse_args()
    if args.request:
        request, result = read_request(args.request)
    else:
        try:
            request, result = strict_json(sys.stdin.read()), None
        except ValueError:
            request, result = None, invalid_response(None, "malformed_json", "Standard input was not a valid request JSON.")
    if result is None:
        result = predict_request(request, args.model_dir, args.backend)
    text = json.dumps(result, indent=2, allow_nan=False)
    print(text)
    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        except OSError as error:
            print(f"Could not write response: {error}", file=sys.stderr)
            return 4
    return {"ok": 0, "uncertain": 1, "invalid_input": 2, "unavailable": 3}[result["status"]]


if __name__ == "__main__":
    sys.exit(main())
