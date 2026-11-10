"""Verify an optional local Pi wheel cache against its transfer manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def verify_wheels(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "MANIFEST.json").read_text())
    for item in manifest["files"]:
        filename = item["filename"]
        if Path(filename).name != filename or "/" in filename or "\\" in filename:
            raise ValueError("Wheel filenames must stay inside the cache directory")
        path = folder / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Wheel checksum mismatch: {path.name}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=ROOT / "pi_wheels")
    args = parser.parse_args()
    try:
        manifest = verify_wheels(args.folder)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Wheel cache unavailable: {error}", file=sys.stderr)
        print(
            "A Git clone has no vendor wheels. setup_pi.sh --core installs from the package index.",
            file=sys.stderr,
        )
        return 2
    for item in manifest["files"]:
        print("Verified", item["filename"])
    print(f"{len(manifest['files'])} wheels verified for {manifest['target']}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
