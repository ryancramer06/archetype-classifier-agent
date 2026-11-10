"""Verify the included Pi wheel files against the saved transfer manifest."""

import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/"pi_wheels"
    manifest=json.loads((folder/"MANIFEST.json").read_text())
    for item in manifest["files"]:
        path=folder/item["filename"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item["sha256"]:
            raise ValueError(f"Wheel checksum mismatch: {path.name}")
        print("Verified",path.name)
    print(f"{len(manifest['files'])} wheels verified for {manifest['target']}.")
    return 0


if __name__=="__main__":
    sys.exit(main())
