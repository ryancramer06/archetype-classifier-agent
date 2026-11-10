# Raspberry Pi setup

The target is 64-bit Raspberry Pi OS with CPython 3.13. The standard-library
backend also supports Python 3.11 and 3.12. Copy or clone the whole project:

```bash
cd ~/Desktop
git clone https://github.com/ryancramer06/archetype-classifier-agent.git
cd archetype-classifier-agent
python3 demo.py --backend portable --auto
python3 batch.py --backend portable
python3 scripts/pi_check.py
```

The check writes `local_results/pi_check.json`, including Python, machine,
device-tree board model when available, 1,000 reference-vector checks, and batch
results. `raspberry_pi_hardware_detected` distinguishes a board run from a
desktop compatibility check. No physical Pi run is claimed by desktop results.

## Native saved pipeline

```bash
bash scripts/setup_pi.sh --core
.venv/bin/python demo.py
.venv/bin/python scripts/pi_check.py --with-sklearn
```

The operating system must provide Python's venv module. A fresh Git clone has no
vendor wheels, so `--core` downloads the pinned binary packages from the package
index. With an existing complete `pi_wheels/` cache, it verifies the supplied
manifest before installing offline. That original manifest targets CPython 3.13
and Linux ARM64; it is not a Windows package source.

`--full` installs plotting dependencies as well, for training/report generation.
`--lite` runs portable checks without creating an environment. To inspect a
local cache manually, run `python3 scripts/verify_wheels.py`. Missing/corrupt
files produce a clear failure, not a successful verification.

The launcher `bash run_demo.sh` uses `.venv/bin/python` if present and otherwise
uses portable prediction. It is a manual launcher, not an automatic boot service.
