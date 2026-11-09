#!/usr/bin/env python3
"""Dataset step-label collection; run --help for configuration."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mixed_pilot_harness.step_collection.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
