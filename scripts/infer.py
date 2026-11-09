#!/usr/bin/env python3
"""Dataset inference. Run --help; implementation lives in mixed_pilot_harness.inference."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mixed_pilot_harness.inference.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
