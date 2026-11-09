#!/usr/bin/env python3
"""Train Y online using a complete frozen cold-start model for C."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'MTRouter'/'src'))
from mixed_pilot_mtrouter.online.frozen_cost.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
