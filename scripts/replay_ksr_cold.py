#!/usr/bin/env python3
"""Re-train KSR cold-start labels and evaluate historical 300/500-step stages."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'MTRouter'/'src'))
from mixed_pilot_mtrouter.online.frozen_cost.replay import main

if __name__ == '__main__':
    raise SystemExit(main())
