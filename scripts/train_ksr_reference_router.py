#!/usr/bin/env python3
"""Run KSR cold-start training from the new Reference Teacher child branch."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'MTRouter'/'src'))
from mixed_pilot_mtrouter.online.ksr.cli import main
if __name__ == '__main__':
    raise SystemExit(main())
