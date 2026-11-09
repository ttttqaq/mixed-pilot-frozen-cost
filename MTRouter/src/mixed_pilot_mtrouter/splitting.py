"""Group splitting prevents prefixes of one task leaking into validation."""
from __future__ import annotations
import random


def split_indices(samples, val_ratio=0.1, seed=42):
    if not 0 <= val_ratio < 1:
        raise ValueError('val_ratio must be in [0, 1)')
    groups = sorted({(s['benchmark'], s['task_id']) for s in samples})
    if val_ratio and len(groups) < 2:
        raise ValueError('Validation needs at least two task instances; use val_ratio=0 for a smoke test')
    random.Random(seed).shuffle(groups)
    n = max(1, min(len(groups)-1, round(len(groups)*val_ratio))) if val_ratio else 0
    validation = set(groups[:n])
    train, val = [], []
    for i, s in enumerate(samples):
        (val if (s['benchmark'], s['task_id']) in validation else train).append(i)
    return train, val
