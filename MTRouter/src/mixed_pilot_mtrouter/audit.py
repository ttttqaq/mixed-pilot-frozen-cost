"""Read-only, stdlib-only validation before expensive embedding/training."""
from __future__ import annotations
import argparse
import json
import math
from collections import Counter
from pathlib import Path
from .research.trajectory.parser import TrajectoryParser, find_training_sample_files, find_trajectory_files


def audit_directories(directories, model_names):
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError('model_names must be nonempty and unique')
    paths = [p.resolve() for d in directories for p in find_trajectory_files(d)]
    sample_paths = [p.resolve() for d in directories for p in find_training_sample_files(d)]
    if not paths and not sample_paths:
        raise ValueError("No .traj.json or samples.jsonl files found")
    if set(paths) & set(sample_paths):
        raise ValueError("Input files overlap")
    if len(paths) != len(set(paths)):
        raise ValueError('Overlapping input directories contain duplicate trajectories')
    if len(sample_paths) != len(set(sample_paths)):
        raise ValueError('Overlapping input directories contain duplicate samples.jsonl files')
    parser = TrajectoryParser(model_names=model_names)
    counts, currencies, policies = Counter(), set(), Counter()
    for path in sorted(paths):
        data = json.loads(path.read_text()); info = data.get('info', {})
        if info.get('labels_valid') is False or info.get('cost_complete') is False:
            raise ValueError(f'{path}: invalid label or incomplete cost')
        currency = info.get('cost_currency')
        if not currency:
            raise ValueError(f'{path}: explicit info.cost_currency required')
        currencies.add(currency)
        if info.get('final_score') is None or not math.isfinite(float(info['final_score'])):
            raise ValueError(f'{path}: finite terminal final_score required')
        traj = parser.parse(path)
        if not traj.steps:
            raise ValueError(f'{path}: empty trajectory')
        if 'steps' in data and len(traj.steps) != len(data['steps']):
            raise ValueError(f'{path}: messages and steps disagree')
        for step in traj.steps:
            if step.model_idx < 0:
                raise ValueError(f'{path}: unknown model {step.model_name!r}')
            if not math.isfinite(step.step_cost) or step.step_cost < 0:
                raise ValueError(f'{path}: invalid cost')
            counts[step.model_name] += 1
        policy = info.get('policy', {})
        policies[policy if isinstance(policy,str) else policy.get('mode','unknown')] += 1
    valid_sample_rows = 0
    invalid_sample_rows = 0
    for path in sorted(sample_paths):
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("row must be an object")
                    if row.get("labels_valid") is False or row.get("cost_complete") is False or row.get("y") is None:
                        invalid_sample_rows += 1
                        continue
                    y = float(row["y"])
                    if row.get("schema") == "mtrouter-step-v1" and row.get("cost") is None:
                        raise ValueError("explicit cost required")
                    model = row.get("action_model") or row.get("action_profile")
                    cost = float(row.get("cost", 0.0) or 0.0)
                    if not math.isfinite(y) or not 0.0 <= y <= 1.0:
                        raise ValueError("invalid y")
                    if not model or model not in model_names:
                        raise ValueError(f"unknown model {model!r}")
                    if not math.isfinite(cost) or cost < 0:
                        raise ValueError("invalid cost")
                    remaining = float(row.get("remaining_cost", row.get("remaining_api_cost", cost)))
                    if not math.isfinite(remaining) or remaining < 0:
                        raise ValueError("invalid remaining cost")
                    currency = row.get("cost_currency")
                    if row.get("schema") == "mtrouter-step-v1" and not currency:
                        raise ValueError("explicit cost_currency required")
                    currencies.add(currency or "CNY")  # Legacy samples used the CNY snapshot.
                    counts[model] += 1
                    valid_sample_rows += 1
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{line_no}: invalid samples.jsonl row: {exc}") from exc
        policies["jsonl"] += 1
    if valid_sample_rows == 0 and sample_paths:
        raise ValueError("No valid samples.jsonl rows found")
    if len(currencies) != 1:
        raise ValueError(f'Mixed currencies require explicit conversion: {sorted(currencies)}')
    result = {'trajectories': len(paths), 'steps': sum(counts.values()),
              'samples': valid_sample_rows, 'invalid_samples': invalid_sample_rows,
              'models': dict(counts), 'currency': next(iter(currencies), 'CNY'), 'policies': dict(policies)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, action='append', required=True)
    p.add_argument('--models', nargs='+', required=True)
    a = p.parse_args(); audit_directories(a.data_dir, a.models)

if __name__ == '__main__': main()
