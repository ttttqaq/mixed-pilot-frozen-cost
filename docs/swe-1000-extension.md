# Extend SWE-neo-data-1000

The requested collection retains the original 100 train tasks (50 per source), adds
450 previously uncollected tasks from each source, and assigns all 1,000 tasks to
train. The original collection also contains 20 test tasks: these are excluded
from new sampling and retained in the complete original-directory backup.

The `extend-train` subcommand reuses `convert_row` and `validate_task` from
`scripts/convert_swe_tasksets.py`. It uses the existing collection's pinned upstream
revisions, including the older SWE-rebench V2 revision, rather than current main.
Original task packages are retained byte-for-byte. The upstream `source.split`
remains provenance; every experiment `routing_metadata.router_split` is `train`.

## Current checkpoint

- `run_1002/SWE-neo-data-1000-staging/train/tasks/`: 500 Rebench tasks, including
  50 preserved tasks and 450 additions. Each was validated by the converter.
- Fixed Rebench Parquet SHA-256:
  `0e0bf9355f892ad74ae98d4e1c404f39fd6654a8e351ee3e6ab162e4a64cd3ad`.
- Pro Verified's download has not executed because the automatic approval service
  reported a usage limit. Its additional 450 tasks are not yet collected.
- The original `/Users/liangjianfeng/Data/Router/SWE-neo-data-1000` is unchanged.
- No Docker images, containers, or inference jobs are started by this workflow.

## Resume and publish

Run in a local terminal. The cached Rebench conversion is reused; the remaining
Pro Verified JSONL is downloaded at its pinned revision. Both sources are checked
for duplicate IDs, original package preservation, patches, test metadata, counts,
and train-only assignment before publication. If interrupted, rerun the command.

```bash
cd /Users/liangjianfeng/Documents/Codex/2026-08-25/zon/outputs/mixed-pilot
PYTHONPATH="$PWD/run_1002/data-expansion/python-libs${PYTHONPATH:+:$PYTHONPATH}" \
  /Users/liangjianfeng/miniconda3/envs/opensquilla/bin/python -u \
  scripts/convert_swe_tasksets.py extend-train \
  --dataset-root /Users/liangjianfeng/Data/Router/SWE-neo-data-1000 \
  --staging run_1002/SWE-neo-data-1000-staging \
  --cache run_1002/data-expansion \
  --target-per-source 500 --seed 20261002 --publish
```

Publication renames the original directory to a sibling
`SWE-neo-data-1000.backup-<timestamp>` and installs the validated staging collection
at the original path. The final collection has `train/tasks`, `train.json`,
`train/catalog.json`, and a root `catalog.json`; no `test/` or `test.json`.
If the final rename fails, the original directory is restored. Do not rerun
`--publish` after it reports successful publication; inspect the final
`validation_report.json` instead.

The temporary PyArrow installation was moved into `run_1002/data-expansion/python-libs`
so it remains available for this command without changing the conda environment.
Gold patches, test patches, and raw source records are grading-only data and must
not be exposed as part of an inference agent's prompt or workspace.
