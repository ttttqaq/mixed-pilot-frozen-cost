# SWE task dataset conversion pilot

`scripts/convert_swe_tasksets.py` converts small samples from two public
Hugging Face datasets into the task-directory format used by HLE,
mixed-pilot-data, and ScienceWorld:

```text
tasks/<id>/
  manifest.json
  prompt.md
  prompt.txt
  workspace/
  rubrics.json
  reference/
    gold.patch
    test.patch
    native_record.json
```

The sources are:

- `nebius/SWE-rebench-V2`, `default/train`:
  https://huggingface.co/datasets/nebius/SWE-rebench-V2
- `opencompass/SWEBench-Pro-Verified`, `default/test`:
  https://huggingface.co/datasets/opencompass/SWEBench-Pro-Verified

The second source is the Pro Verified release associated with arXiv:2609.08149.
It is distinct from `ScaleAI/SWE-bench_Pro` and from the original
`SWE-bench_Verified` dataset.

## Workflow

The downloader uses the public Hugging Face API and limits each call to 20
rows. It records the dataset revision observed at download time. No entire
dataset download is needed.

```bash
python3 scripts/convert_swe_tasksets.py fetch \
  --source swe-rebench-v2 --offset 0 --limit 2 \
  --output run/new-swe-pilot/raw/swe-rebench-v2.json
python3 scripts/convert_swe_tasksets.py fetch \
  --source swe-bench-pro-verified --offset 0 --limit 2 \
  --output run/new-swe-pilot/raw/swe-bench-pro-verified.json

python3 scripts/convert_swe_tasksets.py convert \
  --input run/new-swe-pilot/raw/swe-rebench-v2.json \
  --output run/new-swe-pilot/converted
python3 scripts/convert_swe_tasksets.py convert \
  --input run/new-swe-pilot/raw/swe-bench-pro-verified.json \
  --output run/new-swe-pilot/converted

# Download the base-commit repository only for selected task directories.
python3 scripts/convert_swe_tasksets.py materialize \
  --task-dir run/new-swe-pilot/converted/tasks/<task-id>

python3 scripts/convert_swe_tasksets.py validate \
  --output run/new-swe-pilot/converted
```

`convert --materialize-indices 0,1` also materializes selected zero-based
sample indices during conversion. Without that option, `workspace/` contains
only a repository descriptor; the package is suitable for schema review, not
for solving the coding task. `materialize` downloads a GitHub archive at the
recorded base commit and creates a local single-commit repository. It does not
carry later upstream Git history into the agent workspace.

## Field mapping

| Output | SWE-rebench V2 | SWE-Bench Pro Verified |
| --- | --- | --- |
| `source.upstream_id` | `instance_id` | `instance_id` |
| `source.base_commit` | `base_commit` | `base_commit` |
| `prompt.md` | `problem_statement`, public `interface` | `problem_statement`, `requirements`, `interface` |
| `reference/gold.patch` | `patch` | `patch` |
| `reference/test.patch` | `test_patch` | `test_patch` |
| `reference/native_record.json` | `FAIL_TO_PASS`, `PASS_TO_PASS`, `image_name`, `install_config` | `fail_to_pass`, `pass_to_pass`, `dockerhub_tag`, setup metadata |

Both native test lists are normalized to `FAIL_TO_PASS` and `PASS_TO_PASS` in
`native_record.json`. The source-specific runtime metadata remains in that
private file. `rubrics.json` is a 100-point representation of those two test
groups, with the passing threshold set to 100.

## Evaluation boundary

These packages have `evaluation.ready=false` and
`verification_level=native_evaluator_pending`. Conversion does not install
Docker images, run official tests, or validate an agent patch. In particular,
the current `code-task` evaluator's generic `native_test` handling is not a
substitute for either dataset's container evaluator. A run should only be
reported as benchmark-verified after the appropriate native environment and
test runner are integrated and checked against an unmodified base and the
reference patch.

The Pro Verified release also describes network and local-history leakage
controls. The history-free source snapshots here remove one local leakage
route, but the converter does not implement its full anti-hacking protocol.
Agent runs must apply the official isolation protocol to claim comparable
Pro Verified results.

## Pilot produced on 2026-09-27

Four rows were converted: two from each dataset. WTForms (SWE-rebench V2) and
qutebrowser (Pro Verified) have materialized source snapshots; elastic/synthetics
and NodeBB remain descriptor-only. The samples are stored under
`run/task-conversion-pilot/`, which is ignored by Git. The converter's
`validate` command passed for all four packages. Native evaluation was not run.

## Random train/test collection

`scripts/build_swe_neo_data.py` supports reproducible random sampling from the
full source splits and task-level train/test assignment. By default seed 42
selects 60 distinct row indices per source, assigns the first 50 in random
order to train and the remaining 10 to test, and retains the upstream split
separately from the new experiment split. The Pro Verified source is streamed
from its immutable JSONL revision; rebench rows come from the dataset viewer,
with the Hub revision checked unchanged before and after sampling. The
downloader rejects truncated rows and supports resuming partially completed
downloads without changing the selection.

```bash
python3 scripts/build_swe_neo_data.py sample \
  --output run/swe-neo-build/raw --seed 42 --workers 1
python3 scripts/build_swe_neo_data.py build \
  --raw-dir run/swe-neo-build/raw --output run/swe-neo-build/dataset --seed 42
python3 scripts/build_swe_neo_data.py validate \
  --output run/swe-neo-build/dataset
```

The collection has `train/tasks/` (100 tasks) and `test/tasks/` (20 tasks),
split indexes, a root catalog, sampling provenance, complete private source
records, and checksums. Code repositories remain descriptors and can be
materialized on demand. A train/test split by task is not a repository holdout.
