#!/usr/bin/env python3
"""Build a reproducible original SWE-bench sample in the SWE-neo task layout."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mixed_pilot_harness.inference import swebench_runtime as official
from mixed_pilot_harness.inference.dataset import load_dataset
from convert_swe_tasksets import convert_row, require_row, validate_task, write_json

DATASET = "princeton-nlp/SWE-bench"
REVISION = "e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6"
PARQUET = "data/test-00000-of-00001.parquet"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(cache):
    cache.mkdir(parents=True, exist_ok=True)
    for name, url in [
        ("hf.json", f"https://huggingface.co/api/datasets/{DATASET}/revision/{REVISION}"),
        ("test.parquet", f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{PARQUET}"),
    ]:
        if (cache / name).is_file():
            continue
        temp = cache / (name + ".partial")
        with urllib.request.urlopen(url, timeout=180) as response, temp.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        temp.replace(cache / name)


def build(rows, metadata, source_hash, output, count=1000, seed=20261014, verify_official=False):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if metadata["sha"] != REVISION:
        raise ValueError("Source metadata does not match pinned dataset revision")
    published = next(v["num_examples"] for v in metadata["cardData"]["dataset_info"]["splits"]
                     if v["name"] == "test")
    if len(rows) != published:
        raise ValueError("Parquet row count differs from the official test split")
    if len({row["instance_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate instance IDs in the source")
    if not 1 <= count <= len(rows):
        raise ValueError("Sample count outside source population")
    # Validate the whole population before sampling: never silently replace bad
    # selected rows with a different sample or hide dropped source records.
    for row in rows:
        require_row(row)
        official.test_names(row, "FAIL_TO_PASS")
        official.test_names(row, "PASS_TO_PASS")
        official.source_image_for(row)
    indices = random.Random(seed).sample(range(len(rows)), count)
    provenance = {"dataset": DATASET, "revision": REVISION, "config": "default", "split": "test",
                  "url": f"https://huggingface.co/datasets/{DATASET}",
                  "license": metadata.get("cardData", {}).get("license") or "unknown",
                  "data_file": PARQUET, "data_file_sha256": source_hash}
    sampling = {"seed": seed, "population_size": len(rows), "sample_count": count,
                "algorithm": "Python random.Random(int_seed).sample(range(population_size), count)",
                "without_replacement": True, "indices_in_random_order": indices,
                "source_split": "test", "router_split": "train"}
    (output / "raw").mkdir(parents=True)
    (output / "workflow").mkdir()
    write_json(output / "raw/swe-bench.json", {"source": provenance, "sampling": sampling,
                                              "rows": [rows[i] for i in indices]})
    write_json(output / "raw/source_metadata.json", metadata)
    entries, images = [], []
    for rank, index in enumerate(indices):
        row = rows[index]
        image, repo_dir = official.source_image_for(row)
        if verify_official:
            spec = official.test_spec(row)
            if spec.instance_image_key != image or not spec.eval_script:
                raise ValueError("Official evaluator/image validation failed")
        task = convert_row("swe-bench", provenance, row, output / "train", materialize=False)
        manifest = json.loads((task / "manifest.json").read_text())
        manifest["source"]["row_index"] = index
        manifest["routing_metadata"] = {"router_split": "train", "sampling_seed": seed,
                                        "sampling_rank": rank, "source_dataset": "swe-bench"}
        manifest["environment"] = {"benchmark": "swebench", "source_image": image,
                                    "image_policy": "upstream", "platform": "linux/amd64",
                                    "repository_path": repo_dir, "harness_version": official.VERSION,
                                    "harness_revision": official.REVISION}
        manifest["evaluation"]["limitations"] = [
            "Docker image availability and empty/gold execution baselines have not been verified.",
            "Native evaluation requires the mixed-pilot swebench extra and a running Docker daemon."]
        write_json(task / "manifest.json", manifest)
        native = json.loads((task / "reference/native_record.json").read_text())
        native["runtime"].update(manifest["environment"])
        write_json(task / "reference/native_record.json", native)
        validate_task(task)
        if json.loads((task / "reference/source_record.json").read_text()) != row:
            raise ValueError("Source record was modified during conversion")
        entries.append({"id": task.name, "path": task.relative_to(output).as_posix(), "dataset": "swe-bench",
                        "upstream_id": row["instance_id"], "repo": row["repo"],
                        "base_commit": row["base_commit"], "language": "python", "source_row_index": index})
        images.append({"image": image, "instance_id": row["instance_id"], "task_id": task.name,
                       "base_commit": row["base_commit"], "platform": "linux/amd64"})
        if (rank + 1) % 100 == 0:
            print(f"Packaged {rank + 1}/{count}", flush=True)
    catalog = {"schema_version": "1.0", "name": output.name, "task_count": count, "tasks": entries}
    write_json(output / "catalog.json", catalog)
    write_json(output / "train/catalog.json", {**catalog, "name": output.name + "/train"})
    write_json(output / "train.json", {"schema_version": "1.0", "split": "train", "task_count": count,
                                      "tasks": entries})
    # An explicit empty index supports the harness's --split all. No test tasks
    # or test directory are created; all requested samples belong to train.
    write_json(output / "test.json", {"schema_version": "1.0", "split": "test", "task_count": 0, "tasks": []})
    write_json(output / "sampling.json", {"schema_version": "1.0", "source": provenance, **sampling})
    write_json(output / "workflow/images.json", {"image_policy": "upstream", "platform": "linux/amd64",
                                               "total_images": len(images), "images": images,
                                               "local_prepared_images_required": False})
    (output / "workflow/requirements.txt").write_text("swebench==4.1.0\ndatasets>=2.18,<5\n", encoding="utf-8")
    (output / "workflow/REBUILD.md").write_text(
        f"# Rebuild\n\nSource: {DATASET}@{REVISION}, {PARQUET}.\n\n"
        f"From the mixed-pilot repository (Python 3.10+):\n\n```bash\n"
        "python -m pip install '.[swebench]'\n"
        f"python scripts/build_swe_data.py --cache /tmp/swebench-cache --download "
        f"--output /data/{output.name} --count {count} --seed {seed} --verify-official\n```\n"
        "The output must not already exist. Sampling is deterministic; source_row_index is the pinned Parquet row index.\n",
        encoding="utf-8")
    (output / "README.md").write_text(
        f"# {output.name}\n\n"
        f"从原版 SWE-bench 官方 full test 的 {len(rows)} 条中，使用种子 {seed} 无放回随机抽取 {count} 条。"
        "不包含 SWE-rebench 或 SWE-bench Pro。全部属于本地 train，source.split=test 保留官方来源。\n\n"
        "结构与 SWE-neo-data-1000 一致：train.json、catalog.json、train/tasks/<id>/，以及 prompt、manifest、"
        "rubrics、workspace/README.md、reference/{source_record.json,native_record.json,gold.patch,test.patch}。"
        "额外的 test.json 是空索引，方便 --split all；不存在本地测试任务。\n\n"
        "完整官方字段保存在 reference/source_record.json 和 raw/swe-bench.json，未添加 hints_text 到问题提示。"
        "gold/test 补丁和测试清单仅供控制器评测，不在 agent 可见的 workspace 中。不要将整个数据目录挂给 agent。\n\n"
        f"数据 revision：`{REVISION}`；原始 Parquet SHA-256：`{source_hash}`。sampling.json 保存抽样顺序与原始行号，"
        "checksums.json 保存数据集文件校验和。\n\n"
        "mixed-pilot 使用原版 SWE-bench 独立适配器，固定官方 harness 4.1.0；采用官方 eval script 和 get_eval_report 评分。"
        "workspace 是环境描述，不是完整仓库。服务器从公开 registry 拉取实例镜像，在 /testbed 提供 base_commit 源码与 testbed conda 环境。"
        "workflow/images.json 使用上游镜像引用，不依赖本机 swe-neo-prepared 镜像或绝对路径。\n\n"
        "在 Linux x86_64 服务器，复制本数据集和更新后的 mixed-pilot 项目后：\n\n"
        "```bash\npython -m pip install '.[swebench]'\n"
        f"python scripts/pull_swe_images.py --dataset /data/{output.name} --split train "
        f"--output /data/{output.name}-images --dry-run\n"
        f"python scripts/pull_swe_images.py --dataset /data/{output.name} --split train "
        f"--output /data/{output.name}-images\n```\n\n"
        "如 shell 设置了旧的 SWE_NEO_PREPARED_IMAGES，正式推理前执行 `unset SWE_NEO_PREPARED_IMAGES`。"
        "官方实例镜像已带依赖，优先使用原镜像；不要直接套用 SWE-neo 的依赖重建脚本。"
        "建议先用 mixed-pilot-infer --precheck --limit 1 检查空补丁与 gold 基线，再扩大运行。\n\n"
        "本次仅构建并校验数据与评测配置，没有下载 Docker 镜像、运行测试或调用模型 API。evaluation.ready=false。"
        "镜像标签是上游 latest，未锁定 registry digest，未来可用性和镜像内容需要服务器拉取时核实。\n\n"
        "这些训练任务来自官方 test：用于路由器训练后，不能再把重叠任务作为独立 SWE-bench 测试集。\n",
        encoding="utf-8")
    loaded = load_dataset(output, "train")
    if len(loaded) != count or len(load_dataset(output, "all")) != count:
        raise ValueError("Harness task count differs from the sample")
    report = {"status": "valid", "task_count": count, "split_counts": {"train": count, "test": 0},
              "source_counts": {"swe-bench": count}, "upstream_split": "test", "population_size": len(rows),
              "seed": seed, "duplicate_ids": 0, "native_records_preserved": True,
              "repo_counts": dict(sorted(Counter(row["repo"] for row in (rows[i] for i in indices)).items())),
              "harness_load_train": len(loaded), "harness_load_all": count,
              "official_spec_validation": count if verify_official else 0,
              "image_policy": "upstream", "local_prepared_images_required": False,
              "evaluation_ready": False, "docker_preparation_performed": False,
              "registry_availability_verified": False, "gold_baselines_run": 0,
              "model_api_calls": 0, "reference_dataset_modified": False}
    write_json(output / "validation_report.json", report)
    hashes = {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "checksums.json", hashes)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--verify-official", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a fresh destination")
    if args.download:
        download(args.cache)
    import pyarrow.parquet as pq
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".swe-data-build-", dir=args.output.parent) as temporary:
        stage = Path(temporary) / args.output.name
        build(pq.read_table(args.cache / "test.parquet").to_pylist(),
              json.loads((args.cache / "hf.json").read_text()), digest(args.cache / "test.parquet"),
              stage, args.count, args.seed, args.verify_official)
        if args.output.exists():
            raise FileExistsError(args.output)
        stage.rename(args.output)


if __name__ == "__main__":
    main()
