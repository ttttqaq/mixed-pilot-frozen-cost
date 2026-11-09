"""Original SWE-bench: public images and pinned official scripts/log grading."""
from __future__ import annotations

import importlib.metadata
import json
import re
import shutil
import subprocess
import time

VERSION = "4.1.0"
REVISION = "726c5461e2ef52d83cf1ea2107870a8bb3328d57"
NAMESPACE = "swebench"


def matches(row):
    return (not row.get("image_name") and not row.get("dockerhub_tag")
            and all(key in row for key in ("version", "FAIL_TO_PASS", "PASS_TO_PASS")))


def test_names(row, key):
    value = row[key]
    value = json.loads(value) if isinstance(value, str) else value
    if not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value):
        raise ValueError(f"Invalid SWE-bench {key}")
    if key == "FAIL_TO_PASS" and not value:
        raise ValueError("Empty SWE-bench FAIL_TO_PASS")
    return value


def source_image_for(row):
    # TestSpec.instance_image_key in the pinned official revision. This pure
    # resolver also works for dry-run planning without installing swebench.
    identifier = row["instance_id"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+", identifier):
        raise ValueError(f"Invalid SWE-bench instance ID: {identifier}")
    return (f"{NAMESPACE}/sweb.eval.x86_64.{identifier.lower()}:latest".replace("__", "_1776_"),
            "/testbed")


def require_official():
    try:
        version = importlib.metadata.version("swebench")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Original SWE-bench requires: python -m pip install '.[swebench]'") from exc
    if version != VERSION:
        raise RuntimeError(f"Expected swebench=={VERSION}, found {version}; install '.[swebench]'")


def test_spec(row):
    require_official()
    from swebench.harness.constants import MAP_REPO_TO_EXT, MAP_REPO_VERSION_TO_SPECS
    from swebench.harness.test_spec.test_spec import TestSpec
    from swebench.harness.test_spec.create_scripts import make_eval_script_list

    specs = MAP_REPO_VERSION_TO_SPECS[row["repo"]][row["version"]]
    # Public instance images already contain the installed repository and env.
    # Generate only the official evaluation script: building env scripts would
    # unnecessarily fetch historical requirements from GitHub on the controller.
    spec = TestSpec(instance_id=row["instance_id"], repo=row["repo"], version=row["version"],
                    repo_script_list=[], env_script_list=[],
                    eval_script_list=make_eval_script_list(
                        row, specs, "testbed", "/testbed", row["base_commit"], row["test_patch"]),
                    arch="x86_64", FAIL_TO_PASS=test_names(row, "FAIL_TO_PASS"),
                    PASS_TO_PASS=test_names(row, "PASS_TO_PASS"),
                    language=MAP_REPO_TO_EXT[row["repo"]], docker_specs=specs.get("docker_specs", {}),
                    namespace=NAMESPACE)
    if spec.instance_image_key != source_image_for(row)[0]:
        raise ValueError("Official SWE-bench image mapping differs from the runtime resolver")
    return spec


def grade_logs(row, patch, log_path):
    from swebench.harness.grading import get_eval_report
    prediction = {"instance_id": row["instance_id"], "model_name_or_path": "mixed-pilot",
                  "model_patch": patch.read_text(encoding="utf-8")}
    report = get_eval_report(test_spec(row), prediction, log_path, include_tests_status=True)
    detail = report[row["instance_id"]]
    if not detail.get("patch_successfully_applied"):
        return {"status": "error", "resolved": None, "error_category": "official_test_execution",
                "official_report": report}
    return {"status": "passed" if detail["resolved"] else "failed",
            "resolved": detail["resolved"], "rule": "official_swebench_resolution",
            "official_report": report}


def evaluate(row, patch, cache, output, timeout, cpus, memory):
    from . import swe_runtime as native
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(patch, output / "prediction.patch")
    session = None
    phase = "preparation"
    started = time.monotonic()
    try:
        spec = test_spec(row)
        from swebench.harness.run_evaluation import GIT_APPLY_CMDS
        (output / "eval.sh").write_text(spec.eval_script, encoding="utf-8")
        phase = "container_start"
        # Evaluation gets a fresh image, never the agent's modified workspace.
        session = native.DockerSession(row, output, cpus=cpus, memory=memory,
                                      mounts=[(output.resolve(), "/swe-neo-eval", False)])
        phase = "prediction_patch"
        # Empty patches are used by --precheck to measure the unchanged image.
        applied = not (output / "prediction.patch").stat().st_size
        attempts = []
        for cmd in ([] if applied else GIT_APPLY_CMDS):
            result = session.exec(cmd + " /swe-neo-eval/prediction.patch", timeout=120)
            attempts.append(result.stdout + result.stderr)
            if result.returncode == 0:
                applied = True
                break
        (output / "patch-apply.log").write_text("\n".join(attempts), encoding="utf-8")
        if not applied:
            raise RuntimeError("Official patch application failed; see patch-apply.log")
        phase = "test_execution"
        result = session.exec("bash /swe-neo-eval/eval.sh > /swe-neo-eval/test_output.txt 2>&1", timeout=timeout)
        (output / "container.stdout.log").write_text(result.stdout, encoding="utf-8")
        (output / "container.stderr.log").write_text(result.stderr, encoding="utf-8")
        phase = "test_parsing"
        value = grade_logs(row, output / "prediction.patch", output / "test_output.txt")
        value["returncode"] = result.returncode
        native.write_json(output / "report.json", value["official_report"])
    except Exception as exc:
        value = {"status": "error", "resolved": None, "error_type": type(exc).__name__,
                 "error": str(exc), "error_category": "evaluation_timeout"
                 if isinstance(exc, subprocess.TimeoutExpired) else phase}
    finally:
        if session:
            session.close()
    value.update(instance_id=row["instance_id"], image=native.image_for(row)[0],
                 runtime_revision="swebench-official-v1", official_harness_version=VERSION,
                 official_harness_revision=REVISION, phase=phase,
                 elapsed_seconds=time.monotonic() - started)
    native.write_json(output / "result.json", value)
    return value
