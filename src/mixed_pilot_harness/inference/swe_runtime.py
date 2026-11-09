"""Docker tools and native SWE graders used by the unified inference adapters.

Evaluation logic follows the pinned upstream revisions below. No model requests
are made by preparation or grading. Benchmark references stay on the controller.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import uuid
import time
from functools import lru_cache
from pathlib import Path

from mixed_pilot_harness.tools import WorkspaceTools
from mixed_pilot_harness.processes import run as run_process, kill_group, noninteractive_env, NONINTERACTIVE

REBENCH_REV = "c71902a8cf8d2b725f63d51f199f4d3e56f68d2d"
PRO_REV = "66f92766bba642462d4bbe5479e83f91f9211862"
AGENTCOMPASS_REV = "d409251420bf97b97d0b103287759353f925adc3"
PATCH_READY = "__SWE_NEO_PATCH_READY__"
DOCKER_PLATFORM = os.environ.get("SWE_NEO_DOCKER_PLATFORM", "linux/amd64")
RUNTIME_DIR = Path(__file__).resolve().parent / "docker_runtime"
RUNTIME_REVISION = "native-eval-v5-image-source"
SNAPSHOT_REVISION = "image-source-v1"
IMAGE_BASELINE_MARKER = "SWE image source baseline: "


@lru_cache(maxsize=128)
def image_node_options(image):
    values = json.loads(command(["docker", "image", "inspect", "--format",
                                 "{{json .Config.Env}}", image]).stdout) or []
    existing = next((value.partition("=")[2] for value in values
                     if value.startswith("NODE_OPTIONS=")), "")
    return (existing + " --require=/swe-neo-runtime/system-chromium.cjs").strip()


@lru_cache(maxsize=128)
def image_dependency_dirs(image, repo_dir):
    # The agent's bind mount hides files at the image's repository path. Keep
    # installed, untracked dependencies using per-container anonymous volumes.
    # Docker populates each volume from the image before applying bind mounts.
    script = ("find " + shlex.quote(repo_dir) + " -maxdepth 4 "
              "-name .git -prune -o -type d "
              "\\( -name node_modules -o -name .venv -o -name venv -o -name vendor \\) -prune -print")
    name = 'swe-neo-scan-' + uuid.uuid4().hex
    try:
        result = command(["docker", "run", "--rm", "--name", name, "--platform", DOCKER_PLATFORM,
                          "--network", "none", "--entrypoint", "/bin/sh", image, "-c", script], timeout=60)
    finally:
        command(['docker', 'rm', '--force', '--volumes', name], check=False, timeout=30)
    return tuple(path for path in result.stdout.splitlines()
                 if path.startswith(repo_dir.rstrip("/") + "/")
                 and not {".git", ".."}.intersection(Path(path).parts))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def command(argv, *, cwd=None, timeout=600, check=True, env=None):
    result = run_process(argv, cwd=cwd, timeout=timeout, env=env)
    if check and result.returncode:
        raise RuntimeError(f"{argv[0]} failed ({result.returncode}): {result.stderr[-3000:]}")
    return result


def source_image_for(row: dict) -> tuple[str, str]:
    """Resolve the upstream registry image without reading host-specific overrides."""
    from . import swebench_runtime
    if swebench_runtime.matches(row):
        return swebench_runtime.source_image_for(row)
    if row.get("image_name"):
        image, repo = row["image_name"], "/" + row["repo"].split("/")[1]
    else:
        tag = row.get("dockerhub_tag")
        if not tag:
            raise ValueError("Task has neither image_name nor dockerhub_tag")
        image, repo = "jefzda/sweap-images:" + tag, "/app"
    return image, repo


@contextmanager
def original_images():
    """Scope planning/preparation to upstream images, restoring caller settings."""
    previous = os.environ.pop('SWE_NEO_PREPARED_IMAGES', None)
    try:
        yield
    finally:
        os.environ.pop('SWE_NEO_PREPARED_IMAGES', None)
        if previous is not None:
            os.environ['SWE_NEO_PREPARED_IMAGES'] = previous


def image_for(row: dict) -> tuple[str, str]:
    image, repo = source_image_for(row)
    prepared = os.environ.get("SWE_NEO_PREPARED_IMAGES")
    if prepared:
        manifest = json.loads(Path(prepared).read_text())
        entry = manifest["images"].get(row["instance_id"])
        if entry is None:
            if manifest.get("require_prepared", True):
                raise ValueError(f"No verified prepared image for {row['instance_id']}")
        else:
            if entry["source_image"] != image or entry["base_commit"] != row["base_commit"]:
                raise ValueError(f"Prepared image provenance mismatch for {row['instance_id']}")
            image = entry["image"]
    return image, repo


def bootstrap_rebench(cache: Path) -> Path:
    root = cache / "tooling" / "SWE-rebench-V2"
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root.parent) as tmp:
            staging = Path(tmp) / "repo"
            command(["git", "init", "-q", str(staging)])
            command(["git", "-C", str(staging), "fetch", "--depth", "1",
                     "https://github.com/SWE-rebench/SWE-rebench-V2.git", REBENCH_REV])
            command(["git", "-C", str(staging), "checkout", "-q", "FETCH_HEAD"])
            staging.rename(root)
    actual = command(["git", "-C", str(root), "rev-parse", "HEAD"]).stdout.strip()
    if actual != REBENCH_REV:
        raise ValueError(f"Unexpected SWE-rebench evaluator revision: {actual}")
    return root


def pro_file(cache: Path, relative: str, optional=False) -> Path:
    path = cache / "tooling" / "SWE-bench_Pro-os" / PRO_REV / relative
    if path.is_file():
        return path
    url = f"https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/{PRO_REV}/{relative}"
    try:
        with urllib.request.urlopen(url, timeout=90) as response:
            data = response.read()
    except urllib.error.HTTPError as exc:
        if optional and exc.code == 404:
            return path
        raise
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temporary.write_bytes(data)
    temporary.replace(path)
    return path


def prepare_native(row: dict, cache: Path) -> None:
    from . import swebench_runtime
    if swebench_runtime.matches(row):
        swebench_runtime.test_spec(row)
        return
    if row.get("image_name"):
        bootstrap_rebench(cache)
    else:
        identifier = row["instance_id"]
        for name in ("run_script.sh", "parser.py"):
            pro_file(cache, f"run_scripts/{identifier}/{name}")
        for kind in ("base", "instance"):
            pro_file(cache, f"dockerfiles/{kind}_dockerfile/{identifier}/Dockerfile", optional=True)


def pull_image(row: dict, log_dir: Path, timeout: int) -> None:
    image, _ = image_for(row)
    if command(["docker", "image", "inspect", image], check=False).returncode == 0:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "image-pull.log").open("w", encoding="utf-8") as stream:
        result = run_process(["docker", "pull", "--platform", DOCKER_PLATFORM, image], stdout=stream,
                                stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Image pull failed; see {log_dir / 'image-pull.log'}")


def copy_image_source(row, image_id, workspace):
    """Copy the evaluator's pre-patch source from a disposable offline container."""
    session = DockerSession(row, workspace.parent / 'docker', image=image_id)
    try:
        result = session.exec('set -e; git config core.hooksPath /dev/null; git reset --hard ' +
                              shlex.quote(row['base_commit']) + '; git clean -fd', timeout=120)
        if result.returncode:
            raise RuntimeError('Cannot prepare image source: ' + result.stderr[-3000:])
        tracked = session.exec('git ls-files --recurse-submodules -z', timeout=60)
        if tracked.returncode:
            raise RuntimeError('Cannot list image source files: ' + tracked.stderr[-3000:])
        workspace.mkdir(parents=True)
        command(['docker', 'cp', session.name + ':' + session.repo_dir + '/.', str(workspace)], timeout=1200)
        return tracked.stdout.split('\0')
    finally:
        session.close()


def repo_snapshot(row: dict, cache: Path) -> tuple[Path, str]:
    """Mirror the chosen image, isolate hidden tests, then discard upstream history."""
    image, _ = image_for(row)
    image_id = command(['docker', 'image', 'inspect', '--format', '{{.Id}}', image], timeout=60).stdout.strip()
    if not image_id:
        raise ValueError('Selected image has no immutable ID')
    identity = hashlib.sha256((image_id + row['base_commit']).encode()).hexdigest()[:16]
    # Never reuse older GitHub/submodule snapshots, or a snapshot of a stale tag.
    root = cache / SNAPSHOT_REVISION / row['instance_id'] / identity
    stamp = root / "snapshot.json"
    if stamp.exists():
        metadata = json.loads(stamp.read_text())
        if metadata["base_commit"] != row["base_commit"]:
            raise ValueError("Cached snapshot base commit differs")
        workspace = root / "workspace"
        actual = command(["git", "-C", str(workspace), "rev-parse", "HEAD"]).stdout.strip()
        dirty = command(["git", "-C", str(workspace), "status", "--porcelain"]).stdout
        if actual != metadata["snapshot_commit"] or dirty:
            raise ValueError(f"Cached base snapshot was modified: {workspace}")
        return workspace, metadata["snapshot_commit"]
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent) as tmp:
        staging = Path(tmp) / "snapshot"
        workspace = staging / "workspace"
        # Record exact tracked paths (also inside initialized submodules) while
        # the image's original gitdir references are still valid.
        tracked = copy_image_source(row, image_id, workspace)
        hidden = []
        if not row.get("image_name"):
            for name in row.get("selected_test_files_to_run", []):
                name = name.removeprefix("/app/")
                relative = Path(name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe hidden-test path: {name}")
                hidden.append(name)
                target = workspace / relative
                if not target.parent.resolve().is_relative_to(workspace.resolve()):
                    raise ValueError(f"Hidden-test parent escapes source workspace: {name}")
                if target.is_symlink() or target.is_file():
                    target.unlink()
                elif target.is_dir():
                    shutil.rmtree(target)
        shutil.rmtree(workspace / ".git")
        for nested in list(workspace.rglob(".git")):
            if nested.is_dir():
                shutil.rmtree(nested)
            else:
                nested.unlink()
        command(["git", "init", "-q", str(workspace)])
        command(["git", "-C", str(workspace), "config", "core.hooksPath", "/dev/null"])
        command(["git", "-C", str(workspace), "add", "-A"])
        remaining = [p for p in tracked if p and ((workspace / p).is_file() or (workspace / p).is_symlink())
                     and not any(p == h or p.startswith(h + "/") for h in hidden)]
        for start in range(0, len(remaining), 200):
            command(["git", "--literal-pathspecs", "-C", str(workspace), "add", "-f", "--",
                     *remaining[start:start + 200]])
        command(["git", "-C", str(workspace), "-c", "user.name=SWE Neo",
                 "-c", "user.email=swe-neo@localhost", "commit", "--allow-empty",
                 "--no-gpg-sign", "-qm", IMAGE_BASELINE_MARKER + image_id])
        base = command(["git", "-C", str(workspace), "rev-parse", "HEAD"]).stdout.strip()
        write_json(staging / "snapshot.json", {"snapshot_revision": SNAPSHOT_REVISION,
                   "base_commit": row["base_commit"], "image": image, "image_id": image_id,
                   "snapshot_commit": base, "hidden_test_files": hidden})
        staging.rename(root)
    return root / "workspace", base


class DockerSession:
    def __init__(self, row, log_dir, cpus=4, memory="8g", workspace=None, mounts=(), image=None):
        self.name = "swe-neo-" + uuid.uuid4().hex
        selected, self.repo_dir = image_for(row)
        self.image = image or selected
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        argv = ["docker", "run", "--platform", DOCKER_PLATFORM, "--detach", "--init", "--name", self.name, "--network", "none",
                "--cpus", str(cpus), "--memory", memory, "--entrypoint", "/bin/bash"]
        argv += ["--mount", f"type=bind,source={RUNTIME_DIR},target=/swe-neo-runtime,readonly",
                 "--env", "NODE_OPTIONS=" + image_node_options(self.image)]
        from . import swebench_runtime
        if swebench_runtime.matches(row):
            # Bash tool invocations are non-login shells. Select the same conda
            # environment that the official eval script activates explicitly.
            values = json.loads(command(["docker", "image", "inspect", "--format",
                                         "{{json .Config.Env}}", self.image]).stdout) or []
            image_path = next((v.partition("=")[2] for v in values if v.startswith("PATH=")),
                              "/usr/local/bin:/usr/bin:/bin")
            argv += ["--env", "PATH=/opt/miniconda3/envs/testbed/bin:" + image_path,
                     "--env", "CONDA_DEFAULT_ENV=testbed"]
        for key, value in NONINTERACTIVE.items():
            argv += ['--env', key + '=' + value]
        self.dependency_dirs = []
        if workspace:
            argv += ["--mount", f"type=bind,source={workspace},target={self.repo_dir}"]
            hidden = [Path(p.removeprefix('/app/')) for p in row.get('selected_test_files_to_run', [])]
            for path in image_dependency_dirs(self.image, self.repo_dir):
                relative = Path(path).relative_to(self.repo_dir)
                # A dependency volume must not restore tests removed from the
                # source snapshot, even when the whole directory is now absent.
                if any(h.is_relative_to(relative) or relative.is_relative_to(h) for h in hidden):
                    continue
                if not (Path(workspace) / relative).exists():
                    argv += ["--mount", f"type=volume,target={path}"]
                    self.dependency_dirs.append(path)
        for source, target, readonly in mounts:
            argv += ["--mount", f"type=bind,source={source},target={target}" +
                     (",readonly" if readonly else "")]
        argv += [self.image, "-c", "while :; do sleep 3600; done"]
        try:
            result = command(argv, timeout=120)
            (self.log_dir / "container-start.log").write_text(result.stdout + result.stderr)
            write_json(self.log_dir / "runtime.json", {
                "runtime_revision": RUNTIME_REVISION, "image": self.image,
                "platform": DOCKER_PLATFORM, "network": "none",
                "browser_adapter": "alpine_system_chromium_when_available",
                "dependency_volumes": self.dependency_dirs})
        except BaseException:
            self.close()
            raise

    def close(self):
        try:
            command(["docker", "rm", "--force", "--volumes", self.name], timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired):
            (self.log_dir / "cleanup-error.log").write_text(
                f"Could not remove container {self.name}; run docker rm -f {self.name}\n")

    def exec(self, script, timeout):
        try:
            return command(["docker", "exec", "-w", self.repo_dir, self.name,
                            "/bin/bash", "-c", 'exec </dev/null\n' + script], timeout=timeout, check=False)
        except BaseException:
            # Killing the docker client does not stop the remote exec process.
            self.close()
            raise


class DockerWorkspaceTools(WorkspaceTools):
    """Keep the harness file tools; execute shell/git tools inside the task image."""
    def __init__(self, workspace, session, timeout, log_dir, max_timeout=3600):
        self.session = session
        self.max_timeout = max_timeout
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.shell_calls = 0
        self.fatal_error = None
        super().__init__(workspace, timeout=timeout)

    async def execute(self, name, arguments):
        if self.fatal_error:
            raise RuntimeError(self.fatal_error)
        result = await super().execute(name, arguments)
        # The base tool wrapper turns ordinary errors into observations. A
        # removed container is terminal: do not spend more tokens on dead tools.
        if self.fatal_error:
            raise RuntimeError(self.fatal_error)
        return result

    async def shell(self, arguments):
        self.shell_calls += 1
        stem = self.log_dir / f"shell-{self.shell_calls:04d}"
        requested = float(arguments.get("timeout") or self.timeout)
        if not math.isfinite(requested) or requested <= 0:
            return json.dumps({"ok": False, "error": "timeout must be finite and positive"})
        timeout = min(requested, self.max_timeout)
        # -s works with both GNU coreutils and Alpine's BusyBox timeout.
        argv = ["docker", "exec", "-w", self.session.repo_dir, self.session.name,
                "timeout", "-s", "KILL", str(math.ceil(timeout)), "/bin/bash", "-c",
                str(arguments["command"])]
        with stem.with_suffix(".stdout").open("wb") as out, stem.with_suffix(".stderr").open("wb") as err:
            process = await asyncio.create_subprocess_exec(*argv, stdout=out, stderr=err,
                stdin=asyncio.subprocess.DEVNULL, start_new_session=True, env=noninteractive_env())
            try:
                await asyncio.wait_for(process.wait(), timeout=timeout + 30)
            except BaseException:
                self.fatal_error = 'Shell interrupted or timed out; task container removed'
                kill_group(process)
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                finally:
                    self.session.close()
                raise
            if process.returncode in {124, 137, 143}:
                self.fatal_error = f'Shell timed out or was killed (exit {process.returncode}); task container removed; logs: {stem}'
                self.session.close()
                raise TimeoutError(f'Shell timed out or was killed after at most {timeout}s; '
                                   f'container removed; see {stem}.stdout and {stem}.stderr')
        def tail(path):
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 100_000))
                return stream.read().decode("utf-8", errors="replace")
        return json.dumps({"ok": process.returncode == 0, "returncode": process.returncode,
                           "stdout": tail(stem.with_suffix(".stdout")),
                           "stderr": tail(stem.with_suffix(".stderr")),
                           "full_stdout": str(stem.with_suffix(".stdout")),
                           "full_stderr": str(stem.with_suffix(".stderr"))}, ensure_ascii=False)

    async def git_diff(self, arguments):
        return await self.shell({"command": "git diff --no-ext-diff --binary HEAD; git status --short"})


def export_patch(workspace: Path, base: str, destination: Path):
    """Use a temporary index: include new files and edits committed by the agent."""
    # Legacy incomplete baselines cannot distinguish recovered submodule files
    # from genuine additions. Fail explicitly instead of exporting a huge patch
    # that collides with the evaluator's already-populated dependency tree.
    message = command(['git', '-C', str(workspace), 'show', '-s', '--format=%B', base]).stdout
    # Image baselines may intentionally have uninitialized optional submodules.
    # Their absence matches the evaluator and is not evidence of a bad snapshot.
    if not message.startswith(IMAGE_BASELINE_MARKER):
        config = command(["git", "-C", str(workspace), "config", "--blob", base + ":.gitmodules",
                          "--null", "--get-regexp", r"^submodule\..*\.path$"], check=False)
        for entry in config.stdout.split("\0"):
            if not entry:
                continue
            _, path = entry.split("\n", 1)
            tracked = command(["git", "--literal-pathspecs", "-C", str(workspace),
                               "ls-tree", "-r", base, "--", path]).stdout
            if not tracked or any(line.startswith("160000 ") for line in tracked.splitlines()):
                raise RuntimeError(f"Incomplete submodule baseline for {path!r}; rebuild the source snapshot "
                                   "before inference. Do not treat recovered dependency files as prediction edits.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        env = os.environ.copy()
        env["GIT_INDEX_FILE"] = str(Path(tmp) / "index")
        # Seed from the current index so explicitly tracked/staged additions
        # remain included even if an ignore rule covers their directory.
        current_tree = command(["git", "-C", str(workspace), "write-tree"]).stdout.strip()
        command(["git", "-C", str(workspace), "read-tree", current_tree], env=env)
        command(["git", "-C", str(workspace), "add", "-A"], env=env)
        result = command(["git", "-C", str(workspace), "diff", "--cached", "--binary",
                          "--no-ext-diff", base], env=env)
    destination.write_text(result.stdout, encoding="utf-8")


def grade_tests(row, tests, normalize=lambda name: name, exact=False):
    from . import swebench_runtime
    if swebench_runtime.matches(row):
        row = {**row, "FAIL_TO_PASS": swebench_runtime.test_names(row, "FAIL_TO_PASS"),
               "PASS_TO_PASS": swebench_runtime.test_names(row, "PASS_TO_PASS")}
    fail = set(map(normalize, row.get("FAIL_TO_PASS", row.get("fail_to_pass", []))))
    keep = set(map(normalize, row.get("PASS_TO_PASS", row.get("pass_to_pass", []))))
    if not fail:
        raise ValueError("Empty FAIL_TO_PASS: cannot report a valid resolution")
    passed = {normalize(str(t["name"])) for t in tests if t["status"] == "PASSED"}
    missing_fail, missing_keep = sorted(fail - passed), sorted(keep - passed)
    extra = sorted(passed - fail - keep)
    resolved = not missing_fail and not missing_keep and (not exact or not extra)
    return {"status": "passed" if resolved else "failed", "resolved": resolved,
            "fail_to_pass_count": len(fail), "pass_to_pass_count": len(keep),
            "fail_to_pass_missing": missing_fail, "pass_to_pass_missing": missing_keep,
            "extra_passed_tests": extra, "tests": tests}


def classify_test_output(row, tests, logs, normalize=lambda name: name, exact=False):
    """Fail only with test/source evidence; unresolved parser output stays unscored."""
    plain = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", logs)
    # Some upstream parsers emit a synthetic failure instead of any test names.
    actual = [t for t in tests if t["name"] not in {"Build/Runtime Error", "Build Error", "Runtime Error"}]
    value = grade_tests(row, actual, normalize=normalize, exact=exact)
    if value['resolved']:
        return value

    def lines(pattern):
        return [line.strip() for line in plain.splitlines() if re.search(pattern, line, re.I)][:8]

    dependency = lines(
        r"rosetta error:|Cannot start ChromeHeadless|ChromeHeadless failed .*\(cannot start\)|"
        r"error while loading shared libraries:|(?:bash|sh): .*: command not found|"
        r"Could not resolve host|Temporary failure in name resolution|network is unreachable|"
        r"\bENOTFOUND\b|\bEAI_AGAIN\b|failed to download|could not download|"
        r"fatal error: [^:]+: No such file or directory|"
        r"^\s*(?:E\s+)?ModuleNotFoundError: No module named|^\s*Error: Cannot find package")
    if dependency:
        return {**value, 'status': 'error', 'resolved': None, 'error_type': 'NativeEnvironmentError',
                'error_category': 'runtime_dependency', 'diagnostic_lines': dependency,
                'error': 'Test dependencies/runtime unavailable; required tests were not all validated'}
    compilation = lines(
        r"\.go:\d+:\d+: .*\bundefined\b|"
        r"\.go:\d+:\d+: unknown field .* in struct literal|"
        r"Cannot find module ['\"]\.{1,2}/[^'\"]+['\"] from |"
        r"error\s+Unable to resolve path to module ['\"]\.{1,2}/[^'\"]+['\"]\s+import/no-unresolved|"
        r"error\s+TS2307:.*Cannot find module ['\"]\.{1,2}/|"
        r"error\s+TS2339:.*Property .* does not exist|"
        r"\.(?:ts|tsx)\(\d+,\d+\): error TS(?:2322|2345|2551|2304):|"
        r"error\[E0(?:425|432|433|599|609)\]:|"
        r"ImportError: cannot import name .* from .*\((?!.*(?:site-packages|dist-packages)).+\.py\)|"
        r"AttributeError: (?:module|type object) .* has no attribute")
    required = set(map(normalize, row.get('FAIL_TO_PASS', row.get('fail_to_pass', [])) +
                                 row.get('PASS_TO_PASS', row.get('pass_to_pass', []))))
    failed = [t for t in actual if t['status'] in {'FAILED', 'ERROR'} and normalize(t['name']) in required]
    # The upstream Pro parser can omit Go FAIL lines. An exact required-test
    # failure is sufficient evidence of failure, but never evidence of success.
    go_failures = [line.strip() for line in plain.splitlines()
                   if (match := re.match(r'^\s*--- FAIL: (\S+)\s+\([\d.]+s\)\s*$', line))
                   and normalize(match.group(1)) in required]
    # Jest/mithril summaries cover real assertions even when the pinned parser
    # cannot recover individual names. They can establish failure, never success.
    assertions = lines(r"^\s*Tests:?\s+.*[1-9]\d* failed|^\s*[1-9]\d* out of \d+ assertions failed|"
                       r"^test result: FAILED\. .*\b[1-9]\d* failed;|"
                       r"^\s*[1-9]\d* failing\s*$")
    if compilation or failed or assertions or go_failures:
        value.update(failure_kind='source_compilation' if compilation else 'test_failure',
                     diagnostic_lines=compilation or assertions or go_failures)
        return value
    return {**value, 'status': 'error', 'resolved': None, 'error_type': 'NativeParserError',
            'error_category': 'incomplete_test_results' if actual else 'zero_tests',
            'error': 'Required tests missing without explicit assertion/source failure; inspect raw logs'}


def patch_paths(patch):
    """Ask Git to decode quoted paths, including additions/deletions/renames."""
    raw = command(['git', 'apply', '--numstat', '-z', str(patch.resolve())]).stdout
    parts = iter(raw.split('\0'))
    paths = []
    for part in parts:
        if not part:
            continue
        path = part.split('\t', 2)[2]
        if path:
            paths.append(path)
        else:
            paths.extend([next(parts), next(parts)])
    return sorted(set(paths))


def prediction_command(protected):
    # Official test files belong to the evaluator. Applying agent changes there
    # first causes test-patch conflicts and Pro's hidden-test add/add collisions.
    excludes = ' '.join(shlex.quote('--exclude=' + path.removeprefix('/app/')) for path in protected)
    return ('if test -s /swe-neo-eval/prediction.patch; then git apply -v ' + excludes +
            ' --recount --ignore-space-change --whitespace=nowarn /swe-neo-eval/prediction.patch; fi')


def native_evaluate(row, patch: Path, cache: Path, output: Path, timeout=3600, cpus=4, memory="8g"):
    """Apply the submitted patch in a fresh image and run upstream log parsers."""
    from . import swebench_runtime
    if swebench_runtime.matches(row):
        return swebench_runtime.evaluate(row, patch, cache, output, timeout, cpus, memory)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(patch, output / "prediction.patch")
    q = shlex.quote
    session = None
    phase = "preparation"
    started = time.monotonic()
    try:
        prepare_native(row, cache)
        mounts = [(output.resolve(), "/swe-neo-eval", False)]
        if row.get("image_name"):
            upstream = bootstrap_rebench(cache)
            spec = importlib.util.spec_from_file_location("swe_neo_rebench_eval", upstream / "scripts/eval.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            install = row["install_config"]
            parser = module.get_parser(install["log_parser"])
            (output / "test.patch").write_text(row["test_patch"], encoding="utf-8")
            cmds = module.normalize_command_list(install["test_cmd"], "test_cmd", row["instance_id"])
            script = "\n".join([
                "set -e", "git config core.hooksPath /dev/null",
                "git reset --hard " + q(row["base_commit"]), "git clean -fd",
                "echo __SWE_NEO_PHASE_prediction_patch__",
                prediction_command(patch_paths(output / "test.patch")),
                "echo __SWE_NEO_PHASE_test_patch__",
                "git apply -v --3way --recount --ignore-space-change --whitespace=nowarn /swe-neo-eval/test.patch",
                "echo " + PATCH_READY,
                "export _JAVA_OPTIONS=-Djava.net.preferIPv6Addresses=false", *cmds])
        else:
            identifier = row["instance_id"]
            for name in ("run_script.sh", "parser.py"):
                shutil.copy2(pro_file(cache, f"run_scripts/{identifier}/{name}"), output / name)
            # Docker already applies both legacy and key=value ENV instructions,
            # including expansions and continuations. Inherit the image environment.
            # The official Verified evaluator restores the selected hidden tests
            # using the last upstream before_repo_set_cmd, after applying changes.
            before = (row.get("before_repo_set_cmd") or "").strip().splitlines()
            script = "\n".join([
                "set -e", "git config core.hooksPath /dev/null",
                "git reset --hard " + q(row["base_commit"]), "git clean -fd",
                "git checkout " + q(row["base_commit"]),
                "echo __SWE_NEO_PHASE_prediction_patch__",
                prediction_command(row["selected_test_files_to_run"]),
                "echo __SWE_NEO_PHASE_test_patch__",
                before[-1] if before else "true", "echo " + PATCH_READY,
                "set +e", "bash /swe-neo-eval/run_script.sh " + q(",".join(row["selected_test_files_to_run"])) +
                " > /swe-neo-eval/tests.stdout.log 2> /swe-neo-eval/tests.stderr.log",
                "printf '%s\\n' \"$?\" > /swe-neo-eval/tests.returncode",
                "python /swe-neo-eval/parser.py /swe-neo-eval/tests.stdout.log /swe-neo-eval/tests.stderr.log /swe-neo-eval/tests.json",
                "parser_status=$?", "printf '%s\\n' \"$parser_status\" > /swe-neo-eval/parser.returncode",
                "exit \"$parser_status\""])
        (output / "entryscript.sh").write_text(script + "\n", encoding="utf-8")
        phase = "container_start"
        session = DockerSession(row, output, cpus=cpus, memory=memory, mounts=mounts)
        phase = "test_execution"
        try:
            result = session.exec(script, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            for name, data in [("container.stdout.log", exc.stdout), ("container.stderr.log", exc.stderr)]:
                (output / name).write_text(data.decode(errors="replace") if isinstance(data, bytes)
                                           else data or "", encoding="utf-8")
            raise
        (output / "container.stdout.log").write_text(result.stdout, encoding="utf-8")
        (output / "container.stderr.log").write_text(result.stderr, encoding="utf-8")
        logs = result.stdout + "\n" + result.stderr
        if PATCH_READY not in result.stdout:
            markers = re.findall(r"__SWE_NEO_PHASE_(\w+)__", result.stdout)
            phase = markers[-1] if markers else 'repository_setup'
            raise RuntimeError(f"Native evaluation failed during {phase}; see container logs")
        phase = "test_parsing"
        if row.get("image_name"):
            parsed = parser(logs.split(PATCH_READY, 1)[-1])
            tests = [{"name": key, "status": value} for key, value in parsed.items()]
            value = classify_test_output(row, tests, logs,
                                         normalize=module._normalize_test_name)
            value["rule"] = "all_required_f2p_and_p2p_pass"
        else:
            parsed = json.loads((output / "tests.json").read_text())
            test_logs = logs
            for name in ("tests.stdout.log", "tests.stderr.log"):
                if (output / name).exists():
                    test_logs += "\n" + (output / name).read_text(errors="replace")
            value = classify_test_output(row, parsed["tests"], test_logs)
            value["rule"] = "all_required_f2p_and_p2p_pass"
            for name in ("tests", "parser"):
                path = output / (name + ".returncode")
                if path.exists():
                    value[name + "_returncode"] = int(path.read_text().strip())
        value["returncode"] = result.returncode
    except Exception as exc:
        value = {"status": "error", "resolved": None,
                 "error_type": type(exc).__name__, "error": str(exc),
                 "error_category": "evaluation_timeout" if isinstance(exc, subprocess.TimeoutExpired) else phase}
    finally:
        if session:
            session.close()
    value.update({"instance_id": row["instance_id"], "image": image_for(row)[0],
                  "runtime_revision": RUNTIME_REVISION, "phase": phase,
                  "elapsed_seconds": time.monotonic() - started,
                  "rebench_evaluator_revision": REBENCH_REV,
                  "pro_scripts_revision": PRO_REV, "agentcompass_reference_revision": AGENTCOMPASS_REV})
    write_json(output / "result.json", value)
    return value
