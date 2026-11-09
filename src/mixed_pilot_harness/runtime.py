"""Task-scoped environments and trace lifecycle helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .trace import JsonlTrace
from .processes import run as run_process
from .types import (
    Message,
    ProviderEvent,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
    TurnResult,
    WorkspaceError,
)


def run_command(
    command: List[str],
    cwd: Optional[Path] = None,
    timeout: float = 300,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    try:
        return run_process(
            command,
            cwd=str(cwd) if cwd else None,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise WorkspaceError("command timed out: %s" % " ".join(command)) from exc


def _command_error(*results: subprocess.CompletedProcess) -> str:
    """Join stderr and stdout, since venv reports ensurepip failures on stdout."""
    parts = [
        stream.strip()
        for result in results
        for stream in (result.stderr, result.stdout)
        if stream and stream.strip()
    ]
    return "\n".join(parts) if parts else "no output"


def isolated_process_env() -> Dict[str, str]:
    """Remove the harness Python environment from a task subprocess."""
    env = os.environ.copy()
    prefixes = [env.get("VIRTUAL_ENV", ""), env.get("CONDA_PREFIX", "")]
    entries = env.get("PATH", "").split(os.pathsep)
    env["PATH"] = os.pathsep.join(
        entry
        for entry in entries
        if entry
        and not any(
            prefix and _inside_path(Path(prefix), Path(entry)) for prefix in prefixes
        )
    )
    for name in list(env):
        if name in {
            "VIRTUAL_ENV",
            "PYTHONHOME",
            "PYTHONPATH",
            "PIP_PREFIX",
            "PIP_TARGET",
            "__PYVENV_LAUNCHER__",
        } or name.startswith("CONDA_"):
            env.pop(name, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _inside_path(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def is_git_repo(path: Path) -> bool:
    result = run_command(["git", "-C", str(path), "rev-parse", "--show-toplevel"])
    if result.returncode != 0:
        return False
    return Path(result.stdout.strip()).resolve() == path.resolve()


def prepare_workspace(source: str, destination: Path, shallow: bool = False) -> None:
    source_path = Path(source).expanduser()
    if source_path.exists() and not is_git_repo(source_path):
        shutil.copytree(source_path, destination, symlinks=True)
        return
    command = ["git", "clone", "--no-hardlinks"]
    if shallow:
        command.extend(["--depth", "1"])
    command.extend([str(source_path.resolve()) if source_path.exists() else source, str(destination)])
    result = run_command(command, timeout=600)
    if result.returncode == 0:
        return
    if source_path.exists():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source_path, destination, symlinks=True)
        return
    raise WorkspaceError("git clone failed: %s" % result.stderr.strip()[-2000:])


def copy_untracked(source: Path, destination: Path, files: List[str]) -> None:
    for relative in files:
        src = source / relative
        dst = destination / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_symlink():
            dst.symlink_to(src.readlink())
        elif src.is_file():
            shutil.copy2(src, dst)


@dataclass
class TaskEnvironment:
    name: str
    root: Path

    @property
    def workspace(self) -> Path:
        return self.root / "workspace"

    @property
    def venv(self) -> Path:
        return self.root / "venv"

    def prepare_process(self) -> None:
        (self.root / "tmp").mkdir(exist_ok=True)
        (self.root / "cache").mkdir(exist_ok=True)

    def prepare_venv(self) -> None:
        self.prepare_process()
        python = os.environ.get("MIXED_PILOT_AGENT_PYTHON", "python3")
        env = isolated_process_env()
        result = run_command(
            [python, "-m", "venv", str(self.venv)],
            timeout=300,
            env=env,
        )
        if result.returncode != 0:
            # Some distros ship venv without ensurepip (Debian's python3-venv is a
            # separate package). Fall back to seeding pip from the parent interpreter.
            shutil.rmtree(self.venv, ignore_errors=True)
            fallback = run_command(
                [python, "-m", "venv", "--without-pip", str(self.venv)],
                timeout=300,
                env=env,
            )
            if fallback.returncode == 0:
                fallback = run_command(
                    [
                        python,
                        "-m",
                        "pip",
                        "--python",
                        str(self.venv / "bin" / "python"),
                        "install",
                        "--disable-pip-version-check",
                        "pip",
                        "setuptools",
                        "wheel",
                    ],
                    timeout=300,
                    env=env,
                )
            if fallback.returncode != 0:
                raise WorkspaceError(
                    "could not create %s environment: %s"
                    % (self.name, _command_error(result, fallback))
                )

    def shell_env(self) -> Dict[str, str]:
        env = isolated_process_env()
        if self.venv.is_dir():
            env["VIRTUAL_ENV"] = str(self.venv)
            env["PATH"] = os.pathsep.join([str(self.venv / "bin"), env.get("PATH", "")])
            env["PIP_REQUIRE_VIRTUALENV"] = "1"
        env["TMPDIR"] = str(self.root / "tmp")
        env["XDG_CACHE_HOME"] = str(self.root / "cache")
        return env

    def cleanup(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)


class TaskRuntime:
    """Own the isolated workspaces used by one code-task attempt."""

    def __init__(self, run_dir: Path, source: str, shallow: bool = False) -> None:
        self.run_dir = run_dir
        self.source = source
        self.shallow = shallow
        self.environments_root = run_dir / "environments"
        self.tracer = RuntimeTracer(JsonlTrace(run_dir / "trace" / "events.jsonl"))

    def environment(self, name: str) -> TaskEnvironment:
        return TaskEnvironment(name=name, root=self.environments_root / name)

    def prepare_agent(self, config: Optional[Dict[str, str]] = None) -> TaskEnvironment:
        config = config or {"kind": "process"}
        kind = config.get("kind", "process")
        environment = self.environment("agent")
        environment.root.mkdir(parents=True, exist_ok=False)
        try:
            prepare_workspace(self.source, environment.workspace, shallow=self.shallow)
            if kind == "python_venv":
                environment.prepare_venv()
            elif kind == "process":
                environment.prepare_process()
            else:
                raise WorkspaceError("unsupported agent environment kind: %s" % kind)
            setup_command = config.get("setup_command", "")
            if setup_command:
                setup = run_command(
                    ["/bin/sh", "-lc", setup_command],
                    cwd=environment.workspace,
                    timeout=600,
                    env=environment.shell_env(),
                )
                (self.run_dir / "agent.setup.stdout.txt").write_text(
                    setup.stdout, encoding="utf-8"
                )
                (self.run_dir / "agent.setup.stderr.txt").write_text(
                    setup.stderr, encoding="utf-8"
                )
                if setup.returncode != 0:
                    raise WorkspaceError(
                        "agent environment setup failed: %s" % _command_error(setup)
                    )
        except Exception:
            environment.cleanup()
            raise
        return environment

    def clone_agent_result(
        self,
        agent: TaskEnvironment,
        name: str,
        patch_path: str,
        untracked: List[str],
    ) -> TaskEnvironment:
        environment = self.environment(name)
        environment.root.mkdir(parents=True, exist_ok=False)
        try:
            if is_git_repo(agent.workspace):
                clone = run_command(
                    [
                        "git",
                        "clone",
                        "--no-hardlinks",
                        str(agent.workspace),
                        str(environment.workspace),
                    ],
                    timeout=600,
                )
                if clone.returncode != 0:
                    if environment.workspace.exists():
                        shutil.rmtree(environment.workspace)
                    shutil.copytree(agent.workspace, environment.workspace, symlinks=True)
                else:
                    if patch_path and Path(patch_path).stat().st_size:
                        apply = run_command(
                            ["git", "apply", "--recount", "--whitespace=nowarn", patch_path],
                            cwd=environment.workspace,
                        )
                        if apply.returncode != 0:
                            raise WorkspaceError(
                                "could not apply agent patch for %s: %s"
                                % (name, apply.stderr)
                            )
                    copy_untracked(agent.workspace, environment.workspace, untracked)
            else:
                shutil.copytree(agent.workspace, environment.workspace, symlinks=True)
        except Exception:
            environment.cleanup()
            raise
        return environment


@dataclass(frozen=True)
class IterationTrace:
    turn_id: str
    step: int
    model_leg_id: str
    iteration_id: str


class RuntimeTracer:
    """Build trace records outside TurnRunner while preserving the trace schema."""

    def __init__(self, sink: JsonlTrace) -> None:
        self.sink = sink
        self.task_id = ""
        self.workspace = Path(".")
        self.provider = ""
        self.max_steps = 0
        self.system_prompt = ""
        self.tools: List[ToolDefinition] = []
        self.completed_steps = 0
        self.completed_provider_calls = 0
        self.completed_tool_calls = 0

    @property
    def path(self):
        return self.sink.path

    @property
    def sse_path(self):
        return self.sink.sse_path

    def configure(
        self,
        task_id: str,
        workspace: Path,
        provider: str,
        max_steps: int,
        system_prompt: str,
        tools: List[ToolDefinition],
    ) -> None:
        self.task_id = task_id
        self.workspace = workspace
        self.provider = provider
        self.max_steps = max_steps
        self.system_prompt = system_prompt
        self.tools = tools

    def environment_created(self, environment: TaskEnvironment) -> None:
        self.sink.emit(
            "environment_created",
            node_type="environment",
            node_id="environment:%s" % environment.name,
            parent_id=self.task_id or None,
            task_id=self.task_id or None,
            environment=environment.name,
            workspace=str(environment.workspace),
        )

    def runtime_completed(self, evaluation_status: str, rubric_status: str) -> None:
        self.sink.emit(
            "runtime_completed",
            node_type="runtime",
            node_id="runtime:%s" % self.task_id,
            parent_id=self.task_id,
            task_id=self.task_id,
            evaluation_status=evaluation_status,
            rubric_status=rubric_status,
        )

    def runtime_failed(self, phase: str, error: Exception) -> None:
        self.sink.emit(
            "runtime_failed",
            node_type="runtime",
            node_id="runtime:%s" % self.task_id,
            parent_id=self.task_id or None,
            task_id=self.task_id or None,
            phase=phase,
            error_type=type(error).__name__,
            error=str(error),
            **self.progress(),
        )

    def progress(self):
        return {
            "steps": self.completed_steps,
            "provider_calls": self.completed_provider_calls,
            "tool_calls": self.completed_tool_calls,
        }

    def task_started(self, prompt: str, turn_id: str) -> None:
        self.sink.emit(
            "task_started",
            node_type="task",
            node_id=self.task_id,
            parent_id=None,
            task_id=self.task_id,
            prompt=prompt,
            workspace=str(self.workspace),
            system_prompt=self.system_prompt,
            tools=[asdict(tool) for tool in self.tools],
        )
        self.sink.emit(
            "turn_started",
            node_type="logical_turn",
            node_id=turn_id,
            parent_id=self.task_id,
            task_id=self.task_id,
            turn_id=turn_id,
            provider=self.provider,
            workspace=str(self.workspace),
            max_steps=self.max_steps,
        )

    def iteration_started(
        self, turn_id: str, step: int, messages: List[Message]
    ) -> IterationTrace:
        trace = IterationTrace(
            turn_id=turn_id,
            step=step,
            model_leg_id="%s:leg:%d" % (turn_id, step),
            iteration_id="%s:iteration:%d" % (turn_id, step),
        )
        common = self._common(trace)
        self.sink.emit(
            "model_leg_started",
            node_type="model_leg",
            node_id=trace.model_leg_id,
            parent_id=turn_id,
            **common,
        )
        self.sink.emit(
            "iteration_started",
            node_type="provider_tool_iteration",
            node_id=trace.iteration_id,
            parent_id=trace.model_leg_id,
            messages=[asdict(message) for message in messages],
            **common,
        )
        return trace

    def provider_started(self, trace: IterationTrace) -> None:
        self._sse("provider_call_started", trace, source="provider")

    def provider_event(self, trace: IterationTrace, event: ProviderEvent) -> None:
        self._sse(event.event, trace, source="provider", **event.data)

    def provider_failed(self, trace: IterationTrace, error: Exception) -> None:
        failure = {"error_type": type(error).__name__, "error": str(error)}
        common = self._common(trace)
        self.sink.emit(
            "iteration_failed",
            node_type="provider_tool_iteration",
            node_id=trace.iteration_id,
            parent_id=trace.model_leg_id,
            **common,
            **failure,
        )
        self.sink.emit(
            "model_leg_failed",
            node_type="model_leg",
            node_id=trace.model_leg_id,
            parent_id=trace.turn_id,
            **common,
            **failure,
        )
        self._sse("turn_failed", trace, source="runner", **failure)
        self.sink.emit(
            "task_failed",
            node_type="task",
            node_id=self.task_id,
            parent_id=None,
            task_id=self.task_id,
            **failure,
        )

    def provider_completed(self, trace: IterationTrace, response: ProviderResponse) -> None:
        self.completed_provider_calls += 1
        self._sse(
            "provider_call_completed",
            trace,
            source="provider",
            model=response.model,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            tool_call_count=len(response.tool_calls),
            cost_complete=response.cost_complete,
            api_attempts=response.api_attempts,
            context_management=response.context_management,
        )
        if response.text:
            self._sse("assistant_message", trace, source="provider", text=response.text)

    def tool_use(self, trace: IterationTrace, call: ToolCall) -> None:
        self._sse(
            "tool_use",
            trace,
            source="provider",
            tool_use_id=call.id,
            tool_name=call.name,
            arguments=call.arguments,
        )

    def tool_result(
        self, trace: IterationTrace, call: ToolCall, output: str, is_error: bool
    ) -> None:
        self.completed_tool_calls += 1
        self._sse(
            "tool_result",
            trace,
            source="tool",
            tool_use_id=call.id,
            tool_name=call.name,
            is_error=is_error,
            output=output,
        )

    def iteration_completed(self, trace: IterationTrace) -> None:
        self.completed_steps += 1
        common = self._common(trace)
        self.sink.emit(
            "iteration_completed",
            node_type="provider_tool_iteration",
            node_id=trace.iteration_id,
            parent_id=trace.model_leg_id,
            **common,
        )
        self.sink.emit(
            "model_leg_completed",
            node_type="model_leg",
            node_id=trace.model_leg_id,
            parent_id=trace.turn_id,
            **common,
        )

    def task_completed(self, turn_id: str, result: TurnResult) -> None:
        self.sink.emit(
            "turn_completed",
            node_type="logical_turn",
            node_id=turn_id,
            parent_id=self.task_id,
            task_id=self.task_id,
            **result.to_dict(),
        )
        self.sink.emit(
            "task_completed",
            node_type="task",
            node_id=self.task_id,
            parent_id=None,
            task_id=self.task_id,
            result=result.to_dict(),
        )

    def task_failed(self, turn_id: str, step: int, error: Exception) -> None:
        failure = {"error_type": type(error).__name__, "error": str(error)}
        self.sink.emit(
            "turn_failed",
            node_type="logical_turn",
            node_id=turn_id,
            parent_id=self.task_id,
            task_id=self.task_id,
            turn_id=turn_id,
            step=step,
            **failure,
        )
        self.sink.emit(
            "task_failed",
            node_type="task",
            node_id=self.task_id,
            parent_id=None,
            task_id=self.task_id,
            **failure,
        )

    def emit(self, event: str, **data):
        """Keep compatibility for environment lifecycle events outside the runner."""
        return self.sink.emit(event, **data)

    def _common(self, trace: IterationTrace):
        return {
            "task_id": self.task_id,
            "turn_id": trace.turn_id,
            "model_leg_id": trace.model_leg_id,
            "iteration_id": trace.iteration_id,
            "step": trace.step,
        }

    def _sse(self, event: str, trace: IterationTrace, source: str, **data) -> None:
        self.sink.emit(
            event,
            node_type="sse_event",
            parent_id=trace.iteration_id,
            source=source,
            **self._common(trace),
            **data,
        )
