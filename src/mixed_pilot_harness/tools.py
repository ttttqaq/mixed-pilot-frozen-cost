"""Workspace-scoped file tools plus a shell bound to the workspace cwd."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .types import ToolDefinition
from .processes import kill_group, noninteractive_env


ToolHandler = Callable[[Dict[str, Any]], Awaitable[str]]


def _inside(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class WorkspaceTools:
    def __init__(
        self,
        workspace: Path,
        timeout: float = 120,
        secret_env_names: Optional[List[str]] = None,
        shell_env: Optional[Dict[str, str]] = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.timeout = timeout
        self.secret_env_names = set(secret_env_names or [])
        self.shell_env = shell_env
        self._handlers: Dict[str, ToolHandler] = {
            "read_file": self.read_file,
            "list_files": self.list_files,
            "search_files": self.search_files,
            "write_file": self.write_file,
            "replace_in_file": self.replace_in_file,
            "shell": self.shell,
            "git_diff": self.git_diff,
        }

    def definitions(self) -> List[ToolDefinition]:
        object_schema = {"type": "object", "additionalProperties": False}
        return [
            ToolDefinition(
                "read_file",
                "Read a UTF-8 text file inside the workspace.",
                {**object_schema, "properties": {"path": {"type": "string"}}, "required": ["path"]},
            ),
            ToolDefinition(
                "list_files",
                "List files below a workspace directory.",
                {
                    **object_schema,
                    "properties": {
                        "path": {"type": "string", "default": "."},
                        "limit": {"type": "integer", "default": 200},
                    },
                },
            ),
            ToolDefinition(
                "search_files",
                "Search UTF-8 workspace files with a regular expression.",
                {
                    **object_schema,
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string", "default": "."},
                        "limit": {"type": "integer", "default": 100},
                    },
                    "required": ["pattern"],
                },
            ),
            ToolDefinition(
                "write_file",
                "Create or replace a UTF-8 file inside the workspace.",
                {
                    **object_schema,
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"],
                },
            ),
            ToolDefinition(
                "replace_in_file",
                "Replace one exact text occurrence in a workspace file.",
                {
                    **object_schema,
                    "properties": {
                        "path": {"type": "string"},
                        "old": {"type": "string"},
                        "new": {"type": "string"},
                    },
                    "required": ["path", "old", "new"],
                },
            ),
            ToolDefinition(
                "shell",
                "Run a shell command with the workspace as cwd. This is not an OS sandbox.",
                {
                    **object_schema,
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "number"},
                    },
                    "required": ["command"],
                },
            ),
            ToolDefinition("git_diff", "Show the current Git diff and untracked files.", object_schema),
        ]

    async def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        handler = self._handlers.get(name)
        if handler is None:
            return json.dumps({"ok": False, "error": "unknown tool: %s" % name})
        try:
            return await handler(arguments)
        except Exception as exc:  # tool errors are observations, not turn crashes
            return json.dumps({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)})

    def _path(self, value: str) -> Path:
        candidate = (self.workspace / value).resolve()
        if not _inside(self.workspace, candidate):
            raise ValueError("path escapes workspace: %s" % value)
        return candidate

    async def read_file(self, arguments: Dict[str, Any]) -> str:
        path = self._path(str(arguments["path"]))
        data = path.read_bytes()
        if len(data) > 250_000:
            raise ValueError("file exceeds 250000-byte read limit")
        return data.decode("utf-8")

    async def list_files(self, arguments: Dict[str, Any]) -> str:
        root = self._path(str(arguments.get("path") or "."))
        limit = max(1, min(int(arguments.get("limit") or 200), 2000))
        if root.is_file():
            return str(root.relative_to(self.workspace))
        paths = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and ".git" not in path.parts:
                paths.append(str(path.relative_to(self.workspace)))
                if len(paths) >= limit:
                    break
        return "\n".join(paths)

    async def search_files(self, arguments: Dict[str, Any]) -> str:
        root = self._path(str(arguments.get("path") or "."))
        pattern = re.compile(str(arguments["pattern"]))
        limit = max(1, min(int(arguments.get("limit") or 100), 1000))
        matches = []
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            if not path.is_file() or ".git" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    matches.append("%s:%d:%s" % (path.relative_to(self.workspace), number, line))
                    if len(matches) >= limit:
                        return "\n".join(matches)
        return "\n".join(matches)

    async def write_file(self, arguments: Dict[str, Any]) -> str:
        path = self._path(str(arguments["path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(arguments["content"]), encoding="utf-8")
        return json.dumps({"ok": True, "path": str(path.relative_to(self.workspace))})

    async def replace_in_file(self, arguments: Dict[str, Any]) -> str:
        path = self._path(str(arguments["path"]))
        old = str(arguments["old"])
        new = str(arguments["new"])
        text = path.read_text(encoding="utf-8")
        count = text.count(old)
        if count != 1:
            raise ValueError("expected exactly one match, found %d" % count)
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        return json.dumps({"ok": True, "path": str(path.relative_to(self.workspace))})

    async def shell(self, arguments: Dict[str, Any]) -> str:
        command = str(arguments["command"])
        timeout = min(float(arguments.get("timeout") or self.timeout), self.timeout)
        env = self.shell_env.copy() if self.shell_env is not None else os.environ.copy()
        for name in self.secret_env_names:
            env.pop(name, None)
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(self.workspace),
            env=noninteractive_env(env),
            stdin=asyncio.subprocess.DEVNULL,
            start_new_session=True,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except BaseException as exc:
            kill_group(process)
            await asyncio.wait_for(process.communicate(), timeout=5)
            if isinstance(exc, asyncio.TimeoutError):
                return json.dumps({"ok": False, "error": "command timed out", "timeout": timeout})
            raise
        return json.dumps(
            {
                "ok": process.returncode == 0,
                "returncode": process.returncode,
                "stdout": stdout.decode("utf-8", errors="replace")[-100_000:],
                "stderr": stderr.decode("utf-8", errors="replace")[-100_000:],
            },
            ensure_ascii=False,
        )

    async def git_diff(self, arguments: Dict[str, Any]) -> str:
        del arguments
        quoted = shlex.quote(str(self.workspace))
        return await self.shell(
            {"command": "git -C %s diff --no-ext-diff --binary; git -C %s status --short" % (quoted, quoted)}
        )
