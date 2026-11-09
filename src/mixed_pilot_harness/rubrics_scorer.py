"""Rubric scoring in a disposable clone of the agent result."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .runtime import TaskEnvironment, run_command


SCORER_SYSTEM_PROMPT = """You are an evaluator, not the task-solving agent.
Inspect the completed workspace and the private rubric files under .mixed-pilot-scorer/.
Do not modify the workspace. Score only the requested criteria using concrete artifact evidence.
Return exactly one JSON object and no markdown."""


@dataclass
class RubricScore:
    status: str
    score: Optional[float] = None
    max_score: float = 0
    pass_threshold: float = 0
    passed: Optional[bool] = None
    criteria: List[Dict[str, Any]] = field(default_factory=list)
    scorer_environment: str = ""
    scorer_turn: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RubricsScorer:
    def __init__(
        self,
        rubrics_path: Path,
        reference_dir: Optional[Path] = None,
        runner_factory: Optional[Callable] = None,
    ) -> None:
        self.rubrics_path = rubrics_path
        self.reference_dir = reference_dir
        self.runner_factory = runner_factory

    async def score(
        self,
        environment: TaskEnvironment,
        run_dir: Path,
        evaluation: Dict[str, Any],
        agent_response: str = "",
    ) -> RubricScore:
        private = environment.workspace / ".mixed-pilot-scorer"
        private.mkdir()
        shutil.copy2(self.rubrics_path, private / "rubrics.json")
        if self.reference_dir and self.reference_dir.is_dir():
            shutil.copytree(self.reference_dir, private / "reference")
        (private / "evaluation.json").write_text(
            json.dumps(evaluation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        # Short-answer tasks deliver the result in the response, not a file. Without
        # this the scorer sees an empty workspace and scores a correct answer zero.
        (private / "agent_response.txt").write_text(agent_response or "", encoding="utf-8")
        rubric = json.loads(self.rubrics_path.read_text(encoding="utf-8"))
        max_score = float(rubric.get("max_score") or 0)
        threshold = float(rubric.get("pass_threshold") or 0)
        criteria: List[Dict[str, Any]] = []
        unresolved = []
        for criterion in rubric.get("criteria") or []:
            method = criterion.get("method")
            points = float(criterion.get("points") or 0)
            if method == "native_test":
                criteria.append(self._native_criterion(criterion, points, evaluation))
            elif method == "manual_diff_review" and self._reference_scope_matches(environment):
                criteria.append(
                    self._criterion(
                        criterion,
                        points,
                        "reference gold patch is present in the merged agent workspace",
                        [".mixed-pilot-scorer/reference/gold.patch"],
                    )
                )
            else:
                unresolved.append(criterion)

        scorer_turn: Dict[str, Any] = {}
        try:
            if unresolved:
                if self.runner_factory is None:
                    return self._result(
                        "partial",
                        max_score,
                        threshold,
                        criteria,
                        environment,
                        error="criteria require a scorer runner",
                    )
                scorer_run_dir = run_dir / "scorer"
                scorer_run_dir.mkdir(exist_ok=True)
                environment.prepare_venv()
                runner = self.runner_factory(environment.workspace, scorer_run_dir)
                runner.set_shell_env(environment.shell_env())
                turn = await runner.run(self._prompt(unresolved))
                scorer_turn = turn.to_dict()
                criteria.extend(self._parse_judgment(turn.text, unresolved))
            return self._result(
                "scored",
                max_score,
                threshold,
                criteria,
                environment,
                scorer_turn=scorer_turn,
            )
        except Exception as exc:
            return self._result(
                "error",
                max_score,
                threshold,
                criteria,
                environment,
                scorer_turn=scorer_turn,
                error=str(exc),
            )

    def _native_criterion(
        self, criterion: Dict[str, Any], points: float, evaluation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Grade one native_test criterion from its own declared test id list.

        Falls back to the broad evaluator status only when the criterion declares no
        expected_ref group, and says so in the reason rather than implying the named
        tests were checked.
        """
        group = str(criterion.get("expected_ref") or "").partition("#")[2].lstrip("/").strip()
        checks = evaluation.get("native_checks") or {}
        check = checks.get(group) if group else None
        if isinstance(check, dict) and check.get("status") in {"passed", "failed", "timeout"}:
            awarded = points if check.get("status") == "passed" else 0.0
            return self._criterion(
                criterion,
                awarded,
                "%s: %s of %s declared tests, status=%s"
                % (
                    group,
                    "all" if check.get("status") == "passed" else "not all",
                    check.get("test_count", "?"),
                    check.get("status"),
                ),
                [check.get("stdout_path", ""), check.get("stderr_path", "")],
            )
        awarded = points if evaluation.get("status") == "passed" else 0.0
        if isinstance(check, dict) and check.get("status") == "not_configured":
            reason = (
                "%s not verified by test id (%s); fell back to broad evaluator status=%s"
                % (group, check.get("error", "no test command"), evaluation.get("status", "unknown"))
            )
        elif group:
            reason = "%s unavailable; fell back to broad evaluator status=%s" % (
                group,
                evaluation.get("status", "unknown"),
            )
        else:
            reason = "no expected_ref declared; broad evaluator status=%s" % evaluation.get(
                "status", "unknown"
            )
        return self._criterion(
            criterion,
            awarded,
            reason,
            [evaluation.get("stdout_path", ""), evaluation.get("stderr_path", "")],
        )

    def _reference_scope_matches(self, environment: TaskEnvironment) -> bool:
        if not self.reference_dir:
            return False
        gold = self.reference_dir / "gold.patch"
        if not gold.is_file() or not gold.stat().st_size:
            return False
        check = run_command(
            [
                "git",
                "apply",
                "--reverse",
                "--check",
                "--recount",
                "--whitespace=nowarn",
                str(gold.resolve()),
            ],
            cwd=environment.workspace,
        )
        if check.returncode != 0:
            content = gold.read_text(encoding="utf-8")
            normalized = "".join(
                line
                for line in content.splitlines(keepends=True)
                if not (line.startswith("+") and not line[1:].strip())
            )
            if normalized == content:
                return False
            normalized_gold = environment.workspace / ".mixed-pilot-scorer/gold.patch"
            normalized_gold.write_text(normalized, encoding="utf-8")
            check = run_command(
                [
                    "git",
                    "apply",
                    "--reverse",
                    "--check",
                    "--recount",
                    "--whitespace=nowarn",
                    str(normalized_gold),
                ],
                cwd=environment.workspace,
            )
            if check.returncode != 0:
                return False
        allowed = self._patch_files(gold)
        test_patch = self.reference_dir / "test.patch"
        if test_patch.is_file():
            allowed.update(self._patch_files(test_patch))
        changed = run_command(
            ["git", "diff", "--name-only", "HEAD"], cwd=environment.workspace
        )
        untracked = run_command(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=environment.workspace,
        )
        if changed.returncode != 0 or untracked.returncode != 0:
            return False
        actual = {
            item
            for item in (changed.stdout + untracked.stdout).splitlines()
            if item and not item.startswith(".mixed-pilot-scorer/")
        }
        return actual.issubset(allowed)

    @staticmethod
    def _patch_files(path: Path) -> set:
        files = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("+++ b/"):
                files.add(line[6:])
        return files

    @staticmethod
    def _criterion(
        criterion: Dict[str, Any], score: float, reason: str, evidence: List[str]
    ) -> Dict[str, Any]:
        return {
            "id": criterion.get("id"),
            "method": criterion.get("method"),
            "score": score,
            "max_score": float(criterion.get("points") or 0),
            "reason": reason,
            "evidence": [item for item in evidence if item],
        }

    @staticmethod
    def _prompt(criteria: List[Dict[str, Any]]) -> str:
        requested = json.dumps(criteria, indent=2, ensure_ascii=False)
        return """Score the completed task artifacts in this workspace.

Private evaluation inputs are available at:
- .mixed-pilot-scorer/rubrics.json
- .mixed-pilot-scorer/evaluation.json
- .mixed-pilot-scorer/agent_response.txt (the agent's final response; empty if it
  produced none). For tasks whose deliverable is an answer rather than a file,
  this is the submission to grade.
- .mixed-pilot-scorer/reference/ when supplied

Score only these unresolved criteria:
%s

Return this exact shape:
{"criteria":[{"id":"criterion id","score":0,"reason":"brief reason","evidence":["relative/path"]}]}

Each score must be between 0 and that criterion's points. Use workspace-relative evidence paths.
""" % requested

    def _parse_judgment(
        self, text: str, unresolved: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        stripped = text.strip()
        payload = self._extract_judgment_object(stripped)
        items = payload.get("criteria")
        if not isinstance(items, list):
            raise ValueError("scorer JSON must contain a criteria array")
        returned = {}
        for item in items:
            if not isinstance(item, dict) or item.get("id") is None:
                raise ValueError("each scorer criterion must be an object with an id")
            criterion_id = str(item["id"])
            if criterion_id in returned:
                raise ValueError("scorer returned duplicate rubric criterion %s" % criterion_id)
            returned[criterion_id] = item
        result = []
        for criterion in unresolved:
            criterion_id = str(criterion.get("id"))
            if criterion_id not in returned:
                raise ValueError("scorer omitted rubric criterion %s" % criterion_id)
            judgment = returned[criterion_id]
            maximum = float(criterion.get("points") or 0)
            awarded = max(0.0, min(maximum, float(judgment.get("score") or 0)))
            result.append(
                self._criterion(
                    criterion,
                    awarded,
                    str(judgment.get("reason") or ""),
                    [str(item) for item in judgment.get("evidence") or []],
                )
            )
        return result

    @staticmethod
    def _extract_judgment_object(text: str) -> Dict[str, Any]:
        """Accept strict JSON or one JSON object surrounded by prose/code fences."""
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        for offset, character in enumerate(text):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(text[offset:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and isinstance(candidate.get("criteria"), list):
                return candidate
        raise ValueError("scorer response does not contain a valid JSON criteria object")

    @staticmethod
    def _result(
        status: str,
        max_score: float,
        threshold: float,
        criteria: List[Dict[str, Any]],
        environment: TaskEnvironment,
        scorer_turn: Optional[Dict[str, Any]] = None,
        error: str = "",
    ) -> RubricScore:
        score = sum(float(item.get("score") or 0) for item in criteria)
        complete = status == "scored"
        return RubricScore(
            status=status,
            score=score,
            max_score=max_score,
            pass_threshold=threshold,
            passed=score >= threshold if complete else None,
            criteria=criteria,
            scorer_environment=str(environment.workspace),
            scorer_turn=scorer_turn or {},
            error=error,
        )
