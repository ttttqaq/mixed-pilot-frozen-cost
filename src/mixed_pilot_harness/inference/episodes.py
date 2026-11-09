"""Adapters retain native ScienceWorld scores, HLE judging, and SWE Docker grading."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace

from ..collection.episodes import HLEEpisode, ScienceWorldEpisode, Transition
from ..collection.collector import load_prompt_config
from ..collection.upstream.scienceworld import ScienceWorldEnvironment
from ..runtime import prepare_workspace
from ..runner import DEFAULT_SYSTEM_PROMPT
from ..types import Message
from . import swe_runtime as native


class HLE(HLEEpisode):
    score = 0.0


class SWE(HLE):
    def __init__(self, task, path, options):
        super().__init__(task, path, options.tool_timeout)
        self.options = options
        self.session = None
        self.root = path.parent
        self.row = json.loads(Path(task['record_path']).read_text())
        _, repo = native.image_for(self.row)
        self.system = DEFAULT_SYSTEM_PROMPT + (
            f'\nShell tools run in a network-disabled task container at {repo}. '
            'Use relative file paths. Do not access evaluation artifacts or alter Git metadata.')

    def start(self):
        o = self.options
        native.prepare_native(self.row, o.cache)
        native.pull_image(self.row, o.cache / 'preparation' / self.task['task_id'], o.pull_timeout)
        source, self.base = native.repo_snapshot(self.row, o.cache)
        self.workspace.parent.mkdir(parents=True, exist_ok=True)
        prepare_workspace(str(source), self.workspace)
        self.session = native.DockerSession(self.row, self.root / 'docker', cpus=o.cpus,
                                           memory=o.memory, workspace=self.workspace.resolve())
        ready = self.session.exec('command -v timeout && command -v git && '
                                  'git config --global --add safe.directory "$(pwd)" && '
                                  'git config core.hooksPath /dev/null', timeout=60)
        if ready.returncode:
            raise RuntimeError('Task container lacks timeout/git or cannot access workspace')
        self.executor = native.DockerWorkspaceTools(self.workspace, self.session, o.tool_timeout,
                                                   self.root / 'tool-outputs', getattr(o, 'max_tool_timeout', 3600))
        allowed = set(self.task['allowed_tools'])
        self.tools = [t for t in self.executor.definitions() if t.name in allowed]
        question = Path(self.task['prompt_path']).read_text().replace(
            'The repository is described in `README.md` but is not present in this pilot workspace.',
            'The repository at the benchmark base commit is in the current directory.')
        return [Message('user', question)]

    async def advance(self, response):
        if response.tool_calls:
            transition = await super().advance(response)
            # Controller log paths differ on replay and are not agent-visible files.
            for result, message in zip(transition.record['tool_results'], transition.messages):
                try:
                    output = json.loads(result['output'])
                except (ValueError, TypeError):
                    continue
                if isinstance(output, dict):
                    output.pop('full_stdout', None)
                    output.pop('full_stderr', None)
                    result['output'] = message.content = json.dumps(output, ensure_ascii=False)
            return transition
        if not response.text.strip() or response.stop_reason in {'length', 'max_tokens'}:
            return Transition([Message('user', 'Continue with a tool action or a complete final answer.')],
                              {'observation': 'Incomplete response'}, format_error=True)
        self.final_response = response.text
        return Transition(done=True, status='Submitted')

    async def evaluate(self, **kwargs):
        o = self.options
        patch = self.root / 'prediction.patch'
        native.export_patch(self.workspace, self.base, patch)
        result = native.native_evaluate(self.row, patch, o.cache, self.root / 'native-eval',
                                        o.eval_timeout, o.cpus, o.memory)
        valid = result['status'] in {'passed', 'failed'}
        return {'status': 'scored' if valid else 'error', 'method': 'swe_native',
                'score': 100.0 if result['status'] == 'passed' else 0.0,
                'success': result['status'] == 'passed', 'cost': 0.0, 'cost_complete': True,
                'api_calls': 0, 'native_evaluation': result}

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None


def factory(options, output):
    def create(task, node, attempt):
        if task['benchmark'] == 'scienceworld':
            return ScienceWorldEpisode(task, load_prompt_config(), ScienceWorldEnvironment)
        if task['benchmark'] == 'alfworld':
            from .virtualworld import ALFWorldEpisode
            return ALFWorldEpisode(task, load_prompt_config())
        path = output / 'workspaces' / node['id'] / f'replay-{attempt}' / 'episode.json'
        # A process may die before recording its attempt. Keep that workspace
        # and its logs intact instead of cloning over it on resume.
        while path.parent.exists():
            attempt += 1
            path = output / 'workspaces' / node['id'] / f'replay-{attempt}' / 'episode.json'
        node['workspace_attempt'] = attempt
        if task['benchmark'] == 'hle':
            return HLE(task, path, options.tool_timeout)
        return SWE(task, path, options)
    return create


def evaluator(options, tasks):
    benchmarks = {t['task_id']: t['benchmark'] for t in tasks}
    async def evaluate(collector, ep, node, job):
        if node['status'] in {'global_budget', 'collection_aborted', 'task_budget'}:
            node.update(labels_valid=False, final_score=None)
            return
        if node['status'] == 'EmptyResponseLimit':
            result = {'status': 'scored', 'method': 'empty_response_limit', 'score': 0.0,
                      'success': False, 'cost': 0.0, 'cost_complete': True, 'api_calls': 0}
        else:
            try:
                result = await ep.evaluate(config=collector.config, pricing=collector.pricing,
                    settings=SimpleNamespace(hle_evaluator=options.hle_evaluator, judge_profile=options.judge_profile),
                    provider_factory=collector.provider_factory)
            except Exception as exc:
                # API and pricing failures are captured inside HLE.evaluate.
                # Exceptions escaping here are local setup/export/grading errors.
                result = {'status': 'error', 'cost': 0.0, 'cost_complete': True, 'api_calls': 0,
                          'error_category': 'local_evaluation',
                          'error': f'{type(exc).__name__}: {exc}'}
        node['evaluation'] = result
        collector.total_cost += result.get('cost', 0.0)
        job['api_cost'] += result.get('cost', 0.0)
        if not result.get('cost_complete', False):
            collector.cost_complete = False
            collector.abort = collector.s.stop_on_unknown_cost
        valid = result['status'] == 'scored'
        node.update(labels_valid=valid, final_score=result.get('score') if valid else None,
                    success=result.get('success') if valid else None)
        if valid:
            node['label_value'] = ((node['final_score'] + 100) / 200 if benchmarks[node['task_id']] == 'scienceworld'
                                   else node['final_score'] / 100)
        else:
            node['status'] = 'evaluation_error'
    return evaluate
