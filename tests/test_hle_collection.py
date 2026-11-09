"""HLE end-to-end collection checks with local tools and scripted model responses."""
from __future__ import annotations

import copy
import io
import json
import shlex
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from mixed_pilot_harness.collection.cli import main
from mixed_pilot_harness.collection.collector import CollectionSettings, collect
from mixed_pilot_harness.collection.episodes import extract_answer
from mixed_pilot_harness.collection.tasks import load_tasks, select_tasks, validate_tasks
from mixed_pilot_harness.config import HarnessConfig, ProviderProfile
from mixed_pilot_harness.types import ProviderError, ProviderResponse, ToolCall, Usage


class Responses:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    async def complete(self, messages, tools, system):
        self.requests.append(copy.deepcopy({'messages': [asdict(m) for m in messages],
                                           'tools': [asdict(t) for t in tools], 'system': system}))
        item = next(self.responses)
        if isinstance(item, Exception):
            raise item
        return item


def response(text='', calls=None, stop='stop'):
    return ProviderResponse(text=text, tool_calls=calls or [], stop_reason=stop, model='mock-returned',
                            usage=Usage(100, 10))


class HLECollectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = self.root / 'tasks/hle-1'
        (self.task / 'workspace').mkdir(parents=True)
        (self.task / 'reference').mkdir()
        (self.task / 'workspace/README.md').write_text('Use Python if helpful.')
        (self.task / 'prompt.md').write_text('What is 2 + 2?')
        (self.task / 'reference/gold.json').write_text(json.dumps({
            'answer': 'FOUR_ONLY_IN_REFERENCE', 'answer_type': 'exactMatch', 'rationale': 'PRIVATE_RATIONALE'}))
        self.manifest = {'id': 'hle-1', 'source': {'dataset': 'HLE', 'upstream_id': 'original-1',
            'split': 'test', 'revision': 'fixture'}, 'prompt': 'prompt.md',
            'workspace': {'path': 'workspace'}, 'evaluation': {'reference_dir': 'reference'},
            'agent': {'tools': ['shell', 'write_file', 'read_file']},
            'routing_metadata': {'router_split': 'train', 'hle_category': 'Math'}}
        self.save_manifest()
        self.config = HarnessConfig(self.root / 'config.json', 'a', {
            'a': ProviderProfile('scripted', 'agent'), 'b': ProviderProfile('scripted', 'judge')})
        self.prices = {'currency': 'CNY', 'as_of': 'fixture', 'profiles': {
            'a': {'input_per_million': 1, 'output_per_million': 2},
            'b': {'input_per_million': 10, 'output_per_million': 20}}}
        self.settings = CollectionSettings('baseline', ['a'], 'a', benchmark='hle', judge_profile='b')
        self.out = self.root / 'out'

    def save_manifest(self):
        (self.task / 'manifest.json').write_text(json.dumps(self.manifest))

    async def run_collection(self, agent, judge=None, tasks=None, settings=None):
        def factory(profile, **kwargs):
            return judge if profile.model == 'judge' else agent
        with redirect_stdout(io.StringIO()):
            summary = await collect(tasks or load_tasks(self.root / 'tasks', benchmark='hle'),
                self.config, settings or self.settings, self.prices, self.out, provider_factory=factory)
        trace = json.loads(next(self.out.glob('*.traj.json')).read_text())
        rows = [json.loads(line) for line in Path(summary['steps_file']).read_text().splitlines()]
        return summary, trace, rows

    async def test_empty_limit_is_scored_failure_without_judge(self):
        agent = Responses([response(stop='length') for _ in range(3)])
        judge = Responses([])
        summary, trace, rows = await self.run_collection(agent, judge)
        self.assertEqual(trace['info']['exit_status'], 'EmptyResponseLimit')
        self.assertEqual(trace['info']['final_score'], 0)
        self.assertFalse(trace['info']['success'])
        self.assertEqual(trace['info']['evaluation']['method'], 'empty_response_limit')
        self.assertEqual(summary['valid_traces'], 1)
        self.assertEqual(summary['success_rate'], 0)
        self.assertAlmostEqual(summary['known_cost'], .00036)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(judge.requests), 0)

    async def test_multistep_tools_judge_cost_history_and_resume(self):
        command = shlex.quote(sys.executable) + ' -c "print(2 + 2)"'
        agent = Responses([
            response(calls=[ToolCall('call-1', 'shell', {'command': command})]),
            response('Explanation: Computed with Python.\nAnswer: 4\nConfidence: 100%')])
        judge = Responses([response('{"correct": true, "reason": "Equivalent answer"}')])
        summary, trace, rows = await self.run_collection(agent, judge)
        self.assertEqual(summary['valid_traces'], 1)
        self.assertEqual(summary['success_rate'], 1)
        self.assertEqual(len(rows), 2)
        self.assertEqual(trace['info']['model_stats']['api_calls'], 2)
        self.assertEqual(trace['info']['evaluation']['api_calls'], 1)
        self.assertAlmostEqual(summary['agent_cost'], .00024)
        self.assertAlmostEqual(summary['evaluation_cost'], .0012)
        self.assertAlmostEqual(summary['known_cost'], .00144)
        self.assertAlmostEqual(rows[0]['remaining_cost'], .00024)
        self.assertEqual(rows[1]['state']['previous_actions'], ['a'])
        self.assertEqual(rows[1]['state']['history'][1]['tool_calls'][0]['id'], 'call-1')
        tool_message = rows[1]['state']['history'][2]
        self.assertEqual(tool_message['role'], 'tool')
        self.assertEqual(tool_message['tool_call_id'], 'call-1')
        self.assertEqual(json.loads(tool_message['content'])['stdout'].strip(), '4')
        self.assertEqual(trace['steps'][0]['tool_results'][0]['tool_call_id'], 'call-1')
        self.assertEqual(rows[0]['source_split'], 'test')
        self.assertEqual(rows[0]['split'], 'train')
        self.assertEqual(rows[0]['trace_id'], trace['info']['trace_id'])
        self.assertNotIn('environment_score', json.dumps(trace))
        self.assertNotIn('FOUR_ONLY_IN_REFERENCE', json.dumps(agent.requests))
        self.assertNotIn('PRIVATE_RATIONALE', json.dumps(judge.requests))
        self.assertIn('FOUR_ONLY_IN_REFERENCE', json.dumps(judge.requests))
        workspace = Path(trace['info']['workspace'])
        self.assertTrue((workspace / 'README.md').is_file())
        self.assertFalse((workspace / 'reference').exists())
        resumed, _, _ = await self.run_collection(Responses([]), Responses([]))
        self.assertEqual(resumed['skipped_existing'], 1)
        self.assertAlmostEqual(resumed['known_cost'], summary['known_cost'])

    async def test_exact_only_is_explicit_and_does_not_call_judge(self):
        self.settings.hle_evaluator = 'exact'
        self.settings.judge_profile = None
        summary, trace, rows = await self.run_collection(Responses([response('Answer: 4')]))
        self.assertEqual(summary['success_rate'], 0)
        self.assertEqual(trace['info']['evaluation']['method'], 'normalized_exact_match')
        self.assertEqual(summary['evaluation_cost'], 0)
        self.assertEqual(len(rows), 1)

    async def test_equal_answer_fast_path(self):
        summary, trace, _ = await self.run_collection(Responses([response('Answer: FOUR_ONLY_IN_REFERENCE')]), Responses([]))
        self.assertEqual(summary['success_rate'], 1)
        self.assertEqual(trace['info']['evaluation']['api_calls'], 0)

    async def test_bad_judge_output_is_billed_but_not_a_failure_label(self):
        summary, trace, rows = await self.run_collection(Responses([response('Answer: 4')]),
                                                        Responses([response('{"correct": "true"}')]))
        self.assertEqual(summary['valid_traces'], 0)
        self.assertEqual(rows, [])
        self.assertEqual(trace['info']['exit_status'], 'EvaluationError')
        self.assertIsNone(trace['info']['success'])
        self.assertTrue(summary['cost_complete'])
        self.assertAlmostEqual(summary['evaluation_cost'], .0012)

    async def test_judge_timeout_marks_cost_unknown(self):
        summary, trace, rows = await self.run_collection(Responses([response('Answer: 4')]),
                                                        Responses([ProviderError('timeout')]))
        self.assertFalse(summary['cost_complete'])
        self.assertEqual(rows, [])
        self.assertEqual(trace['info']['exit_status'], 'EvaluationError')
        self.assertAlmostEqual(summary['known_cost'], .00012)

    async def test_tool_step_limit_is_not_a_submitted_answer(self):
        self.settings.max_steps = 1
        agent = Responses([response(calls=[ToolCall('1', 'write_file', {'path': 'answer.txt', 'content': '4'})])])
        summary, trace, rows = await self.run_collection(agent, Responses([]))
        self.assertEqual(trace['info']['exit_status'], 'LimitsExceeded')
        self.assertEqual(trace['info']['evaluation']['method'], 'no_final_answer')
        self.assertFalse(rows[0]['terminal_success'])
        self.assertEqual(summary['evaluation_cost'], 0)

    async def test_truncated_answer_is_not_graded_as_complete(self):
        self.settings.max_steps = 1
        summary, trace, _ = await self.run_collection(Responses([response('Answer: FOUR_ONLY_IN_REFERENCE', stop='length')]), Responses([]))
        self.assertEqual(summary['success_rate'], 0)
        self.assertEqual(trace['info']['format_errors'], 1)

    async def test_agent_failure_never_calls_judge(self):
        summary, trace, rows = await self.run_collection(Responses([ProviderError('timeout')]), Responses([]))
        self.assertEqual(trace['info']['exit_status'], 'ProviderError')
        self.assertEqual(trace['info']['evaluation']['status'], 'not_run')
        self.assertFalse(summary['cost_complete'])
        self.assertFalse(rows)

    def test_reader_split_hashes_manifest_and_saved_task_list(self):
        rows = load_tasks(self.root, benchmark='hle')
        self.assertEqual(rows[0]['source_split'], 'test')
        self.assertEqual(rows[0]['split'], 'train')
        self.assertNotIn('FOUR_ONLY_IN_REFERENCE', json.dumps(rows))
        saved = self.root / 'saved.json'
        saved.write_text(json.dumps({'tasks': rows}))
        self.assertEqual(rows, load_tasks(saved, benchmark='hle'))
        old_hash = rows[0]['content_hashes']['prompt_path']
        (self.task / 'prompt.md').write_text('Updated question')
        self.assertNotEqual(load_tasks(saved, benchmark='hle')[0]['content_hashes']['prompt_path'], old_hash)
        self.assertEqual(select_tasks(rows, split='train', limit=1), rows)
        with self.assertRaisesRegex(ValueError, 'duplicate HLE instance'):
            validate_tasks([rows[0], {**rows[0], 'task_id': 'alias', 'split': 'test'}])

    def test_reader_rejects_reference_in_workspace_and_symlinks(self):
        self.manifest['evaluation']['reference_dir'] = 'workspace'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'reference must be outside'):
            load_tasks(self.task, benchmark='hle')
        self.manifest['evaluation']['reference_dir'] = 'reference'
        self.save_manifest()
        (self.task / 'workspace/leak').symlink_to(self.task / 'reference')
        with self.assertRaisesRegex(ValueError, 'symlinks'):
            load_tasks(self.task, benchmark='hle')

    def test_hle_dry_run_needs_no_jvm_datasets_or_credentials(self):
        repo = Path(__file__).resolve().parents[1]
        with patch('mixed_pilot_harness.collection.cli.doctor', side_effect=AssertionError('No JVM check')), redirect_stdout(io.StringIO()) as output:
            status = main(['--benchmark', 'hle', '--tasks-file', str(self.root), '--split', 'train',
                '--config', str(repo / 'configs/collection.providers.json'), '--model-pool', 'deepseek-flash',
                '--pricing', str(repo / 'configs/collection.prices.json'), '--mode', 'baseline',
                '--profile', 'deepseek-flash', '--judge-profile', 'deepseek-flash', '--dry-run'])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())['traces'], 1)

    def test_answer_field_not_arbitrary_text(self):
        self.assertEqual(extract_answer('Explanation: math\n**Answer:** 4\nConfidence: 90%'), '4')
        self.assertIsNone(extract_answer('I think 4'))
        self.assertIsNone(extract_answer('Answer: 4\nAnswer: 5'))


if __name__ == '__main__':
    unittest.main()
