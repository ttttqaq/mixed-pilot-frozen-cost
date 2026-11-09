"""Regression cases from run_0929; no Docker, downloads or paid APIs."""
import asyncio
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, AsyncMock, Mock

from mixed_pilot_harness.inference import swe_runtime as native
from mixed_pilot_harness.inference.cli import execute
from mixed_pilot_harness.inference.maintenance import evaluate_saved
from mixed_pilot_harness.config import ProviderProfile
from mixed_pilot_harness.types import Message, ToolCall, ProviderError
from test_providers import CaptureOpenAI, TOOL
import test_inference as fixtures
from test_inference import Episode, Provider
import test_swe_neo_batch as swe_fixtures


class ClassificationTests(unittest.TestCase):
    row = {'fail_to_pass':['f'], 'pass_to_pass':['p']}

    def classify(self, logs, tests=()):
        return native.classify_test_output(self.row, list(tests), logs)

    def test_downlevel_jest_assertion_is_failed_even_when_parser_empty(self):
        r=self.classify('FAIL ./index.test.js\nTests:       1 failed, 1 total\n')
        self.assertEqual(r['status'],'failed')
        self.assertEqual(r['failure_kind'],'test_failure')

    def test_extra_passing_tests_do_not_fail_a_solution(self):
        tests=[{'name':n,'status':'PASSED'} for n in ['f','p','agent-added']]
        self.assertEqual(self.classify('',tests)['status'],'passed')

    def test_partial_parser_and_sentinel_are_unscored(self):
        for tests in ([{'name':'f','status':'PASSED'}], [{'name':'Build/Runtime Error','status':'FAILED'}]):
            self.assertEqual(self.classify('',tests)['status'],'error')

    def test_dependency_errors_override_partial_test_failures(self):
        tests=[{'name':'f','status':'FAILED'}]
        for text in ["E   ModuleNotFoundError: No module named 'adblock'",
                     'maven.org: Temporary failure in name resolution',
                     'dial tcp: connect: network is unreachable']:
            r=self.classify(text,tests)
            self.assertEqual(r['status'],'error')
            self.assertEqual(r['error_category'],'runtime_dependency')

    def test_mocked_dependency_exception_is_not_an_environment_error(self):
        r=self.classify('E   assert SelectionInfo(pyqt6="ModuleNotFoundError: No module named PyQt6")',
                        [{'name':'f','status':'FAILED'}])
        self.assertEqual(r['status'],'failed')

    def test_missing_local_python_symbol_and_typescript_type_error_are_failed(self):
        for text in ["E   ImportError: cannot import name 'get_non_isbn_asin' from 'openlibrary.utils' (/app/openlibrary/utils/__init__.py)",
                     '../src/api/Blob.ts(77,4): error TS2322: Type null is not assignable to string']:
            self.assertEqual(self.classify(text)['status'],'failed')
        self.assertEqual(self.classify("ImportError: cannot import name 'x' from 'lib' (/usr/site-packages/lib.py)")['status'],'error')


class PatchTests(unittest.TestCase):
    setUp = swe_fixtures.SWENeoBatchTests.setUp
    repo = swe_fixtures.SWENeoBatchTests.repo

    def test_official_tests_override_agent_tests_without_losing_source_changes(self):
        repo,base=self.repo()
        (repo/'test.txt').write_text('base test\n')
        native.command(['git','-C',str(repo),'add','.'])
        native.command(['git','-C',str(repo),'-c','user.name=Test','-c','user.email=t@x','commit','-qm','test base'])
        base=native.command(['git','-C',str(repo),'rev-parse','HEAD']).stdout.strip()
        (repo/'answer.txt').write_text('agent fix\n')
        (repo/'test.txt').write_text('agent test\n')
        pred=self.root/'prediction.patch';native.export_patch(repo,base,pred)
        native.command(['git','-C',str(repo),'reset','--hard',base])
        (repo/'test.txt').write_text('official test\n')
        official=self.root/'official.patch';native.export_patch(repo,base,official)
        native.command(['git','-C',str(repo),'reset','--hard',base])
        command=native.prediction_command(native.patch_paths(official)).replace('/swe-neo-eval',str(self.root))
        native.command(['bash','-c',command],cwd=repo)
        native.command(['git','-C',str(repo),'apply',str(official)])
        self.assertEqual((repo/'answer.txt').read_text(),'agent fix\n')
        self.assertEqual((repo/'test.txt').read_text(),'official test\n')


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    def provider(self, **kwargs):
        p=CaptureOpenAI(ProviderProfile('openai','test',base_url='https://a.test',api_key_optional=True,**kwargs))
        p.response={'choices':[{'message':{'content':'done'}}]}
        return p

    async def test_reasoning_round_trip_and_provider_switch(self):
        p=self.provider(extra_body={'thinking':{'type':'enabled'}})
        p.response={'choices':[{'message':{'reasoning_content':'work', 'tool_calls':[
            {'id':'x','function':{'name':'read_file','arguments':'{"path":"a"}'}}]}}]}
        r=await p.complete([Message('user','task')],[TOOL],'system')
        history=[Message('user','task'),r.assistant_message(),Message('tool','a',tool_call_id='x')]
        await p.complete(history,[TOOL],'system')
        self.assertEqual(p.captured['payload']['messages'][2]['reasoning_content'],'work')
        p.profile.base_url='https://b.test'
        await p.complete(history,[TOOL],'system')
        self.assertEqual(p.captured['payload']['messages'][2]['reasoning_content'],'')
        self.assertEqual(history[1].reasoning_content,'work')

    async def test_malformed_arguments_keep_usage_and_do_not_execute(self):
        from mixed_pilot_harness.collection.episodes import HLEEpisode
        p=self.provider()
        for raw in ['{"path":', '[]', 'null']:
            p.response={'usage':{'prompt_tokens':100,'completion_tokens':20},'choices':[{'message':{'tool_calls':[
                {'id':'x','function':{'name':'read_file','arguments':raw}}]}}]}
            r=await p.complete([],[],'')
            self.assertEqual(r.usage.output_tokens,20)
            self.assertEqual(r.tool_calls[0].raw_arguments,raw)
            ep=object.__new__(HLEEpisode);ep.tools=[TOOL]
            transition=await ep.advance(r)  # executor intentionally absent
            self.assertIn('Invalid tool arguments',transition.messages[0].content)

    async def test_context_removes_complete_tool_groups_and_retains_original_trace(self):
        p=self.provider(context_window=2200,max_tokens=100)
        messages=[Message('user','task'),Message('assistant',tool_calls=[ToolCall('old','read_file',{})]),
                  Message('tool','x'*4000,tool_call_id='old'),
                  Message('assistant',tool_calls=[ToolCall('new','read_file',{})]),
                  Message('tool','latest',tool_call_id='new')]
        r=await p.complete(messages,[TOOL],'system')
        wire=p.captured['payload']['messages']
        self.assertEqual([m.get('tool_call_id') for m in wire if m['role']=='tool'],['new'])
        self.assertEqual(len(messages),5)
        self.assertEqual(r.context_management['dropped_messages'],2)
        r=await p.complete([Message('user','x'*4000)],[],'system')
        self.assertTrue(r.context_management['lossy'])
        self.assertLessEqual(r.context_management['conservative_total_tokens'],2200)

    async def test_shell_requested_timeout_can_exceed_default(self):
        with tempfile.TemporaryDirectory() as folder:
            process=Mock(returncode=0,wait=AsyncMock(return_value=0))
            session=Mock(repo_dir='/app',name='test')
            tools=native.DockerWorkspaceTools(Path(folder),session,30,Path(folder)/'logs',max_timeout=600)
            with patch('asyncio.create_subprocess_exec',AsyncMock(return_value=process)) as spawn:
                await tools.shell({'command':'npm test','timeout':400})
                self.assertIn('400',spawn.call_args.args)
                await tools.shell({'command':'npm test','timeout':1000})
                self.assertIn('600',spawn.call_args.args)


class EngineTests(unittest.TestCase):
    setUp = fixtures.Tests.setUp
    run_engine = fixtures.Tests.run_engine

    def test_partial_resume_adds_tasks_retries_partial_and_updates_inventory(self):
        with self.fail_first_evaluation():
            self.run_engine()
        root=self.args.output_dir
        original_call=(root/'calls/c00000000.json').read_bytes()
        new=dict(task_id='three',benchmark='scienceworld',split='train')
        expanded=self.tasks+[new]
        self.args.resume_partial=True
        report,traces,steps,calls=self.run_engine(tasks=expanded)
        self.assertEqual(len(calls),8)  # One partial + one new; complete skipped.
        self.assertEqual(report['total_tasks'],3)
        self.assertEqual(report['planned_jobs'],3)
        self.assertEqual(report['started_jobs'],3)
        self.assertEqual(report['unstarted_jobs'],0)
        self.assertEqual(report['scored_traces'],3)
        self.assertEqual(report['pass_rate'],1)
        self.assertAlmostEqual(report['total_api_cost'],.32)
        self.assertEqual((root/'calls/c00000000.json').read_bytes(),original_call)
        plan=json.loads((root/'plan.json').read_text())
        self.assertEqual(plan['tasks'],expanded)
        self.assertEqual(plan['task_extensions'][0]['added_task_ids'],['three'])
        self.assertIn('Total tasks: 3',(root/'SUMMARY.md').read_text())
        self.assertEqual(json.loads((root/'summary.json').read_text())['total_tasks'],3)
        # The new inventory is durable, including ordinary resume afterwards.
        self.args.resume_partial=False;self.args.resume=True
        _,_,_,calls=self.run_engine(tasks=expanded)
        self.assertEqual(calls,[])
        self.args.resume=False;self.args.resume_partial=True
        with self.assertRaisesRegex(ValueError,'existing tasks must remain unchanged'):
            self.run_engine(tasks=self.tasks)
        expanded=expanded+[dict(new,task_id='four')]
        report,_,_,calls=self.run_engine(tasks=expanded)
        self.assertEqual(len(calls),4)
        self.assertEqual(report['total_tasks'],4)
        self.assertEqual(len(json.loads((root/'plan.json').read_text())['task_extensions']),2)

    def test_partial_resume_rejects_changed_missing_and_duplicate_tasks(self):
        self.run_engine()
        root=self.args.output_dir
        original=(root/'plan.json').read_bytes()
        extra=dict(task_id='extra',benchmark='scienceworld',split='train')
        self.args.resume=True
        with self.assertRaisesRegex(ValueError,'use --resume-partial'):
            self.run_engine(tasks=self.tasks+[extra])
        self.args.resume=False;self.args.resume_partial=True
        for tasks in [self.tasks[:1]+[extra],
                      [dict(self.tasks[0],content_hashes={'prompt':'changed'}),self.tasks[1],extra],
                      self.tasks+[self.tasks[0]]]:
            with self.subTest(tasks=tasks),self.assertRaises(ValueError):
                self.run_engine(tasks=tasks)
            self.assertEqual((root/'plan.json').read_bytes(),original)

    def test_expansion_runs_new_tasks_in_every_round(self):
        self.run_engine(tasks=self.tasks[:1],runs=2)
        self.args.resume_partial=True
        report,_,_,calls=self.run_engine(tasks=self.tasks,runs=2)
        self.assertEqual(len(calls),8)
        self.assertEqual(report['total_tasks'],2)
        self.assertEqual(report['planned_jobs'],4)
        self.assertEqual(report['started_jobs'],4)
        self.assertEqual(report['unstarted_jobs'],0)

    def test_expansion_keeps_unstarted_tasks_and_budget(self):
        self.run_engine(total_cost_limit=.08)
        self.args.resume_partial=True
        expanded=self.tasks+[dict(task_id='extra',benchmark='scienceworld',split='train')]
        report,_,_,calls=self.run_engine(tasks=expanded,total_cost_limit=.08)
        self.assertEqual(calls,[])  # Expansion must not reset exhausted budget.
        self.assertEqual(report['total_tasks'],3)
        self.assertEqual(report['planned_jobs'],3)
        self.assertEqual(report['started_jobs'],1)
        self.assertEqual(report['unstarted_jobs'],2)
        self.assertAlmostEqual(report['total_api_cost'],.08)

    def test_expansion_runs_old_unstarted_and_new_tasks_after_interruption(self):
        from mixed_pilot_harness.branch_sampling.core import Collector
        original=Collector.run_job
        async def interrupt(collector,task,run):
            if task['task_id']==self.tasks[1]['task_id']:
                raise KeyboardInterrupt()
            return await original(collector,task,run)
        with patch.object(Collector,'run_job',interrupt),self.assertRaises(KeyboardInterrupt):
            self.run_engine()
        self.args.resume_partial=True
        expanded=self.tasks+[dict(task_id='extra',benchmark='scienceworld',split='train')]
        report,_,_,calls=self.run_engine(tasks=expanded)
        self.assertEqual(len(calls),8)
        self.assertEqual(report['scored_traces'],3)
        self.assertEqual(report['unstarted_jobs'],0)
        self.assertAlmostEqual(report['total_api_cost'],.24)

    def test_resume_allows_recovery_settings_changes_and_legacy_plan(self):
        self.run_engine()
        root=self.args.output_dir
        plan_path=root/'plan.json'
        saved=json.loads(plan_path.read_text())
        # Runs created before retry timing existed must remain resumable.
        saved['settings'].pop('retry_wait')
        saved['settings'].pop('trace_delay')
        plan_path.write_text(json.dumps(saved))
        original_plan=plan_path.read_bytes()
        original_call=(root/'calls/c00000000.json').read_bytes()
        self.args.resume=True
        with patch('mixed_pilot_harness.branch_sampling.core.asyncio.sleep',new_callable=AsyncMock) as sleep:
            report,_,_,calls=self.run_engine(replay_attempts=11,retry_wait=10,trace_delay=3)
        self.assertEqual(calls,[])
        sleep.assert_not_awaited()
        self.assertEqual(plan_path.read_bytes(),original_plan)
        self.assertEqual((root/'calls/c00000000.json').read_bytes(),original_call)
        events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
        self.assertEqual(events[-1]['settings'],dict(replay_attempts=11,retry_wait=10,trace_delay=3))

    def test_resume_still_rejects_sampling_and_budget_changes(self):
        self.run_engine()
        self.args.resume=True
        for options in [dict(seed=99),dict(max_steps=9),dict(total_cost_limit=200)]:
            with self.subTest(options=options),self.assertRaisesRegex(ValueError,'original tasks/settings'):
                self.run_engine(replay_attempts=11,retry_wait=10,**options)

    def test_resume_partial_allows_new_step_limit_and_preserves_old_results(self):
        with self.fail_first_evaluation():
            self.run_engine()
        root=self.args.output_dir
        original_plan=(root/'plan.json').read_bytes()
        original_call=(root/'calls/c00000000.json').read_bytes()
        self.args.resume_partial=True
        report,traces,_,calls=self.run_engine(max_steps=500)
        self.assertEqual(len(calls),4)  # Only the partial task is retried.
        self.assertEqual(report['scored_traces'],2)
        self.assertAlmostEqual(report['total_api_cost'],.24)
        self.assertEqual((root/'plan.json').read_bytes(),original_plan)
        self.assertEqual((root/'calls/c00000000.json').read_bytes(),original_call)
        tree=json.loads((root/'tree.json').read_text())
        self.assertEqual([n['config_snapshot']['settings']['max_steps'] for n in tree['nodes']],[8,8,500])
        self.assertEqual(json.loads((root/'resume-config.json').read_text())['settings']['max_steps'],500)
        for options in [dict(seed=99),dict(total_cost_limit=200)]:
            with self.subTest(options=options),self.assertRaisesRegex(ValueError,'original tasks/settings'):
                self.run_engine(max_steps=500,**options)

    def test_step_limit_change_restarts_running_job_in_partial_resume(self):
        self.run_engine()
        root=self.args.output_dir
        tree=json.loads((root/'tree.json').read_text())
        tree['jobs'][0]['status']='running'
        tree['nodes'][0]['status']='running'
        (root/'tree.json').write_text(json.dumps(tree))
        self.args.resume_partial=True
        report,_,_,calls=self.run_engine(max_steps=500)
        self.assertEqual(len(calls),4)
        self.assertEqual(report['superseded_nodes'],1)
        self.assertEqual(report['jobs'][0]['attempt'],1)

    def test_resume_partial_uses_updated_recovery_settings(self):
        with self.fail_first_evaluation():
            self.run_engine()
        self.args.resume_partial=True
        with patch('mixed_pilot_harness.branch_sampling.core.asyncio.sleep',new_callable=AsyncMock) as sleep:
            report,_,_,calls=self.run_engine(replay_attempts=11,retry_wait=10,trace_delay=3)
        self.assertEqual(len(calls),4)
        self.assertEqual(report['scored_traces'],2)
        sleep.assert_awaited_once_with(3)

    def fail_first_evaluation(self):
        original=Episode.evaluate
        attempted=[]
        async def evaluate(ep, **kwargs):
            attempted.append(True)
            if len(attempted)==1:
                raise OSError('local evaluation interrupted')
            return await original(ep, **kwargs)
        return patch.object(Episode,'evaluate',evaluate)

    def test_resume_partial_restarts_only_partial_and_retains_costs(self):
        with self.fail_first_evaluation():
            self.run_engine()
        root=self.args.output_dir
        old_call=(root/'calls/c00000000.json').read_bytes()
        self.args.resume=True
        report,_,_,calls=self.run_engine()
        self.assertEqual(calls,[])  # Normal resume still skips partial jobs.
        self.args.resume=False;self.args.resume_partial=True
        report,traces,steps,calls=self.run_engine()
        self.assertEqual(len(calls),4)
        self.assertEqual(report['passed'],2)
        self.assertEqual(report['total_traces'],2)
        self.assertEqual(report['unscored_traces'],0)
        self.assertEqual(report['superseded_nodes'],1)
        self.assertAlmostEqual(report['total_api_cost'],.24)
        self.assertAlmostEqual(report['jobs'][0]['api_cost'],.16)
        self.assertEqual(report['jobs'][0]['previous_attempts'][0]['status'],'partial')
        self.assertEqual((root/'calls/c00000000.json').read_bytes(),old_call)
        self.assertEqual(len(traces),3)
        self.assertEqual(sum(t['info']['superseded'] for t in traces),1)
        self.assertEqual(len(steps),8)
        self.assertNotIn('n000000',{s['trace_id'] for s in steps})
        self.assertEqual(json.loads((root/'leaf-data/summary.json').read_text())['traces'],2)
        report,_,_,calls=self.run_engine()
        self.assertEqual(calls,[])  # Successful retries are not run again.
        self.assertAlmostEqual(report['total_api_cost'],.24)

    def test_resume_partial_retries_failures_once_per_invocation(self):
        async def broken(*a,**kw):raise OSError('still broken')
        with patch.object(Episode,'evaluate',broken):
            self.run_engine()
            self.args.resume_partial=True
            for attempt in (1,2):
                report,_,_,calls=self.run_engine()
                self.assertEqual(len(calls),8)
                self.assertEqual(len(report['jobs']),2)
                self.assertEqual(report['jobs'][0]['attempt'],attempt)
                self.assertEqual(len(report['jobs'][0]['previous_attempts']),attempt)
                self.assertAlmostEqual(report['total_api_cost'],.16*(attempt+1))

    def test_resume_partial_ksr_replaces_whole_job_without_mixing_labels(self):
        with self.fail_first_evaluation():
            self.run_engine(mode='ksr',pool=['a','b'],max_depth=0)
        self.args.resume_partial=True
        report,_,steps,calls=self.run_engine(mode='ksr',pool=['a','b'],max_depth=0)
        self.assertEqual(len(calls),8)
        self.assertEqual(report['superseded_nodes'],2)
        self.assertEqual(report['total_traces'],4)
        self.assertEqual(len(steps),16)
        self.assertFalse({'n000000','n000001'} & {s['trace_id'] for s in steps})
        samples=[json.loads(s) for s in (self.args.output_dir/'samples.jsonl').read_text().splitlines()]
        self.assertEqual(len(samples),4)
        self.assertTrue(all(s['paired_actions_complete'] for s in samples))

    def test_resume_partial_flag_is_standalone(self):
        from mixed_pilot_harness.inference.cli import parser
        args=parser().parse_args(['--resume-partial','--output-dir','existing'])
        self.assertTrue(args.resume_partial)
        self.assertFalse(args.resume)

    def test_resume_partial_does_not_retry_scored_model_failures(self):
        report,_,_,_=self.run_engine(pool=['b'])
        self.assertEqual(report['passed'],0)
        self.assertTrue(all(j['status']=='complete' for j in report['jobs']))
        self.args.resume_partial=True
        report,_,_,calls=self.run_engine(pool=['b'])
        self.assertEqual(calls,[])
        self.assertEqual(report['superseded_nodes'],0)

    def test_local_evaluation_error_keeps_cost_and_continues(self):
        async def broken(*args,**kwargs):raise OSError('local parser missing')
        with patch.object(Episode,'evaluate',broken):
            report,traces,steps,calls=self.run_engine()
        self.assertEqual(len(calls),8)
        self.assertTrue(report['cost_complete'])
        self.assertEqual(report['unscored_traces'],2)
        self.assertEqual(report['stop_reason'],'tasks_finished')

    def test_resume_replays_saved_steps_without_rebilling(self):
        report,_,_,_=self.run_engine()
        root=self.args.output_dir
        tree=json.loads((root/'tree.json').read_text())
        # Simulate interruption after step 2, before the next provider call.
        tree['nodes']=tree['nodes'][:1];tree['jobs']=tree['jobs'][:1]
        tree['nodes'][0].update(own_call_ids=['c00000000','c00000001'],status='running',labels_valid=False,final_score=None)
        tree['nodes'][0].pop('evaluation',None)
        tree['jobs'][0]['status']='running'
        (root/'tree.json').write_text(json.dumps(tree))
        for p in (root/'calls').glob('*.json'):
            if p.stem not in {'c00000000','c00000001'}:p.unlink()
        kept=(root/'calls/c00000000.json').read_bytes()
        self.args.resume=True
        result,_,_,calls=self.run_engine()
        self.assertEqual(len(calls),6)
        self.assertEqual(result['api_attempts'],8)
        self.assertAlmostEqual(result['total_api_cost'],.16)

        self.assertEqual((root/'calls/c00000000.json').read_bytes(),kept)
        self.assertEqual(result['pass_rate'],1)
        result,_,_,calls=self.run_engine()
        self.assertEqual(calls,[])
        self.assertAlmostEqual(result['total_api_cost'],.16)

    def test_random_resume_preserves_rng_sequence(self):
        _,_,_,original=self.run_engine(mode='random',pool=['a','b','c'])
        root=self.args.output_dir
        tree=json.loads((root/'tree.json').read_text())
        tree['nodes']=tree['nodes'][:1];tree['jobs']=tree['jobs'][:1]
        tree['nodes'][0].update(own_call_ids=['c00000000','c00000001'],status='running',labels_valid=False,final_score=None)
        tree['nodes'][0].pop('evaluation',None);tree['jobs'][0]['status']='running'
        rng=random.Random(42)
        for _ in range(2):rng.choice(['a','b','c'])
        tree['rng_state']=rng.getstate()
        (root/'tree.json').write_text(json.dumps(tree))
        for p in (root/'calls').glob('*.json'):
            if p.stem not in {'c00000000','c00000001'}:p.unlink()
        self.args.resume=True
        result,_,_,calls=self.run_engine(mode='random',pool=['a','b','c'])
        self.assertEqual(calls,original[2:])
        self.assertAlmostEqual(result['total_api_cost'],.16)

    def test_initialization_is_not_replay_failure(self):
        with patch.object(Episode,'start',side_effect=OSError('cannot initialize')):
            result,traces,_,calls=self.run_engine()
        self.assertEqual(calls,[])
        self.assertTrue(all(t['info']['exit_status']=='initialization_error' for t in traces))

    def test_resume_preserves_previous_workspace(self):
        from mixed_pilot_harness.inference.episodes import factory
        old=self.args.output_dir/'workspaces/n000000/replay-1'
        old.mkdir(parents=True)
        (old/'evidence').write_text('keep')
        node={'id':'n000000'}
        ep=factory(self.args,self.args.output_dir)({'benchmark':'hle'},node,1)
        self.assertIn('replay-2',str(ep.workspace))
        self.assertEqual((old/'evidence').read_text(),'keep')

    def test_resume_recovers_usage_without_repeating_uncertain_action(self):
        self.run_engine()
        root=self.args.output_dir
        tree=json.loads((root/'tree.json').read_text())
        tree['nodes']=tree['nodes'][:1];tree['jobs']=tree['jobs'][:1]
        tree['nodes'][0].update(status='running',own_call_ids=[],labels_valid=False,final_score=None)
        tree['nodes'][0].pop('evaluation',None)
        tree['jobs'][0]['status']='running'
        (root/'tree.json').write_text(json.dumps(tree))
        callfile=root/'calls/c00000000.json'
        call=json.loads(callfile.read_text());call.update(cost=None,status='in_flight')
        callfile.write_text(json.dumps(call))
        for p in (root/'calls').glob('*.json'):
            if p!=callfile:p.unlink()
        self.args.resume=True
        result,traces,_,calls=self.run_engine()
        self.assertEqual(len(calls),4)  # Only the next task; uncertain action is retained.
        self.assertTrue(result['cost_complete'])
        self.assertAlmostEqual(result['total_api_cost'],.1)
        self.assertTrue(json.loads(callfile.read_text())['cost_recovered_from_saved_usage'])
        self.assertEqual(traces[0]['info']['exit_status'],'interrupted_call')

    def test_unknown_api_cost_continues_only_without_budget_guard(self):
        async def broken(*args,**kwargs):raise ProviderError('transport timeout')
        with patch.object(Provider,'complete',broken):
            result,traces,_,_=self.run_engine(stop_on_unknown_cost=False)
        self.assertEqual(len(traces),2)
        self.assertFalse(result['cost_complete'])
        self.assertEqual(result['stop_reason'],'tasks_finished')

    def test_eval_only_uses_patch_and_never_provider(self):
        source=self.root/'old';source.mkdir()
        record=self.root/'record.json';record.write_text('{}')
        (source/'plan.json').write_text(json.dumps({'tasks':[
            {'task_id':'broken','benchmark':'swe','record_path':str(self.root/'missing.json')},
            {'task_id':'x','benchmark':'swe','record_path':str(record)}]}))
        (source/'tree.json').write_text(json.dumps({'nodes':[
            {'id':'n000001','task_id':'broken','children':[],'status':'evaluation_error'},
            {'id':'n000000','task_id':'x','children':[],'status':'evaluation_error'}]}))
        (source/'summary.json').write_text('{"total_api_cost": 3}')
        folder=source/'workspaces/n000000/replay-1';folder.mkdir(parents=True)
        (folder/'prediction.patch').write_text('saved diff')
        (source/'traces').mkdir()
        trace={'info': {'trace_id':'n000000','task_id':'x','run_id':0,'split':'train','benchmark':'swe',
                       'cost_complete':True,'exit_status':'evaluation_error','policy':'single-model','cost_currency':'CNY'},
               'messages':[{'role':'assistant','content':'done'}],
               'steps':[{'step_idx':0,'cost':.2,'state':{},'selected_profile':'a','requested_model':'a',
                         'returned_model':'a','propensity':1}]}
        (source/'traces/n000000.traj.json').write_text(json.dumps(trace))
        self.args.eval_only=source
        with patch.object(native,'native_evaluate',return_value={'status':'passed'}) as grade, \
             patch.object(Provider,'complete',side_effect=AssertionError('must not call')):
            result=evaluate_saved(self.args)
        self.assertEqual(result['passed'],1)
        self.assertEqual(result['unscored'],1)
        self.assertEqual(result['new_api_cost'],0)
        self.assertEqual(grade.call_args.args[1].read_text(),'saved diff')
        self.assertEqual(json.loads((source/'summary.json').read_text())['total_api_cost'],3)
        refreshed=json.loads((self.args.output_dir/'traces/n000000.traj.json').read_text())
        self.assertEqual(refreshed['messages'],trace['messages'])
        self.assertEqual(refreshed['steps'],trace['steps'])
        sample=json.loads((self.args.output_dir/'steps.jsonl').read_text())
        self.assertEqual(sample['terminal_score'],100)
        self.assertEqual(sample['step_cost'],.2)

    def test_precheck_checks_empty_and_gold_without_models(self):
        from mixed_pilot_harness.inference.maintenance import precheck
        record=self.root/'record.json'
        record.write_text(json.dumps({'patch':'gold','fail_to_pass':['f'],'pass_to_pass':[]}))
        tasks=[{'task_id':'x','benchmark':'swe','record_path':str(record)}]
        def grade(row, patch_file, *args):
            return {'status':'passed' if patch_file.read_text()=='gold' else 'failed'}
        with patch('mixed_pilot_harness.inference.maintenance.load_dataset',return_value=tasks), \
             patch.object(native,'prepare_native'),patch.object(native,'pull_image'), \
             patch.object(native,'native_evaluate',side_effect=grade) as evaluator:
            result=precheck(self.args)
        self.assertEqual(result['ready'],1)
        self.assertEqual(evaluator.call_count,2)
        self.assertEqual(result['new_api_cost'],0)


if __name__=='__main__':unittest.main()
