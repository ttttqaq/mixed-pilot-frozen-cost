"""Offline smoke tests: no credentials, paid calls, JVM, Docker or downloads."""
import asyncio
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock, AsyncMock, call

from mixed_pilot_harness.inference.cli import parser, plan, execute
from mixed_pilot_harness.inference.dataset import load_dataset
from mixed_pilot_harness.inference.episodes import SWE
from mixed_pilot_harness.branch_sampling.core import Settings
from mixed_pilot_harness.branch_sampling.rules import Rules
from mixed_pilot_harness.collection.episodes import Transition
from mixed_pilot_harness.config import HarnessConfig, ProviderProfile
from mixed_pilot_harness.providers import ScriptedProvider
from mixed_pilot_harness.types import Message, ProviderResponse, Usage


class Episode:
    system = 'test'
    tools = []
    score = 0
    def start(self):
        self.steps = 0
        return [Message('user', 'Solve')]
    async def advance(self, response):
        self.steps += 1
        self.score = 100 if response.text == 'a' else 0
        return Transition([Message('user', 'not found')], {'observation':'not found','command':response.text},
                          done=self.steps==4, status='EnvironmentDone')
    async def evaluate(self, **kwargs):
        return dict(status='scored',score=self.score,success=self.score==100,cost=0,cost_complete=True,api_calls=0)
    def close(self): pass


class Provider:
    def __init__(self, profile, calls, empty=False):
        self.profile, self.calls, self.empty = profile, calls, empty
    async def complete(self, messages, tools, system):
        self.calls.append(self.profile.model)
        return ProviderResponse(text='' if self.empty else self.profile.model,
                                model=self.profile.model, usage=Usage(10,10))


class Tests(unittest.TestCase):
    def test_both_clis_accept_incomplete_cost_without_hiding_real_errors(self):
        from mixed_pilot_harness.inference.cli import main as infer_main
        from mixed_pilot_harness.step_collection.cli import main as collect_main
        tasks=[dict(task_id='one',benchmark='hle',split='train')]
        for main,mode_args in [(infer_main,['--mode','single-model']),
                               (collect_main,['--mode','run'])]:
            for budget in (False,True):
                settings=Settings(mode='single-model',model_pool=['a'],until_budget=budget)
                good=dict(cost_complete=False,failed_nodes=0,unscored_traces=0,
                          stop_reason='budget_exhausted' if budget else 'tasks_finished')
                for update,expected in [({},0),({'failed_nodes':1},1),
                                        ({'unscored_traces':1},1),
                                        ({'stop_reason':'unknown_api_cost'},1)]:
                    with self.subTest(cli=main.__module__,budget=budget,update=update):
                        with patch('mixed_pilot_harness.inference.cli.plan',return_value=(tasks,self.config,self.prices,settings)), \
                             patch('mixed_pilot_harness.inference.cli.execute',new_callable=AsyncMock,return_value={**good,**update}), \
                             redirect_stdout(io.StringIO()):
                            code=main(['--dataset',str(self.root),'--models','a',
                                       '--output-dir',str(self.root/'unused'),*mode_args])
                        self.assertEqual(code,expected)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = parser().parse_args(['--dataset',str(self.root),'--mode','single-model',
                          '--models','a','--output-dir',str(self.root/'out')])
        self.config = HarnessConfig(self.root/'config.json','a',
            {p:ProviderProfile('scripted',p) for p in ['a','b','c']})
        self.prices = {'currency':'CNY','as_of':'test','profiles':{
            p:{'input_per_million':1000,'output_per_million':1000} for p in ['a','b','c']}}
        self.tasks = [dict(task_id='one',benchmark='scienceworld',split='train'),
                      dict(task_id='two',benchmark='scienceworld',split='test')]

    def run_engine(self, *, mode='single-model', pool=None, tasks=None, empty=False, **settings):
        pool = pool or ['a']
        s = Settings(mode=mode,model_pool=pool, max_steps=8,total_cost_limit=100,
                     path_cost_limit=100,task_cost_limit=100,max_nodes=12,max_depth=1,
                     rules=Rules(2,3,2))
        s = replace(s,**settings)
        calls = []
        with redirect_stdout(io.StringIO()):
            result = asyncio.run(execute(self.args,tasks or self.tasks,self.config,self.prices,s,
                episode_factory=lambda *a:Episode(),provider_factory=lambda p,**kw:Provider(p,calls,empty)))
        traces = [json.loads(p.read_text()) for p in (self.args.output_dir/'traces').glob('*.traj.json')]
        steps = [json.loads(x) for x in (self.args.output_dir/'steps.jsonl').read_text().splitlines()]
        return result,traces,steps,calls

    def test_resume_ignores_unrelated_prices_and_snapshot_metadata(self):
        result, _, _, _ = self.run_engine()
        snapshot = (self.args.output_dir/'plan.json').read_bytes()
        self.args.resume = True
        self.prices['as_of'] = 'new-date'
        self.prices['sources'] = ['new-source']
        self.prices['profiles']['b']['output_per_million'] = 9999
        self.prices['profiles']['new-model'] = {'input_per_million': 1, 'output_per_million': 2}
        self.prices['profiles']['a']['as_of'] = 'new-date'
        resumed, _, _, calls = self.run_engine()
        self.assertEqual(calls, [])
        self.assertEqual(resumed['total_api_cost'], result['total_api_cost'])
        self.assertEqual((self.args.output_dir/'plan.json').read_bytes(), snapshot)

    def test_resume_price_change_preserves_old_calls(self):
        self.run_engine()
        old = {p.name: json.loads(p.read_text()) for p in (self.args.output_dir/'calls').glob('*.json')}
        self.args.resume = True
        self.prices['profiles']['a']['output_per_million'] += 1
        result, _, _, calls = self.run_engine()
        self.assertEqual(calls, [])
        for name, data in old.items():
            self.assertEqual(json.loads((self.args.output_dir/'calls'/name).read_text()), data)

    def test_resume_partial_switches_provider_currency_and_preserves_history(self):
        self.run_engine(total_cost_limit=1e100, task_cost_limit=1e100, path_cost_limit=1e100)
        out = self.args.output_dir
        tree = json.loads((out/'tree.json').read_text())
        tree['jobs'][1]['status'] = 'partial'
        (out/'tree.json').write_text(json.dumps(tree))
        old_calls = {p.name: json.loads(p.read_text()) for p in (out/'calls').glob('*.json')}
        old_total = sum(c['cost'] for c in old_calls.values())
        self.args.resume_partial = True
        self.config.profiles['a'].model = 'new-provider-model'
        self.prices = {'currency': 'USD', 'as_of': 'new', 'profiles': {
            'a': {'input_per_million': 2000, 'output_per_million': 2000}}}
        result, traces, _, calls = self.run_engine(total_cost_limit=1e100,
            task_cost_limit=1e100, path_cost_limit=1e100)
        self.assertEqual(len(calls), 4)
        self.assertAlmostEqual(result['cost_by_currency']['CNY'], old_total)
        self.assertAlmostEqual(result['cost_by_currency']['USD'], .16)
        self.assertAlmostEqual(result['total_api_cost'], .16)
        original = next(t for t in traces if t['info']['trace_id']=='n000000')
        self.assertEqual(original['info']['cost_currency'], 'CNY')
        self.assertEqual(original['info']['config']['profiles']['a']['model'], 'a')
        newest = max(traces, key=lambda t:t['info']['trace_id'])
        self.assertEqual(newest['info']['cost_currency'], 'USD')
        self.assertEqual(newest['info']['config']['profiles']['a']['model'], 'new-provider-model')
        for name, data in old_calls.items():
            self.assertEqual(json.loads((out/'calls'/name).read_text()), data)
        # A second resume must retain both currencies and skip completed tasks.
        again, _, _, calls = self.run_engine(total_cost_limit=1e100,
            task_cost_limit=1e100, path_cost_limit=1e100)
        self.assertEqual(calls, [])
        self.assertEqual(again['cost_by_currency'], result['cost_by_currency'])

    def test_legacy_unpriced_call_recovers_with_original_rate(self):
        self.run_engine()
        out = self.args.output_dir
        path = sorted((out/'calls').glob('*.json'))[0]
        data = json.loads(path.read_text())
        expected = data['cost']
        data['cost'] = None
        for key in ('cost_currency', 'pricing_snapshot', 'provider_snapshot'):
            data.pop(key, None)
        path.write_text(json.dumps(data))
        self.prices['profiles']['a']['input_per_million'] *= 10
        self.args.resume = True
        self.run_engine()
        recovered = json.loads(path.read_text())
        self.assertEqual(recovered['cost'], expected)
        self.assertEqual(recovered['cost_currency'], 'CNY')
        self.assertEqual(recovered['pricing_snapshot']['input_per_million'], 1000)

    def test_billing_terms_preserve_all_cost_controls(self):
        from copy import deepcopy
        from mixed_pilot_harness.collection.pricing import billing_terms
        original = {'currency': 'CNY', 'profiles': {'a': {'bands': [
            {'max_input_tokens': 100, 'input_per_million': 2,
             'cached_input_per_million': 1, 'output_per_million': 3}]}}}
        expected = billing_terms(original, 'a')
        for key in ['max_input_tokens', 'input_per_million', 'cached_input_per_million', 'output_per_million']:
            with self.subTest(key=key):
                changed = deepcopy(original)
                changed['profiles']['a']['bands'][0][key] += 1
                self.assertNotEqual(expected, billing_terms(changed, 'a'))
        changed = deepcopy(original)
        changed['currency'] = 'USD'
        self.assertNotEqual(expected, billing_terms(changed, 'a'))
        changed = deepcopy(original)
        changed['profiles']['a']['schedule'] = 'deepseek_cn_peak'
        self.assertNotEqual(expected, billing_terms(changed, 'a'))
        with self.assertRaisesRegex(ValueError, 'missing pricing'):
            billing_terms(original, 'judge')

    def test_single_two_rounds_outputs_and_cost(self):
        result,traces,steps,calls = self.run_engine(runs=2)
        self.assertEqual(result['pass_rate'],1)
        self.assertEqual(len(traces),4)
        self.assertEqual(len(steps),16)
        self.assertAlmostEqual(result['total_api_cost'],len(calls)*.02)
        self.assertEqual([(j['task_id'],j['run_id']) for j in result['jobs']],
                         [('one',0),('two',0),('one',1),('two',1)])
        self.assertTrue(all(t['messages'] and t['steps'] for t in traces))
        self.assertIn('Pass rate: 100.00%',(self.args.output_dir/'SUMMARY.md').read_text())

    def test_random_three_models_uniform_probability(self):
        result,traces,steps,calls = self.run_engine(mode='random',pool=['a','b','c'],runs=4)
        self.assertEqual(set(calls),{'a','b','c'})
        self.assertTrue(all(s['propensity']==1/3 for s in steps))

    def test_ksr_three_models_shared_prefix_billed_once(self):
        result,traces,steps,calls = self.run_engine(mode='ksr',pool=['a','b','c'],tasks=self.tasks[:1])
        self.assertEqual(len(traces),9)
        self.assertEqual(len(calls),24)
        self.assertEqual(len(steps),36)
        self.assertAlmostEqual(result['total_api_cost'],.48)
        samples=[json.loads(x) for x in (self.args.output_dir/'samples.jsonl').read_text().splitlines()]
        self.assertTrue(all(s['paired_actions_complete'] for s in samples))
        self.assertTrue(all(len(t['steps'])==4 for t in traces))

    def test_budget_and_no_cost_progress(self):
        result,traces,steps,calls = self.run_engine(until_budget=True,total_cost_limit=.05)
        self.assertEqual(result['stop_reason'],'budget_exhausted')
        self.assertEqual(len(calls),3)
        self.assertAlmostEqual(result['total_api_cost'],.06)
        self.args.output_dir=self.root/'free'
        for rate in self.prices['profiles'].values():
            rate.update(input_per_million=0,output_per_million=0)
        result,*_ = self.run_engine(until_budget=True)
        self.assertEqual(result['stop_reason'],'no_cost_progress')

    def test_empty_is_failure_and_continues_dataset(self):
        result,traces,steps,calls = self.run_engine(empty=True)
        self.assertEqual(result['pass_rate'],0)
        self.assertEqual(result['scored_traces'],2)
        self.assertEqual(len(calls),6)
        self.assertTrue(all(t['info']['exit_status']=='EmptyResponseLimit' for t in traces))
        self.assertEqual(len(steps),6)

    def test_hle_real_episode_local_scripted_provider(self):
        root=self.root/'dataset'
        for split in ['train','test']:
            task=root/split/'tasks'/'q'
            (task/'workspace').mkdir(parents=True)
            (task/'reference').mkdir()
            (task/'prompt.md').write_text('Six times seven?')
            (task/'reference/gold.json').write_text('{"answer":"42"}')
            m={'id':'q','source':{'dataset':'HLE','split':split,'upstream_id':split+'q'},
               'routing_metadata':{'router_split':split},'agent':{'tools':[]}}
            (task/'manifest.json').write_text(json.dumps(m))
            (root/f'{split}.json').write_text(json.dumps({'split':split,'task_count':1,
                                                        'tasks':[{'id':'q','path':f'{split}/tasks/q'}]}))
        responses=self.root/'responses.json'
        responses.write_text(json.dumps([{'text':'Explanation: Arithmetic\nAnswer: 42\nConfidence: 100%',
                                         'usage':{'input_tokens':10,'output_tokens':10}}]*2))
        config=self.root/'models.json'
        config.write_text(json.dumps({'profiles':{'a':{'kind':'scripted','model':'a','responses_file':str(responses)}}}))
        prices=self.root/'prices.json';prices.write_text(json.dumps(self.prices))
        a=parser().parse_args(['--dataset',str(root),'--split','all','--models','a','--mode','single-model',
            '--config',str(config),'--pricing',str(prices),'--output-dir',str(self.root/'hle'),'--hle-evaluator','exact'])
        tasks,cfg,rates,settings=plan(a)
        self.assertEqual([t['split'] for t in tasks],['train','test'])
        with redirect_stdout(io.StringIO()):
            result=asyncio.run(execute(a,tasks,cfg,rates,settings))
        self.assertEqual(result['pass_rate'],1)
        self.assertEqual(result['scored_traces'],2)
        self.assertAlmostEqual(result['total_api_cost'],.04)
        # Incorrect index provenance must fail before any provider is created.
        p=root/'test/tasks/q/manifest.json';m=json.loads(p.read_text());m['routing_metadata']['router_split']='train';p.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'split mismatch'):load_dataset(root,'all')

    def test_swe_native_adapter_offline(self):
        from mixed_pilot_harness.inference import episodes
        prompt=self.root/'prompt.md';prompt.write_text('Fix code')
        row=self.root/'row.json';row.write_text(json.dumps({'image_name':'fixture','repo':'x/y'}))
        task={'task_id':'swe','record_path':str(row),'prompt_path':str(prompt),'allowed_tools':[]}
        source=self.root/'source';source.mkdir()
        (source/'code.txt').write_text('original')
        session=Mock();session.exec.return_value.returncode=0
        with patch.object(episodes.native,'prepare_native'), patch.object(episodes.native,'pull_image'), \
             patch.object(episodes.native,'repo_snapshot',return_value=(source,'base')), \
             patch.object(episodes.native,'DockerSession',return_value=session), \
             patch.object(episodes.native,'export_patch') as export_patch, \
             patch.object(episodes.native,'native_evaluate',return_value={'status':'passed'}) as grade:
            ep=SWE(task,self.root/'episode.json',self.args)
            ep.start()
            asyncio.run(ep.advance(ProviderResponse(text='Done')))
            result=asyncio.run(ep.evaluate())
            ep.close()
        self.assertTrue(result['success'])
        self.assertEqual(result['method'],'swe_native')
        export_patch.assert_called_once();grade.assert_called_once();session.close.assert_called_once()

    def test_cardinality_and_budget_validation(self):
        self.args.models=['a','b']
        with self.assertRaisesRegex(ValueError,'one profile'):plan(self.args)
        self.args.models=['a'];self.args.stop_mode='budget-mode'
        with self.assertRaisesRegex(ValueError,'--budget'):plan(self.args)

    def test_retry_validation_and_count(self):
        self.args.max_retries=-1
        with self.assertRaisesRegex(ValueError,'max-retries'):plan(self.args)
        self.args.max_retries=0
        self.args.retry_wait=float('nan')
        with self.assertRaisesRegex(ValueError,'retry-wait'):plan(self.args)
        self.args.retry_wait=0
        self.args.models=['a','b']
        with self.assertRaisesRegex(ValueError,'one profile'):plan(self.args)
        self.assertEqual(self.args.replay_attempts,1)

    def test_completed_traces_pause_but_resume_skips_do_not(self):
        with patch('mixed_pilot_harness.branch_sampling.core.asyncio.sleep',new_callable=AsyncMock) as sleep:
            self.run_engine(trace_delay=3)
            self.assertEqual(sleep.await_args_list,[call(3),call(3)])
            sleep.reset_mock()
            self.args.resume=True
            self.run_engine(trace_delay=3)
            sleep.assert_not_awaited()

    def test_restore_retry_wait_and_exhaustion(self):
        from mixed_pilot_harness.inference.report import InferenceCollector
        settings=Settings(mode='single-model',model_pool=['a'],replay_attempts=3,retry_wait=7)
        episode=Mock();episode.start.side_effect=RuntimeError('temporary setup failure')
        collector=InferenceCollector(self.tasks,self.config,self.prices,settings,self.root/'restore',
                                     episode_factory=lambda *args:episode)
        node={'replay':[]}
        with patch('mixed_pilot_harness.branch_sampling.core.asyncio.sleep',new_callable=AsyncMock) as sleep:
            self.assertEqual(asyncio.run(collector.restore(self.tasks[0],[],None,node)),(None,None))
        self.assertEqual(episode.start.call_count,3)
        self.assertEqual(episode.close.call_count,3)
        self.assertEqual(sleep.await_args_list,[call(7),call(7)])

    def test_http_retry_policy_bounds_and_nontransient_errors(self):
        import urllib.error
        from mixed_pilot_harness.providers import HTTP_RETRY_POLICY, JsonHttpProvider, ProviderError
        provider=JsonHttpProvider(self.config.profile('a'))
        for retries, status, expected in [(0,503,1),(2,503,3),(2,401,1)]:
            with self.subTest(retries=retries,status=status):
                error=ProviderError('upstream failure',cost_known_zero=True)
                error.__cause__=urllib.error.HTTPError('https://example.invalid',status,'error',{},None)
                token=HTTP_RETRY_POLICY.set((retries,7))
                try:
                    with patch.object(provider,'_post_json_sync',side_effect=error) as post, \
                         patch('mixed_pilot_harness.providers.asyncio.sleep',new_callable=AsyncMock) as sleep:
                        with self.assertRaises(ProviderError):
                            asyncio.run(provider._post_json('unused',{},{}))
                    self.assertEqual(post.call_count,expected)
                    self.assertEqual(sleep.await_args_list,[call(7)]*(expected-1))
                finally:
                    HTTP_RETRY_POLICY.reset(token)

    def test_swe_complete_loop_exports_real_patch_and_counts_tokens(self):
        from mixed_pilot_harness.inference import episodes
        native = episodes.native
        source = self.root/'repo';source.mkdir()
        (source/'answer.txt').write_text('old\n')
        native.command(['git','init','-q',str(source)])
        native.command(['git','-C',str(source),'add','-A'])
        native.command(['git','-C',str(source),'-c','user.name=Test','-c','user.email=test@example.org','commit','-qm','base'])
        base = native.command(['git','-C',str(source),'rev-parse','HEAD']).stdout.strip()
        prompt=self.root/'prompt.md';prompt.write_text('Fix answer.txt')
        record=self.root/'record.json';record.write_text(json.dumps({'image_name':'fixture','repo':'x/y'}))
        task=dict(task_id='swe',benchmark='swe',split='train',prompt_path=str(prompt),record_path=str(record),allowed_tools=['write_file'])
        provider=ScriptedProvider([
            {'tool_calls':[{'id':'write','name':'write_file','arguments':{'path':'answer.txt','content':'fixed\n'}}],
             'usage':{'input_tokens':10,'output_tokens':10}},
            {'text':'Done','usage':{'input_tokens':10,'output_tokens':10}}])
        session=Mock();session.exec.return_value.returncode=0
        def grade(row, patch_file, *args):
            self.assertIn('+fixed',patch_file.read_text())
            return {'status':'passed'}
        settings=Settings(mode='single-model',model_pool=['a'])
        with patch.object(native,'prepare_native'), patch.object(native,'pull_image'), \
             patch.object(native,'repo_snapshot',return_value=(source,base)), \
             patch.object(native,'DockerSession',return_value=session), \
             patch.object(native,'native_evaluate',side_effect=grade), redirect_stdout(io.StringIO()):
            report=asyncio.run(execute(self.args,[task],self.config,self.prices,settings,
                                      provider_factory=lambda *a,**kw:provider))
        trace=json.loads(next((self.args.output_dir/'traces').glob('*.traj.json')).read_text())
        self.assertEqual(report['pass_rate'],1)
        self.assertAlmostEqual(report['total_api_cost'],.04)
        self.assertEqual(trace['info']['evaluation']['method'],'swe_native')
        self.assertEqual(len(trace['steps']),2)
        self.assertIn('tool_results',trace['steps'][0])
        self.assertEqual((source/'answer.txt').read_text(),'old\n')
        session.close.assert_called_once()


if __name__=='__main__':unittest.main()
