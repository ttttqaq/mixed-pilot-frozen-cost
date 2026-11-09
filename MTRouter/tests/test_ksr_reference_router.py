"""Method tests: trajectory counters, KSR labels, separate heads, reuse and budgets."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess

import torch
from mixed_pilot_harness.config import HarnessConfig, ProviderProfile
from mixed_pilot_mtrouter.online.smoke import smoke_factories, SmokeEpisode, SmokeProvider
from mixed_pilot_mtrouter.online.learning import state_key
from mixed_pilot_mtrouter.online.runtime import RecoverableRequestError, BudgetExceeded
from mixed_pilot_mtrouter.online.ksr.config import DEFAULTS
from mixed_pilot_mtrouter.online.ksr.cold import labels
from mixed_pilot_mtrouter.online.ksr.learning import SeparateLearner
from mixed_pilot_mtrouter.online.ksr.runtime import request_bound, GuardedRunner
from mixed_pilot_mtrouter.online.ksr.trainer import KSRTrainer
from mixed_pilot_mtrouter.online.ksr.recovery import method_fingerprint, prepare_runtime_repair


class KSRReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = deepcopy(DEFAULTS)
        self.cfg.update(project_root=str(self.root), output_dir=str(self.root/'run'))
        self.cfg['encoder'].update(backend='hash', device='cpu', hash_dim=8)
        self.cfg['model'].update(hidden_dims=[8], ensemble_size=2, dropout=0., model_metadata=None)
        self.cfg['optimization'].update(epochs=1, batch_size=8)
        self.cfg['resilience']['enabled'] = False
        self.cfg['cold_start'].update(trajectory_steps=20, checkpoint_steps=5, ksr_config=str(self.root/'ksr.json'))
        self.cfg['evaluation']['enabled'] = False
        self.cfg['online'].update(iterations=1, tasks_per_iteration=2, states_per_iteration=2)
        self.cfg['short_rollout'].update(horizon=1, max_horizon=1)
        pool = self.cfg['model_pool']
        self.rules = {'strong_model':pool[1], 'weak_model':pool[0], 'default_start':'weak',
            'strong_first_tasks':['*'], 'weak_first_tasks':['*'],
            'KSR_weak2strong':{'signals':['stalled_progress'], 'min_branch_gap':2, 'stall_window':2},
            'KSR_strong2weak':{'signals':['easy_progress'], 'min_branch_gap':2, 'progress_window':2}}
        (self.root/'ksr.json').write_text(json.dumps(self.rules))
        self.train = [dict(task_id='train--fixture-'+str(i), benchmark='scienceworld', split='train') for i in range(6)]
        self.valid = [dict(task_id='test--fixture-9', benchmark='scienceworld', split='test')]
        names = pool + [self.cfg['teacher']['llm']['profile']]
        self.harness = HarnessConfig(self.root/'harness.json', pool[0],
            {n:ProviderProfile(kind='scripted', model=n, name=n, max_tokens=1024) for n in names})
        self.pricing = {'currency':'CNY', 'profiles':{n:{'input_per_million':1., 'output_per_million':1.} for n in names}}
        self.manifest = {'split_fingerprint':'test-split', 'runtime_fingerprint':'test-source', 'reference_fingerprint':'test-refs'}

    def tearDown(self): self.tmp.cleanup()

    def trainer(self, cfg=None, cls=KSRTrainer, **kwargs):
        cfg = cfg or self.cfg
        episode, provider = smoke_factories(cfg)
        return cls(cfg, self.train, self.valid, self.manifest, self.harness, self.pricing,
            references={t['task_id']:{'sha256':'x','actions':[]} for t in self.train},
            episode_factory=episode, provider_factory=provider, simulation=True, **kwargs)

    def test_500_setting_counts_actor_interactions_not_optimizer_updates(self):
        t = self.trainer()
        asyncio.run(t._cold_start())
        data = json.loads((t.out/'cold_start/cache.json').read_text())
        self.assertEqual(len(data['calls']), 20)
        self.assertEqual(len([r for r in t.ledger.records.values() if r['scope']=='cold_start']), 20)
        self.assertEqual(t.ksr_trained_steps,20)
        self.assertNotEqual(t.learner.router_updates,20)
        self.assertEqual(sorted(p.name for p in (t.out/'cold_start/checkpoints').glob('*.pt')),
            ['step_000005.pt','step_000010.pt','step_000015.pt','step_000020.pt'])
        samples = [json.loads(l) for l in (t.out/'cold_start/samples.jsonl').read_text().splitlines()]
        self.assertEqual(len(samples),20)
        self.assertEqual(len({r['call_id'] for r in samples}),20)
        self.assertTrue(all(r['reward'] in (0,1) for r in samples if r['labels_valid']))

    def test_cold_cache_reuse_makes_no_actor_requests(self):
        a = self.trainer(); asyncio.run(a._cold_start())
        cfg = deepcopy(self.cfg); cfg['output_dir'] = str(self.root/'reuse')
        cfg['cold_start']['reuse_from'] = str(a.out)
        b = self.trainer(cfg); asyncio.run(b._cold_start())
        self.assertEqual(len(b.ledger.records),0)
        self.assertEqual(b.ksr_trained_steps,20)
        self.assertGreater(b.learner.router_updates,0)

    def test_exact_500_actor_steps_and_five_collection_checkpoints(self):
        self.cfg['cold_start'].update(trajectory_steps=500, checkpoint_steps=100)
        self.cfg['optimization']['batch_size'] = 64
        self.train = [dict(task_id='train--fixture-'+str(i), benchmark='scienceworld', split='train') for i in range(70)]
        t=self.trainer(); asyncio.run(t._cold_start())
        self.assertEqual(len(t.ledger.records),500)
        self.assertEqual(sorted(p.name for p in (t.out/'cold_start/checkpoints').glob('*.pt')),
                         ['step_%06d.pt'%i for i in (100,200,300,400,500)])

    def test_branch_first_call_intervention_then_base_and_identical_roots(self):
        class BranchEpisode(SmokeEpisode):
            async def advance(self,response):
                tr=await super().advance(response)
                self.score=self.progress*25
                if response.text=='SMOKE_WEAK_OBSERVE':
                    tr.record['observation']='Nothing happens'
                    tr.messages[0].content='Nothing happens'
                return tr
        self.rules['KSR_weak2strong']={'signals':['consecutive_errors'],'min_branch_gap':3,
                                    'error_window':2,'error_markers':['Nothing happens']}
        (self.root/'ksr.json').write_text(json.dumps(self.rules))
        t=self.trainer(); t.runner.episode_factory=BranchEpisode
        asyncio.run(t._cold_start())
        d=json.loads((t.out/'cold_start/cache.json').read_text())
        parents=[n for n in d['nodes'].values() if n['children']]
        self.assertTrue(parents)
        for n in d['nodes'].values():
            expected=[n['model']]+[n['base']]*(len(n['own'])-1)
            self.assertEqual([d['calls'][c]['profile'] for c in n['own']],expected if n['own'] else [])
        for n in parents:
            children=[d['nodes'][c] for c in n['children']]
            self.assertEqual(children[0]['prefix'],children[1]['prefix'])
            if all(c['own'] for c in children):
                self.assertEqual(state_key(d['calls'][children[0]['own'][0]]['before_state']),
                                 state_key(d['calls'][children[1]['own'][0]]['before_state']))

    def test_real_dispatch_task_cap_is_shared_by_both_roots(self):
        class MaxOutputProvider(SmokeProvider):
            async def complete(self,*args,**kwargs):
                r=await super().complete(*args,**kwargs)
                r.usage.input_tokens=100
                r.usage.output_tokens=self.profile.max_tokens
                return r
        self.cfg['cold_start']['trajectory_steps']=5
        for name in self.cfg['model_pool']:
            self.harness.profiles[name]=replace(self.harness.profiles[name],max_tokens=8192)
            self.pricing['profiles'][name]={'input_per_million':8.,'output_per_million':28.}
        t=self.trainer()
        t.runner.provider_factory=lambda p,**kw: MaxOutputProvider(p,strong_model=self.cfg['strong_model'])
        asyncio.run(t._cold_start())
        d=json.loads((t.out/'cold_start/cache.json').read_text())
        tasks={n['task_id'] for n in d['nodes'].values()}
        self.assertEqual(len(tasks),1)
        self.assertLessEqual(t.ledger.cost,1.)
        self.assertEqual({r['profile'] for r in t.ledger.records.values()},set(self.cfg['model_pool']))

    def test_resume_retains_unique_calls_and_progress(self):
        class Pause(KSRTrainer):
            async def cold_checkpoint(self, collector, count):
                await super().cold_checkpoint(collector, count)
                if count == 10: raise RecoverableRequestError('fixture pause')
        a = self.trainer(cls=Pause)
        a._save(0)
        with self.assertRaises(RecoverableRequestError): asyncio.run(a._cold_start())
        self.assertEqual(len(a.ledger.records),10)
        b = self.trainer(resume=True); asyncio.run(b._cold_start())
        self.assertEqual(len(b.ledger.records),20)
        self.assertEqual(len(json.loads((b.out/'cold_start/cache.json').read_text())['calls']),20)

    def test_full_cold_evaluation_then_five_online_rounds(self):
        self.cfg['evaluation'].update(enabled=True,every_iterations=3)
        self.cfg['online']['iterations']=5
        t=self.trainer(); report=asyncio.run(t.run())
        self.assertEqual(report['stop_reason'],'iterations_completed')
        self.assertEqual(report['completed_iterations'],5)
        self.assertEqual(report['cold_trajectory_steps'],20)
        evaluations=[json.loads(l) for l in (t.out/'evaluation.jsonl').read_text().splitlines()]
        self.assertEqual([e['iteration'] for e in evaluations],[0,3,5])
        self.assertTrue(all(e['training_updates'] is False for e in evaluations))
        self.assertTrue(all(r['state']['split']=='train' for r in t.router_rows.values()))
        self.assertTrue(all('delta' not in p for p in t.pairs.values()))

    def test_resume_before_first_checkpoint_preserves_schedule_rng(self):
        class PauseRunner(GuardedRunner):
            async def rollout(self,*a,**kw):
                if len(self.ledger.records)==3: raise RecoverableRequestError('fixture pre-checkpoint pause')
                return await super().rollout(*a,**kw)
        class Pause(KSRTrainer): runner_class=PauseRunner
        a=self.trainer(cls=Pause); a._save(0)
        with self.assertRaises(RecoverableRequestError): asyncio.run(a._cold_start())
        self.assertEqual(len(a.ledger.records),3)
        expected=a.rng.getstate()
        b=self.trainer(resume=True); asyncio.run(b._cold_start())
        self.assertEqual(b.rng.getstate(),expected)
        self.assertEqual(len(b.ledger.records),20)

    def test_unfinished_last_path_is_preserved_without_false_failure_label(self):
        self.cfg['cold_start']['trajectory_steps'] = 19
        t = self.trainer(); asyncio.run(t._cold_start())
        rows = [json.loads(l) for l in (t.out/'cold_start/samples.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),19)
        pending = [r for r in rows if not r['labels_valid']]
        self.assertTrue(pending)
        self.assertTrue(all(r['reward'] is None and r['cost'] is None for r in pending))

    def test_beta_cannot_change_supervised_gradients(self):
        state = {'history':[{'role':'user','content':'fixture'}], 'remaining_budget':1., 'remaining_steps':50, 'step_idx':0,
                 'task_id':'train--fixture', 'split':'train'}
        rows = [dict(state=state,model=m,reward=float(i),cost=.2+i*.1,weight=1.,source='real') for i,m in enumerate(self.cfg['model_pool'])]
        a = SeparateLearner(self.cfg)
        other = deepcopy(self.cfg); other['budget']['lambda_'] = 100.; other['decision']['beta'] = 1000.
        b = SeparateLearner(other)
        b.router.load_state_dict(a.router.state_dict())
        before = deepcopy(a.router.state_dict())
        torch.manual_seed(55); a.train_router(rows,[{'delta':9999}])
        torch.manual_seed(55); b.train_router(rows,[{'delta':-9999}])
        self.assertTrue(all(torch.equal(v,b.router.state_dict()[k]) for k,v in a.router.state_dict().items()))
        changed = [k for k,v in a.router.state_dict().items() if not torch.equal(v,before[k])]
        self.assertTrue(any('.reward.' in k for k in changed))
        self.assertTrue(any('.cost.' in k for k in changed))

    def test_native_success_tree_average_and_expected_cost_not_collection_sum(self):
        data = {'fingerprint':'x','version':'v','nodes':{},'calls':{}}
        for i,(parent,children,reward,cost) in enumerate([(None,[1,2],None,.01),(0,[],1.,.02),
                (0,[3,4],None,.03),(2,[],0.,.04),(2,[],1.,.05)]):
            data['nodes'][str(i)] = dict(task_id='train--x',base='q',prefix=[],own=[str(i)],
                children=list(map(str,children)),status='split' if children else 'done',reward=reward,labels_valid=reward is not None)
            data['calls'][str(i)] = dict(cost=cost,before_state={'split':'train'},profile='q')
        rows,_ = labels(data)
        root = next(r for r in rows if r['call_id']=='0')
        self.assertAlmostEqual(root['reward'],.75)
        self.assertAlmostEqual(root['cost'],.0575)
        data['nodes']['4']['labels_valid'] = False
        rows,_ = labels(data)
        self.assertFalse(next(r for r in rows if r['call_id']=='0')['labels_valid'])
        self.assertTrue(next(r for r in rows if r['call_id']=='1')['labels_valid'])

    def test_request_reservation_reduces_output_to_available_budget(self):
        rates={'input_per_million':8.,'output_per_million':28.}
        bound = request_bound([], '', [], rates, .02, 8192)
        self.assertLess(bound['max_tokens'],8192)
        self.assertLessEqual(bound['cost_bound'],.02)
        self.assertIsNone(request_bound([], '', [], rates,.000001,8192))

    def test_task_cap_counts_both_root_directions_once(self):
        t=self.trainer(); v='cold-ksr'
        for i,m in enumerate(self.cfg['model_pool']):
            cid=t.ledger.begin('cold_start',m,{'task_id':'train--x','policy_version':v})
            t.ledger.finish(cid,cost=.4,complete=True)
        self.assertAlmostEqual(t.runner.task_spent('train--x',v),.8)
        self.assertEqual(t.runner.task_spent('train--x','another-round'),0.)

    def test_task_cap_includes_known_fees_and_bounded_unknown_reservations(self):
        t=self.trainer();task=self.train[0]['task_id'];version='v'
        for cost,reserve in ((None,.4),(.2,.3)):
            cid=t.ledger.begin('branch',self.cfg['model_pool'][0],{'task_id':task,'policy_version':version})
            t.ledger.finish(cid,cost=cost,complete=False,status='provider_error')
            t.ledger.reserve(cid,reserve,{'method':'fixture_conservative_upper_bound'})
        self.assertAlmostEqual(t.runner.task_spent(task,version),.9)
        self.assertAlmostEqual(t.runner.task_spent(task,version),t.ledger.budget_cost)

    def test_actor_500_retry_completes_evaluation_without_claiming_known_cost_labels(self):
        from urllib.error import HTTPError
        from mixed_pilot_harness.types import ProviderError
        class Flaky(SmokeProvider):
            async def complete(self,*args,**kwargs):
                if self.calls==1:
                    self.calls+=1
                    raise ProviderError('fixture HTTP 500') from HTTPError('https://fixture.invalid',500,'fixture',{},None)
                return await super().complete(*args,**kwargs)
        self.cfg['resilience'].update(enabled=True,retry_delay=0.,qwen_min_interval=0.)
        self.cfg['evaluation']['enabled']=True
        t=self.trainer()
        t.runner.provider_factory=lambda p,**kw:Flaky(p,strong_model=self.cfg['strong_model'])
        t._policy=lambda *a,**kw:(lambda state:self.cfg['strong_model'])
        report=asyncio.run(t._evaluate(0))
        self.assertTrue(report['complete'])
        self.assertEqual(report['passed'],1)
        self.assertFalse(report['cost_complete'])
        self.assertIsNone(report['mean_cost'])
        self.assertGreater(t.ledger.reserved_cost,0.)
        self.assertLessEqual(t.runner.task_spent(self.valid[0]['task_id'],report['policy_version']),1.)
        result=json.loads((t.out/'trajectories'/ (report['trajectory_ids'][0]+'.json')).read_text())
        self.assertEqual(len(result['steps']),4)
        self.assertEqual(len(result['call_ids']),5)
        self.assertTrue(result['outcome_valid'])
        self.assertFalse(result['labels_valid'])
        self.assertIsNone(result['reward'])
        self.assertFalse(result['cost_complete'])

    def test_evaluation_budget_cutoff_scores_native_outcome_and_preserves_old_unknown_fee(self):
        self.cfg['evaluation']['enabled']=True
        t=self.trainer();version=t._policy_id(greedy=True)
        cid=t.ledger.begin('test',self.cfg['strong_model'],{'task_id':self.valid[0]['task_id'],'policy_version':version})
        t.ledger.finish(cid,cost=None,complete=False,status='provider_error')
        t.ledger.reserve(cid,.9,{'method':'fixture_conservative_upper_bound'})
        with patch('mixed_pilot_mtrouter.online.ksr.runtime.request_bound',return_value=None):
            report=asyncio.run(t._evaluate(0))
        self.assertTrue(report['complete'])
        self.assertEqual(report['scored_tasks'],1)
        self.assertEqual(report['passed'],0)
        self.assertFalse(report['cost_complete'])
        self.assertEqual(len(t.ledger.records),1)
        self.assertAlmostEqual(t.ledger.reserved_cost,.9)
        result=json.loads((t.out/'trajectories'/(report['trajectory_ids'][0]+'.json')).read_text())
        self.assertEqual(result['status'],'task_collection_budget')
        self.assertEqual(result['evaluation']['api_calls'],0)
        self.assertTrue(result['outcome_valid'])
        self.assertFalse(result['labels_valid'])
        self.assertFalse(result['cost_complete'])
        self.assertIsNone(result['reward'])

    def test_budget_stop_grades_live_environment_without_stochastic_replay(self):
        t = self.trainer()
        episodes = []
        def factory(task):
            self.assertFalse(episodes, 'Budget grading must not recreate the environment')
            episodes.append(SmokeEpisode(task))
            return episodes[-1]
        t.runner.episode_factory = factory
        with patch('mixed_pilot_mtrouter.online.ksr.runtime.request_bound',
                   side_effect=[{'max_tokens':1024, 'cost_bound':1.}, None]):
            result = asyncio.run(t.runner.rollout(self.train[0], lambda s:self.cfg['model_pool'][0], policy_version='v'))
        self.assertEqual(len(episodes),1)
        self.assertTrue(episodes[0].closed)
        self.assertEqual(len(t.ledger.records),1)
        self.assertEqual(result['status'],'path_request_budget')
        self.assertEqual(result['budget_grading'],'live_environment_before_close')
        self.assertTrue(result['outcome_valid'])
        self.assertFalse(result['success'])
        self.assertEqual(result['evaluation']['api_calls'],0)

    def test_shared_collection_cutoff_is_not_graded_as_failure(self):
        t=self.trainer()
        cid=t.ledger.begin('branch',self.cfg['model_pool'][0],{'task_id':self.train[0]['task_id'],'policy_version':'v'})
        t.ledger.finish(cid,cost=.999,complete=True)
        with patch('mixed_pilot_mtrouter.online.ksr.runtime.request_bound',return_value=None):
            result=asyncio.run(t.runner.rollout(self.train[0],lambda s:self.cfg['model_pool'][0],policy_version='v'))
        self.assertEqual(result['status'],'task_collection_budget')
        self.assertFalse(result['outcome_valid'])
        self.assertIsNone(result['reward'])

    def _teacher_fixture(self, provider_class):
        self.cfg['resilience'].update(enabled=True,retry_delay=0.,qwen_min_interval=0.)
        t=self.trainer()
        t.runner.provider_factory=lambda p,**kw:provider_class(p,strong_model=self.cfg['strong_model'])
        result=asyncio.run(t.runner.rollout(self.train[0],lambda s:self.cfg['model_pool'][0],horizon=1,policy_version='v'))
        return t,result['state']

    def test_empty_teacher_usage_is_reserved_and_retried_without_fabricating_label(self):
        class EmptyThenValid(SmokeProvider):
            async def complete(self,*args,**kwargs):
                response=await super().complete(*args,**kwargs)
                if 'training-only assessor' in str(args[2]).lower() and self.calls==1:
                    response.text=' ';response.stop_reason='error'
                    response.usage.input_tokens=0;response.usage.output_tokens=0;response.usage.raw={}
                return response
        t,state=self._teacher_fixture(EmptyThenValid)
        estimate=asyncio.run(t.evidence.get(state,{'fixture':True}))
        self.assertTrue(estimate['valid'])
        calls=[r for r in t.ledger.records.values() if r['scope']=='teacher']
        self.assertEqual(len(calls),2)
        self.assertIsNone(calls[0]['cost'])
        self.assertGreater(calls[0]['reserved_cost'],0.)
        self.assertFalse(calls[0]['cost_complete'])
        self.assertEqual(calls[0]['response']['text'],' ')
        self.assertTrue(t.ledger.spending_safe)
        self.assertFalse(t.ledger.cost_complete)
        self.assertAlmostEqual(t.ledger.budget_cost,t.ledger.cost+t.ledger.reserved_cost)
        retry=json.loads((t.out/'teacher_retries.jsonl').read_text().splitlines()[0])
        self.assertTrue(retry['fee_reserved'])
        self.assertEqual(retry['call_id'],calls[0]['call_id'])

    def test_invalid_teacher_json_is_kept_and_valid_retry_supersedes_error_cache(self):
        class BadThenValid(SmokeProvider):
            async def complete(self,*args,**kwargs):
                response=await super().complete(*args,**kwargs)
                if 'training-only assessor' in str(args[2]).lower() and self.calls==1:
                    response.text='not JSON'
                return response
        t,state=self._teacher_fixture(BadThenValid)
        self.assertTrue(asyncio.run(t.evidence.get(state,{'fixture':True}))['valid'])
        rows=[json.loads(x) for x in (t.out/'teacher_predictions.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),2)
        self.assertIn('error',rows[0]['estimate'])
        self.assertTrue(rows[1]['estimate']['valid'])
        self.assertEqual(rows[0]['key'],rows[1]['key'])
        before=len(t.ledger.records)
        self.assertTrue(asyncio.run(t.evidence.get(state,{'fixture':True}))['valid'])
        self.assertEqual(before,len(t.ledger.records))

    def test_teacher_exhaustion_returns_missing_evidence_for_real_rollout_fallback(self):
        class AlwaysBad(SmokeProvider):
            async def complete(self,*args,**kwargs):
                response=await super().complete(*args,**kwargs)
                if 'training-only assessor' in str(args[2]).lower():response.text='not JSON'
                return response
        t,state=self._teacher_fixture(AlwaysBad)
        self.assertIsNone(asyncio.run(t.evidence.get(state,{'fixture':True})))
        self.assertEqual(sum(r['scope']=='teacher' for r in t.ledger.records.values()),3)
        retries=[json.loads(x) for x in (t.out/'teacher_retries.jsonl').read_text().splitlines()]
        self.assertEqual(retries[-1]['action'],'real_rollout_fallback')
        self.assertTrue(all(not x['estimate'].get('valid') for x in
                            map(json.loads,(t.out/'teacher_predictions.jsonl').read_text().splitlines())))

    def test_teacher_retries_obey_phase_cap(self):
        class AlwaysBad(SmokeProvider):
            async def complete(self,*args,**kwargs):
                response=await super().complete(*args,**kwargs)
                if 'training-only assessor' in str(args[2]).lower():response.text='bad'
                return response
        self.cfg['teacher']['llm']['max_calls_per_iteration']=2
        t,state=self._teacher_fixture(AlwaysBad)
        self.assertIsNone(asyncio.run(t.evidence.get(state,{'fixture':True})))
        self.assertEqual(sum(r['scope']=='teacher' for r in t.ledger.records.values()),2)

    def test_five_rounds_finish_with_real_labels_when_teacher_is_always_malformed(self):
        class AlwaysBad(SmokeProvider):
            async def complete(self,*args,**kwargs):
                response=await super().complete(*args,**kwargs)
                if 'training-only assessor' in str(args[2]).lower():response.text='not JSON'
                return response
        self.cfg['online']['iterations']=5
        self.cfg['resilience'].update(retry_delay=0.,qwen_min_interval=0.)
        t=self.trainer()
        t.runner.provider_factory=lambda p,**kw:AlwaysBad(p,strong_model=self.cfg['strong_model'])
        report=asyncio.run(t.run())
        self.assertEqual(report['stop_reason'],'iterations_completed')
        self.assertEqual(report['completed_iterations'],5)
        self.assertTrue(t.pairs)
        self.assertTrue(all(p['source']=='real' for p in t.pairs.values()))
        self.assertTrue(all(r['source']=='real' for r in t.router_rows.values()))
        self.assertTrue((t.out/'teacher_retries.jsonl').exists())

    def test_teacher_envelope_stops_before_global_budget_dispatch(self):
        t,state=self._teacher_fixture(SmokeProvider)
        t.ledger.limit=t.ledger.budget_cost+.000001
        before=len(t.ledger.records)
        with self.assertRaises(BudgetExceeded):asyncio.run(t.evidence.get(state,{'fixture':True}))
        self.assertEqual(before,len(t.ledger.records))

    def test_teacher_stream_error_opens_phase_circuit_but_keeps_valid_cache(self):
        from mixed_pilot_harness.types import ProviderError
        class StreamFailure(SmokeProvider):
            available=False
            async def complete(self,*args,**kwargs):
                if 'training-only assessor' in str(args[2]).lower() and not self.available:
                    raise ProviderError('Responses stream reported an error')
                return await super().complete(*args,**kwargs)
        t,state=self._teacher_fixture(StreamFailure)
        self.assertIsNone(asyncio.run(t.evidence.get(state,{'fixture':'first'})))
        calls=[r for r in t.ledger.records.values() if r['scope']=='teacher']
        self.assertEqual(len(calls),1)
        self.assertIsNone(calls[0]['cost'])
        self.assertGreater(calls[0]['reserved_cost'],0.)
        self.assertTrue(t.ledger.spending_safe)
        before=len(t.ledger.records)
        self.assertIsNone(asyncio.run(t.evidence.get(state,{'fixture':'different'})))
        self.assertEqual(before,len(t.ledger.records))
        key=t.evidence.key(state,{'fixture':'cached'})
        t.evidence.cache[key]={'valid':True,'ready':True,'reward':.9,'cost':.01}
        self.assertTrue(asyncio.run(t.evidence.get(state,{'fixture':'cached'}))['valid'])
        self.assertEqual(before,len(t.ledger.records))
        StreamFailure.available=True;t.evidence.begin_phase()
        self.assertTrue(asyncio.run(t.evidence.get(state,{'fixture':'first'}))['valid'])
        self.assertEqual(sum(r['scope']=='teacher' for r in t.ledger.records.values()),2)

    def test_teacher_transport_fallback_rejects_unbounded_unknown_fee(self):
        from mixed_pilot_mtrouter.online.runtime import UnknownCostError
        t,state=self._teacher_fixture(SmokeProvider)
        cid=t.ledger.begin('teacher',self.cfg['teacher']['llm']['profile'])
        t.ledger.finish(cid,cost=None,complete=False,status='provider_error')
        async def fail(*a,**kw):raise RecoverableRequestError('fixture transport failure')
        t.runner.complete=fail
        with self.assertRaises(UnknownCostError):asyncio.run(t.evidence.get(state,{'fixture':True}))
        self.assertFalse(t.ledger.spending_safe)
        self.assertIsNone(t.ledger.records[cid]['cost'])

    def test_five_rounds_finish_when_teacher_transport_stays_unavailable(self):
        from mixed_pilot_harness.types import ProviderError
        class StreamFailure(SmokeProvider):
            async def complete(self,*args,**kwargs):
                if 'training-only assessor' in str(args[2]).lower():
                    raise ProviderError('Responses stream reported an error')
                return await super().complete(*args,**kwargs)
        self.cfg['online']['iterations']=5
        self.cfg['resilience'].update(enabled=True,retry_delay=0.,qwen_min_interval=0.)
        t=self.trainer()
        t.runner.provider_factory=lambda p,**kw:StreamFailure(p,strong_model=self.cfg['strong_model'])
        report=asyncio.run(t.run())
        self.assertEqual(report['stop_reason'],'iterations_completed')
        self.assertEqual(report['completed_iterations'],5)
        calls=[r for r in t.ledger.records.values() if r['scope']=='teacher']
        self.assertGreater(len(calls),0)
        self.assertLessEqual(len(calls),5)
        self.assertTrue(all(r['cost'] is None and r.get('reserved_cost',0)>0 for r in calls))
        self.assertTrue(all(r['source']=='real' for r in t.router_rows.values()))
        self.assertTrue(t.ledger.spending_safe)
        self.assertFalse(t.ledger.cost_complete)

    def test_runtime_repair_verifies_ancestor_and_preserves_artifacts(self):
        package=self.root/'MTRouter/src/mixed_pilot_mtrouter/online/ksr'
        package.mkdir(parents=True);(package/'original.py').write_text('original source')
        rules_source=self.root/'src/mixed_pilot_harness/step_collection/ksr.py'
        rules_source.parent.mkdir(parents=True);rules_source.write_text('original rules')
        subprocess.run(['git','init','-q'],cwd=self.root,check=True)
        subprocess.run(['git','add',str(package.relative_to(self.root)),str(rules_source.relative_to(self.root))],cwd=self.root,check=True)
        subprocess.run(['git','-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture'],cwd=self.root,check=True)
        rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.root,text=True).strip()
        expected=method_fingerprint('base',self.rules,self.root,revision=rev)
        out=Path(self.cfg['output_dir']);out.mkdir()
        saved={'runtime_fingerprint':expected,'split_fingerprint':'split','reference_fingerprint':'refs'}
        (out/'split_manifest.json').write_text(json.dumps(saved))
        (out/'config.resolved.json').write_text(json.dumps(self.cfg))
        (out/'ledger.json').write_text('preserve exact bytes')
        before={p.name:p.read_bytes() for p in out.iterdir()}
        manifest={**saved,'runtime_fingerprint':'current-execution'}
        receipt=prepare_runtime_repair(self.cfg,manifest,'base',self.rules,rev)
        self.assertEqual(manifest['runtime_fingerprint'],expected)
        self.assertEqual(receipt['execution_runtime_fingerprint'],'current-execution')
        self.assertEqual(before,{p.name:p.read_bytes() for p in out.iterdir()})
        with self.assertRaises(ValueError):
            prepare_runtime_repair(self.cfg,{**saved,'runtime_fingerprint':'new'},'changed-base',self.rules,rev)
        changed=deepcopy(self.cfg);changed['decision']['beta']=99
        with self.assertRaises(ValueError):prepare_runtime_repair(changed,manifest,'base',self.rules,rev)


if __name__ == '__main__': unittest.main()
