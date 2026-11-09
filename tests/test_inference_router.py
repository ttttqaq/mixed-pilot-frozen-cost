"""Offline routing integration: no paid APIs, Docker, or downloaded weights."""
import asyncio
from dataclasses import replace
import io
import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch, Mock

from mixed_pilot_harness.config import ProviderProfile
from mixed_pilot_harness.inference.cli import parser, plan
from mixed_pilot_harness.inference.routing import configure
from mixed_pilot_harness.providers import create_provider
from mixed_pilot_harness.router.profiles import inline_config, snapshot
from mixed_pilot_harness.router.mtrouter import MTRouterEngine
from mixed_pilot_harness.router.input_adapter import build_router_request
from mixed_pilot_harness.types import Message
from mixed_pilot_harness.branch_sampling.core import Collector
import test_inference as fixtures


class RouterTests(unittest.TestCase):
    setUp = fixtures.Tests.setUp
    run_engine = fixtures.Tests.run_engine

    def spec(self, engine='random', **kw):
        cfg=inline_config(dict(engine=engine,models=['a','b'],**kw),self.config.profiles,self.root)
        return snapshot(cfg,'test-router')

    def test_random_pool_probabilities_billing_and_all_step_exports(self):
        self.prices['profiles']['b'].update(input_per_million=2000,output_per_million=2000)
        report,traces,steps,calls=self.run_engine(mode='router',pool=['a','b'],router=self.spec())
        rng=random.Random(42)
        self.assertEqual(calls,[rng.choice(['a','b']) for _ in range(8)])
        self.assertEqual(len(steps),8)
        self.assertTrue(all(s['propensity']==.5 and s['router_decision']['source']=='random' for s in steps))
        self.assertAlmostEqual(report['total_api_cost'],sum(.02 if p=='a' else .04 for p in calls))
        self.assertEqual(len((self.args.output_dir/'samples.jsonl').read_text().splitlines()),8)
        self.assertEqual([s['selected_profile'] for t in traces for s in t['steps']],calls)

    def test_random_resume_replays_without_rerouting_or_rebilling(self):
        original=Collector.one_call
        async def interrupted(collector,*args):
            result=await original(collector,*args)
            if len(collector.calls)==2: raise KeyboardInterrupt()
            return result
        options=dict(mode='router',pool=['a','b'],router=self.spec())
        with patch.object(Collector,'one_call',interrupted),self.assertRaises(KeyboardInterrupt):
            self.run_engine(**options)
        self.args.resume=True
        report,_,_,calls=self.run_engine(**options)
        rng=random.Random(42)
        expected=[rng.choice(['a','b']) for _ in range(8)]
        self.assertEqual(calls,expected[2:])
        self.assertAlmostEqual(report['total_api_cost'],.16)
        self.assertEqual(report['api_attempts'],8)

    def test_mtrouter_service_selects_actual_profile_at_each_step(self):
        requests=[]
        def response(req,timeout):
            payload=json.loads(req.data);requests.append(payload)
            selected='train-b' if len(requests)%2 else 'train-a'
            return io.BytesIO(json.dumps(dict(model=selected,model_names=['train-a','train-b'],
                q_values={m:1 if m==selected else 0 for m in ['train-a','train-b']})).encode())
        spec=self.spec('mtrouter',service_url='http://localhost:8765',model_names=['train-a','train-b'])
        with patch('mixed_pilot_harness.router.mtrouter.urlopen',side_effect=response):
            report,traces,steps,calls=self.run_engine(mode='router',pool=['a','b'],router=spec)
        self.assertEqual(calls,['b','a']*4)
        self.assertEqual([len(r['messages']) for r in requests],[2,4,6,8]*2)
        self.assertEqual(requests[0]['models'],['train-a','train-b'])
        self.assertTrue(all(s['propensity']==1 for s in steps))
        self.assertTrue(all('q_values' in s['router_decision']['metadata'] for s in steps))
        self.assertAlmostEqual(report['total_api_cost'],.16)

    def test_route_failure_has_known_zero_cost_and_partial_can_retry(self):
        spec=self.spec('mtrouter',service_url='http://localhost:8765')
        with patch('mixed_pilot_harness.router.mtrouter.urlopen',side_effect=OSError('offline')):
            report,traces,steps,calls=self.run_engine(mode='router',pool=['a','b'],router=spec)
            self.args.resume_partial=True
            resumed,_,_,_=self.run_engine(mode='router',pool=['a','b'],router=spec)
        self.assertEqual(calls,[])
        self.assertEqual(report['failed_nodes'],2)
        self.assertTrue(report['cost_complete'])
        self.assertEqual(report['total_api_cost'],0)
        self.assertEqual(steps,[])
        self.assertEqual(resumed['superseded_nodes'],2)
        self.assertIsNone(traces[0]['steps'][0]['selected_profile'])

    def test_router_profile_works_outside_infer(self):
        responses=self.root/'responses.json'
        responses.write_text(json.dumps([{'text':'ok','usage':{'input_tokens':1,'output_tokens':1}}]*4))
        for p in self.config.profiles.values(): p.responses_file=str(responses)
        profile=ProviderProfile('router','random',router={'engine':'random','models':['a','b']})
        provider=create_provider(profile,config_dir=self.root,profiles=self.config.profiles)
        asyncio.run(provider.complete([Message('user','test')],[],'system'))
        self.assertEqual(provider.drain_events()[0].data['router_source'],'random')

    def test_cli_profile_pool_override_and_checkpoint_fingerprint(self):
        checkpoint=self.root/'q.pt';checkpoint.write_bytes(b'test checkpoint')
        self.config.profiles['route']=ProviderProfile('router','mtrouter',router={
            'engine':'mtrouter','models':['a','b'],'service_url':'http://localhost:8765'})
        args=parser().parse_args(['--mode','router','--router-profile','route',
            '--models','b','c','--router-checkpoint',str(checkpoint),'--output-dir','unused'])
        spec=configure(args,self.config)
        self.assertEqual(spec['models'],['b','c'])
        self.assertEqual(spec['model_names'],['b','c'])
        self.assertEqual(spec['service_url'],'')
        self.assertEqual(len(spec['checkpoint_sha256']),64)
        args=parser().parse_args(['--mode','single-model','--models','route','--output-dir','unused'])
        self.assertEqual(configure(args,self.config)['models'],['a','b'])
        self.assertEqual(args.mode,'router')

    def test_invalid_pools_and_checkpoints_fail_before_inference(self):
        for raw in [dict(engine='random',models=['a','a']),dict(engine='random',models=['a','missing']),
                    dict(engine='mtrouter',models=['a','b'],checkpoint='missing.pt'),
                    dict(engine='mtrouter',models=['a','b'],service_url='http://localhost',model_names=['x','x'])]:
            with self.subTest(raw=raw),self.assertRaises(ValueError):
                inline_config(raw,self.config.profiles,self.root)

    def test_resume_rejects_router_policy_or_weight_changes(self):
        self.run_engine(mode='router',pool=['a','b'],router=self.spec())
        self.args.resume_partial=True
        with self.assertRaisesRegex(ValueError,'original tasks/settings'):
            self.run_engine(mode='router',pool=['a','b'],router=self.spec(seed=99))

    def test_local_checkpoint_loader_and_pool_order(self):
        # Import only the lightweight service; the neural loader is mocked here.
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'MTRouter/src'))
        self.addCleanup(lambda:sys.path.remove(str(Path(__file__).resolve().parents[1]/'MTRouter/src')))
        checkpoint=self.root/'q.pt';checkpoint.write_bytes(b'fixture')
        cfg=inline_config(dict(engine='mtrouter',models=['a','b'],checkpoint=str(checkpoint)),self.config.profiles,self.root)
        scorer=Mock(model_names=['a','b']);scorer.predict.return_value=[.2,.9]
        with patch('mixed_pilot_mtrouter.service.load_scorer',return_value=scorer) as load:
            engine=MTRouterEngine(cfg)
            decision=engine.route(build_router_request([Message('user','task')],[],'system',[]))
        self.assertEqual(decision.tier,'b')
        self.assertEqual(load.call_args.kwargs['expected_models'],['a','b'])
        with patch('mixed_pilot_mtrouter.service.load_scorer',side_effect=ValueError('Model names mismatch')):
            with self.assertRaisesRegex(ValueError,'names mismatch'):MTRouterEngine(cfg)

    def test_router_plan_validates_and_prices_only_concrete_profiles(self):
        responses=self.root/'responses.json';responses.write_text('[]')
        for p in self.config.profiles.values():p.responses_file=str(responses)
        self.config.profiles['route']=ProviderProfile('router','random',router={'engine':'random','models':['a','b']})
        args=parser().parse_args(['--dataset',str(self.root),'--mode','router','--router-profile','route',
            '--output-dir',str(self.root/'planned'),'--dry-run'])
        with patch('mixed_pilot_harness.inference.cli.load_dataset',return_value=self.tasks), \
             patch('mixed_pilot_harness.inference.cli.load_config',return_value=self.config), \
             patch('mixed_pilot_harness.inference.cli.load_prices',return_value=self.prices) as prices:
            _,_,_,settings=plan(args)
        self.assertEqual(settings.model_pool,['a','b'])
        self.assertEqual(settings.mode,'router')
        self.assertEqual(prices.call_args.args[1],['a','b'])

    def test_router_transient_retries_do_not_call_selected_llm_twice(self):
        from urllib.error import HTTPError
        result=dict(model='a',model_names=['a','b'],q_values={'a':1,'b':0})
        attempts=[]
        def response(req,timeout):
            attempts.append(1)
            if len(attempts)<3:raise HTTPError(req.full_url,503,'retry',{},io.BytesIO(b'busy'))
            return io.BytesIO(json.dumps(result).encode())
        spec=self.spec('mtrouter',service_url='http://localhost:8765')
        with patch('mixed_pilot_harness.router.mtrouter.urlopen',side_effect=response):
            report,_,_,calls=self.run_engine(mode='router',pool=['a','b'],router=spec,tasks=self.tasks[:1])
        self.assertEqual(len(attempts),6)
        self.assertEqual(len(calls),4)
        self.assertAlmostEqual(report['total_api_cost'],.08)

    def test_trace_messages_can_be_read_by_mtrouter_audit(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'MTRouter/src'))
        self.addCleanup(lambda:sys.path.remove(str(Path(__file__).resolve().parents[1]/'MTRouter/src')))
        from mixed_pilot_mtrouter.audit import audit_directories
        self.run_engine(mode='router',pool=['a','b'],router=self.spec())
        report=audit_directories([self.args.output_dir/'traces'],['a','b'])
        self.assertEqual(report['steps'],8)
        self.assertEqual(report['policies'],{'router':2})

    def test_weighted_random_records_actual_probabilities_and_zero_weight(self):
        from mixed_pilot_harness.router.random import RandomRouterEngine
        cfg=inline_config(dict(engine='random',models=['a','b','c'],possibility={'a':6,'b':3,'c':0}),self.config.profiles,self.root)
        engine=RandomRouterEngine(cfg)
        request=build_router_request([Message('user','task')],[],'system',[])
        draws=[engine.route(request) for _ in range(10000)]
        self.assertNotIn('c',[d.tier for d in draws])
        self.assertAlmostEqual(sum(d.tier=='a' for d in draws)/len(draws),2/3,delta=.02)
        for d in draws:
            self.assertEqual(d.metadata['selection_probability'],d.probabilities[d.tier])
        report,traces,steps,calls=self.run_engine(mode='router',pool=['a','b'],router=self.spec(possibility={'a':0,'b':1}))
        self.assertEqual(calls,['b']*8)
        self.assertTrue(all(s['propensity']==1 for s in steps))
        self.assertTrue(all(s['router_decision']['probabilities']=={'a':0,'b':1} for s in steps))
        self.assertAlmostEqual(report['cost_by_profile']['b'],.16)

    def test_weighted_resume_keeps_rng_and_rejects_changed_weights(self):
        original=Collector.one_call
        async def interrupted(collector,*args):
            result=await original(collector,*args)
            if len(collector.calls)==2:raise KeyboardInterrupt()
            return result
        options=dict(mode='router',pool=['a','b'],router=self.spec(possibility={'a':1,'b':3}))
        with patch.object(Collector,'one_call',interrupted),self.assertRaises(KeyboardInterrupt):
            self.run_engine(**options)
        self.args.resume=True
        report,_,_,calls=self.run_engine(**options)
        rng=random.Random(42)
        expected=[rng.choices(['a','b'],weights=[.25,.75],k=1)[0] for _ in range(8)]
        self.assertEqual(calls,expected[2:])
        self.assertAlmostEqual(report['total_api_cost'],.16)
        self.args.resume_partial=True
        with self.assertRaisesRegex(ValueError,'original tasks/settings'):
            self.run_engine(mode='router',pool=['a','b'],router=self.spec(possibility={'a':3,'b':1}))

    def test_possibility_validation_and_uniform_backwards_compatibility(self):
        from mixed_pilot_harness.router.profiles import from_snapshot
        self.assertEqual(self.spec(),self.spec(possibility={'a':1,'b':1}))
        for weights in ({}, {'a':1}, {'a':1,'b':1,'c':1}, {'a':0,'b':0},
                        {'a':-1,'b':1}, {'a':True,'b':1}, {'a':'1','b':1},
                        {'a':float('nan'),'b':1}, {'a':float('inf'),'b':1}, []):
            with self.subTest(weights=weights),self.assertRaises(ValueError):
                self.spec(possibility=weights)
        with self.assertRaisesRegex(ValueError,'only supported'):
            self.spec('mtrouter',service_url='http://localhost:8765',possibility={'a':1,'b':2})
        spec=self.spec(possibility={'a':1e308,'b':1e307})
        self.assertEqual(snapshot(from_snapshot(spec,self.config.profiles),'test-router'),spec)

    def test_cli_weight_override_and_wrong_mode_rejection(self):
        self.config.profiles['route']=ProviderProfile('router','random',router={
            'engine':'random','models':['a','b'],'possibility':{'a':1,'b':3}})
        args=parser().parse_args(['--mode','router','--router-profile','route','--router-possibility','a=4','b=1','--output-dir','unused'])
        self.assertEqual(configure(args,self.config)['possibility'],{'a':4,'b':1})
        for mode,weights in [('single-model',['a=1','b=2']),('random-router',['a=1','a=2']),
                             ('random-router',['a=no','b=2']),('random-router',['a=1'])]:
            args=parser().parse_args(['--mode',mode,'--models','a','b','--router-possibility',*weights,'--output-dir','unused'])
            with self.subTest(mode=mode,weights=weights),self.assertRaises(ValueError):
                configure(args,self.config)

    def test_toml_possibility_uses_profile_names_with_tier_aliases(self):
        from mixed_pilot_harness.router.config import load_router_config
        from mixed_pilot_harness.router.random import RandomRouterEngine
        path=self.root/'random.toml'
        path.write_text('[router]\nengine="random"\n[router.possibility]\na=0\nb=1\n'
                        '[tiers.first]\nprofile="a"\n[tiers.second]\nprofile="b"\n')
        engine=RandomRouterEngine(load_router_config(path,self.config.profiles))
        request=build_router_request([Message('user','task')],[],'system',[])
        self.assertEqual(engine.route(request).tier,'second')
        self.assertEqual(engine.route(request).probabilities,{'first':0,'second':1})
