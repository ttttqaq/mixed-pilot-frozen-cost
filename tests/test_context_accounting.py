"""Offline regression tests for mixed-provider context and retry accounting."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import urllib.error

from mixed_pilot_harness.config import ProviderProfile, load_config
from mixed_pilot_harness.providers import OpenAICompatibleProvider, AnthropicProvider, GeminiProvider, HTTP_RETRY_POLICY
from mixed_pilot_harness.context import encoded_size
from mixed_pilot_harness.types import Message, ToolCall, ProviderError
from mixed_pilot_harness.branch_sampling.core import Collector
from mixed_pilot_harness.inference.report import InferenceCollector
import test_inference as fixtures


RESPONSES = {
    OpenAICompatibleProvider: {'choices':[{'message':{'content':'done'}}], 'usage':{'prompt_tokens':10,'completion_tokens':10}},
    AnthropicProvider: {'content':[{'type':'text','text':'done'}], 'usage':{'input_tokens':10,'output_tokens':10}},
    GeminiProvider: {'candidates':[{'content':{'parts':[{'text':'done'}]}}], 'usageMetadata':{'promptTokenCount':10,'candidatesTokenCount':10}},
}


def failure(known_zero=False, status=None, detail='transport timeout'):
    error = ProviderError(detail, cost_known_zero=known_zero)
    error.__cause__ = urllib.error.HTTPError('https://test',status,'test',{},None) if status else TimeoutError()
    return error


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_large_latest_tool_result_compacts_for_all_protocols_without_mutating_history(self):
        history = [Message('user','task'), Message('assistant', tool_calls=[ToolCall('id','shell',{'command':'ls'})]),
                   Message('tool', json.dumps({'output':'head\n'+'工具输出'*5000+'\ntail'}),tool_call_id='id',name='shell')]
        original = deepcopy(history)
        for cls, response in RESPONSES.items():
            with self.subTest(provider=cls.__name__):
                provider = cls(ProviderProfile('test','test',api_key_optional=True,context_window=5000,max_tokens=100))
                payloads=[]
                def post(url,payload,headers):
                    payloads.append(deepcopy(payload));return response
                with patch.object(provider,'_post_json_sync',side_effect=post):
                    result = await provider.complete(history,[],'system')
                self.assertEqual(history,original)
                self.assertTrue(result.context_management['lossy'])
                self.assertLessEqual(encoded_size(payloads[0])+1124,5000)
                self.assertIn('shell',json.dumps(payloads[0]))
                self.assertEqual(result.usage.input_tokens,10)

    async def test_huge_arguments_replaced_as_whole_group_and_immutable_system_fails_free(self):
        provider=OpenAICompatibleProvider(ProviderProfile('openai','test',api_key_optional=True,context_window=3000,max_tokens=100))
        history=[Message('user','task'),Message('assistant',tool_calls=[ToolCall('id','edit',{'patch':'x'*10000})]),Message('tool','ok',tool_call_id='id')]
        with patch.object(provider,'_post_json_sync',return_value=RESPONSES[OpenAICompatibleProvider]) as post:
            result=await provider.complete(history,[],'system')
            wire=post.call_args.args[1]['messages']
            self.assertFalse(any(m.get('tool_calls') or m['role']=='tool' for m in wire))
            self.assertEqual(result.context_management['flattened_groups'],1)
            with self.assertRaises(ProviderError) as cm:
                await provider.complete(history,[],'system'*2000)
            self.assertTrue(cm.exception.cost_known_zero)
            self.assertEqual(post.call_count,1)

    async def test_retry_uncertainty_survives_success_on_every_protocol(self):
        token=HTTP_RETRY_POLICY.set((1,0))
        try:
            for cls, response in RESPONSES.items():
                for known_zero in (False,True):
                    provider=cls(ProviderProfile('test','test',api_key_optional=True))
                    error=failure(known_zero,status=429 if known_zero else None)
                    with patch.object(provider,'_post_json_sync',side_effect=[error,response]):
                        result=await provider.complete([Message('user','task')],[],'system')
                    self.assertEqual(result.cost_complete,known_zero)
                    self.assertEqual(len(result.api_attempts),2)
                    self.assertEqual(result.usage.input_tokens,10)
        finally:
            HTTP_RETRY_POLICY.reset(token)

    async def test_server_context_rejection_recompresses_and_preserves_prior_unknown_charge(self):
        token=HTTP_RETRY_POLICY.set((1,0))
        try:
            provider=OpenAICompatibleProvider(ProviderProfile('openai','test',api_key_optional=True,context_window=20000,max_tokens=100))
            captured=[]
            outcomes=[failure(),failure(True,400,'maximum context length exceeded'),RESPONSES[OpenAICompatibleProvider]]
            def post(url,payload,headers):
                captured.append(deepcopy(payload));value=outcomes.pop(0)
                if isinstance(value,Exception): raise value
                return value
            with patch.object(provider,'_post_json_sync',side_effect=post):
                result=await provider.complete([Message('user','x'*10000)],[],'system')
            self.assertLess(encoded_size(captured[-1]),encoded_size(captured[0]))
            self.assertFalse(result.cost_complete)
            self.assertEqual(len(result.api_attempts),3)
            self.assertEqual(len(result.context_management['server_recompressions']),1)
        finally:
            HTTP_RETRY_POLICY.reset(token)

    async def test_separate_input_limit_is_enforced(self):
        provider=OpenAICompatibleProvider(ProviderProfile('openai','test',api_key_optional=True,context_window=20000,max_input_tokens=4000,max_tokens=100))
        with patch.object(provider,'_post_json_sync',return_value=RESPONSES[OpenAICompatibleProvider]) as post:
            await provider.complete([Message('user','x'*15000)],[],'system')
        self.assertLessEqual(encoded_size(post.call_args.args[1])+1024,4000)

    async def test_context_retries_are_bounded_and_free_rejections_stay_free(self):
        provider=OpenAICompatibleProvider(ProviderProfile('openai','test',api_key_optional=True,context_window=100000,max_tokens=100))
        def post(*args):
            raise failure(True,400,'maximum context length exceeded')
        with patch.object(provider,'_post_json_sync',side_effect=post) as request:
            with self.assertRaises(ProviderError) as cm:
                await provider.complete([Message('user','x'*50000)],[],'system')
        self.assertEqual(request.call_count,4)
        self.assertTrue(cm.exception.cost_known_zero)
        self.assertEqual(len(cm.exception.api_attempts),4)
        self.assertEqual(len(cm.exception.context_management['server_recompressions']),3)

    async def test_switching_to_small_window_does_not_destroy_large_window_history(self):
        history=[Message('user','task'),Message('assistant','action'),Message('user','output'*2000)]
        sizes=[]
        for window in (30000,3000,30000):
            provider=OpenAICompatibleProvider(ProviderProfile('openai','test',api_key_optional=True,context_window=window,max_tokens=100))
            with patch.object(provider,'_post_json_sync',return_value=RESPONSES[OpenAICompatibleProvider]) as post:
                await provider.complete(history,[],'system')
                sizes.append(encoded_size(post.call_args.args[1]))
        self.assertEqual(sizes[0],sizes[2])
        self.assertLess(sizes[1],sizes[0])


class AccountingTests(unittest.TestCase):
    setUp=fixtures.Tests.setUp
    run_engine=fixtures.Tests.run_engine

    def test_live_summary_and_resume_retain_uncertainty(self):
        original=fixtures.Provider.complete
        snapshots=[]
        save=InferenceCollector.save
        async def uncertain(provider,*args):
            result=await original(provider,*args)
            result.cost_complete=False
            result.api_attempts=[{'status':'error','cost_known_zero':False},{'status':'response'}]
            return result
        def capture(collector):
            save(collector)
            if collector.total_cost:
                snapshots.append(json.loads((collector.out/'summary.json').read_text()))
        with patch.object(fixtures.Provider,'complete',uncertain),patch.object(InferenceCollector,'save',capture):
            report,traces,steps,calls=self.run_engine(stop_on_unknown_cost=False)
        self.assertEqual(len(calls),8)
        self.assertFalse(report['cost_complete'])
        self.assertAlmostEqual(report['cost_by_profile']['a'],.16)
        self.assertEqual(report['profile_costs']['a']['CNY']['uncertain_calls'],8)
        self.assertTrue(all('cost_by_profile' in r for r in snapshots))
        self.assertTrue(any(r['cost_by_profile']['a']==.02 for r in snapshots))
        self.assertTrue(all(not t['info']['cost_complete'] for t in traces))
        self.assertEqual(steps,[])  # No complete-cost training labels.
        self.assertIn('lower bounds',(self.args.output_dir/'SUMMARY.md').read_text())
        self.args.resume=True
        report,_,_,calls=self.run_engine(stop_on_unknown_cost=False)
        self.assertFalse(report['cost_complete'])
        self.assertEqual(calls,[])
        self.assertAlmostEqual(report['total_api_cost'],.16)

    def test_budget_guard_stops_after_unknown_retry_but_keeps_successful_charge(self):
        original=fixtures.Provider.complete
        async def uncertain(provider,*args):
            result=await original(provider,*args);result.cost_complete=False;return result
        with patch.object(fixtures.Provider,'complete',uncertain):
            report,_,_,calls=self.run_engine()
        self.assertEqual(len(calls),1)
        self.assertEqual(report['stop_reason'],'unknown_api_cost')
        self.assertAlmostEqual(report['total_api_cost'],.02)
        self.args.resume=True
        with self.assertRaisesRegex(ValueError,'unknown API cost'):
            self.run_engine()

    def test_budget_continues_after_uncertain_retry_and_reaches_known_limit(self):
        original=fixtures.Provider.complete
        async def uncertain(provider,*args):
            result=await original(provider,*args)
            result.cost_complete=False
            return result
        with patch.object(fixtures.Provider,'complete',uncertain):
            report,_,steps,calls=self.run_engine(until_budget=True,total_cost_limit=.19,
                                               stop_on_unknown_cost=False)
        self.assertEqual(report['stop_reason'],'budget_exhausted')
        self.assertEqual(len(calls),10)
        self.assertAlmostEqual(report['total_api_cost'],.2)
        self.assertFalse(report['cost_complete'])
        self.assertEqual(steps,[])

    def test_resume_partial_can_disable_previous_unknown_cost_guard(self):
        original=fixtures.Provider.complete
        async def uncertain(provider,*args):
            result=await original(provider,*args)
            result.cost_complete=False
            return result
        with patch.object(fixtures.Provider,'complete',uncertain):
            report,_,_,_=self.run_engine(until_budget=True,total_cost_limit=.19)
        self.assertEqual(report['stop_reason'],'unknown_api_cost')
        self.args.resume_partial=True
        report,_,_,calls=self.run_engine(until_budget=True,total_cost_limit=.19,
                                       stop_on_unknown_cost=False)
        self.assertEqual(report['stop_reason'],'budget_exhausted')
        self.assertFalse(report['cost_complete'])
        self.assertEqual(len(calls),9)
        self.assertAlmostEqual(report['total_api_cost'],.2)

    def test_all_unknown_round_stops_without_infinite_budget_loop(self):
        async def broken(*args,**kwargs):
            raise ProviderError('transport timeout')
        with patch.object(fixtures.Provider,'complete',broken):
            report,traces,_,_=self.run_engine(until_budget=True,total_cost_limit=.19,
                                            stop_on_unknown_cost=False)
        self.assertEqual(len(traces),2)
        self.assertEqual(report['stop_reason'],'no_cost_progress')
        self.assertFalse(report['cost_complete'])
        self.assertEqual(report['total_api_cost'],0)

    def test_profile_totals_include_judge_and_keep_currencies_separate(self):
        original=fixtures.Episode.evaluate
        async def evaluate(ep,**kwargs):
            result=await original(ep,**kwargs)
            result.update(cost=.5,cost_complete=False,selected_profile='b',api_calls=1,
                api_attempts=[{'status':'error','cost_known_zero':False},{'status':'response'}])
            return result
        with patch.object(fixtures.Episode,'evaluate',evaluate):
            report,_,_,_=self.run_engine(stop_on_unknown_cost=False,total_cost_limit=1e101,task_cost_limit=1e101,path_cost_limit=1e101)
        self.assertAlmostEqual(report['profile_costs']['b']['CNY']['evaluation_cost'],1)
        self.assertFalse(report['cost_complete'])
        self.assertAlmostEqual(report['total_api_cost'],1.16)
        self.args.resume_partial=True
        self.prices['currency']='USD'
        report,_,_,_=self.run_engine(stop_on_unknown_cost=False,total_cost_limit=1e101,task_cost_limit=1e101,path_cost_limit=1e101)
        self.assertAlmostEqual(report['profile_costs']['b']['CNY']['evaluation_cost'],1)
        self.assertEqual(report['total_api_cost'],0)
        self.assertAlmostEqual(report['cost_by_currency']['CNY'],1.16)

    def test_configured_concrete_profiles_have_official_windows(self):
        cfg=load_config(Path(__file__).resolve().parents[1]/'harness.json')
        for profile in cfg.profiles.values():
            if profile.kind!='router':self.assertGreater(profile.context_window,profile.max_tokens)
        self.assertEqual(cfg.profile('qwen3-8b').max_input_tokens,98304)

    def test_real_hle_judge_propagates_successful_retry_uncertainty(self):
        from types import SimpleNamespace
        from mixed_pilot_harness.collection.episodes import HLEEpisode
        from mixed_pilot_harness.types import ProviderResponse, Usage
        from unittest.mock import AsyncMock
        reference=self.root/'reference.json'
        reference.write_text(json.dumps({'answer':'gold'}))
        ep=object.__new__(HLEEpisode)
        ep.answer='candidate';ep.question='question';ep.task={'reference_path':str(reference)}
        response=ProviderResponse(text='{"correct":true}',usage=Usage(10,10),cost_complete=False,
            api_attempts=[{'status':'error','cost_known_zero':False},{'status':'response'}])
        result=asyncio.run(ep.evaluate(config=self.config,pricing=self.prices,
            settings=SimpleNamespace(hle_evaluator='judge',judge_profile='b'),
            provider_factory=lambda *a,**kw:SimpleNamespace(complete=AsyncMock(return_value=response))))
        self.assertTrue(result['success'])
        self.assertFalse(result['cost_complete'])
        self.assertAlmostEqual(result['cost'],.02)
        self.assertEqual(len(result['api_attempts']),2)
