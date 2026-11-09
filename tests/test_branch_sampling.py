import asyncio
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from mixed_pilot_harness.branch_sampling import cli
from mixed_pilot_harness.branch_sampling.core import Settings, collect_ksr, collect_ksr_step, collect_random, label_tree
from mixed_pilot_harness.branch_sampling.rules import Rules
from mixed_pilot_harness.branch_sampling.selection import select_ksr_tasks
from mixed_pilot_harness.config import HarnessConfig, ProviderProfile
from mixed_pilot_harness.types import ProviderResponse, Usage

TASK={'task_id':'task0','benchmark':'scienceworld','task_name':'boil','variation_idx':0,'simplification_str':'','split':'train'}


class Env:
    def __init__(self,**kwargs):self.i=0
    def load_task(self,*args):self.i=0;return {'task_description':'boil water','observation':'initial'}
    def step(self,command):
        self.i+=1
        done=self.i>=4
        return {'observation':('done '+command) if done else 'No known action matches that input.',
                'score':(100 if command=='strong' else -100) if done else 0,'done':done}
    def close(self):pass


class Provider:
    def __init__(self,profile,requests):self.profile=profile;self.requests=requests
    async def complete(self,messages,tools,system):
        self.requests.append(self.profile.model)
        return ProviderResponse(text='```text\n'+self.profile.model+'\n```',model=self.profile.model,usage=Usage(10,10))


def config(root):
    return HarnessConfig(Path(root)/'config.json','strong',{
        n:ProviderProfile(kind='scripted',model=n) for n in ['strong','weak']},{},{})

PRICING={'currency':'CNY','as_of':'2026-01-01','profiles':{n:{'input_per_million':1000,'output_per_million':1000} for n in ['strong','weak']}}


class RecoveryEnv(Env):
    """A strong action has a lasting effect that subsequent weak calls preserve."""
    def __init__(self,**kwargs):
        super().__init__(**kwargs);self.recovered=False

    def step(self,command):
        self.i+=1
        self.recovered=self.recovered or command=='strong'
        return {'observation':'No known action matches that input.',
                'score':(100 if self.recovered else -100) if self.i>=8 else 0,
                'done':self.i>=8}


class Tests(unittest.TestCase):
    def run_collect(self,root,mode='ksr',factory=Env,provider=None,step_output_dir=None,**overrides):
        requests=[]
        pf=provider or (lambda p,**kw:Provider(p,requests))
        s=Settings(mode=mode,strong='strong',weak='weak',max_steps=8,total_cost_limit=100,task_cost_limit=100,path_cost_limit=100,
                   max_nodes=6,max_depth=1,rules=Rules(error_window=2,repeat_window=3,min_branch_gap=2))
        s=replace(s,**overrides)
        fn={'ksr':collect_ksr,'ksr_step':collect_ksr_step,'random':collect_random}[mode]
        out=Path(root)/mode
        summary=asyncio.run(fn([TASK],config(root),PRICING,s,out,env_factory=factory,provider_factory=pf,step_output_dir=step_output_dir))
        return summary,json.loads((out/'tree.json').read_text()),[json.loads(x) for x in (out/'samples.jsonl').read_text().splitlines()],requests

    def test_tree_shared_prefix_cost_and_persistent_models(self):
        with tempfile.TemporaryDirectory() as root:
            summary,tree,samples,requests=self.run_collect(root)
            self.assertEqual(len(tree['nodes']),6)
            self.assertEqual(len(requests),12) # roots 2*2 + children 4*2; replay is free
            self.assertAlmostEqual(summary['total_api_cost'],12*0.02)
            self.assertEqual(len(samples),6) # first call of each segment only
            self.assertEqual(sum(x['paired_actions_complete'] for x in samples),6)
            self.assertEqual([n['y'] for n in tree['nodes'][:2]],[0.5,0.5])
            for n in tree['nodes'][2:]:
                self.assertEqual(n['y'],1 if n['profile']=='strong' else 0)
                for cid in n['own_call_ids']:
                    call=json.loads((Path(root)/'ksr/calls'/f'{cid}.json').read_text())
                    self.assertEqual(call['selected_profile'],n['profile'])
                first=json.loads((Path(root)/'ksr/calls'/f"{n['own_call_ids'][0]}.json").read_text())
                self.assertEqual(first['step_idx'],2)
                self.assertEqual(len(first['state']['previous_actions']),2)

    def test_random_samples_every_step(self):
        with tempfile.TemporaryDirectory() as root:
            summary,tree,samples,requests=self.run_collect(root,mode='random')
            self.assertEqual(len(samples),4);self.assertEqual(len(tree['nodes']),1)
            self.assertTrue(all(x['selection_probability']==0.5 for x in samples))
            self.assertEqual(len({x['y'] for x in samples}),1)
            self.assertEqual(len(requests),4)

    def test_single_step_probes_resume_weak_and_label_only_mainline_continuations(self):
        with tempfile.TemporaryDirectory() as root:
            export=Path(root)/'steps'
            summary,tree,samples,requests=self.run_collect(root,mode='ksr_step',factory=RecoveryEnv,
                max_nodes=7,max_depth=3,step_output_dir=export)
            nodes=tree['nodes'];by_id={n['id']:n for n in nodes}
            self.assertEqual(len(tree['jobs'][0]['root_ids']),1)
            initial=nodes[0]
            self.assertEqual(initial['profile'],'weak')
            self.assertIsNone(initial['branch_point_id'])
            self.assertFalse(any(s['node_id']==initial['id'] for s in samples))
            self.assertEqual(len(nodes),7)
            self.assertEqual(summary['complete_pairs'],3)
            self.assertEqual(len(samples),6)
            self.assertEqual(len(requests),20) # 8 mainline calls + 6/4/2 probe calls
            self.assertEqual(requests.count('strong'),3)
            self.assertAlmostEqual(summary['total_api_cost'],0.4)
            self.assertEqual(summary['failed_nodes'],0)
            self.assertEqual(initial['y'],0) # Successful probes never enter the weak label.
            self.assertAlmostEqual(initial['remaining_api_cost'],0.16)
            calls={p.stem:json.loads(p.read_text()) for p in (Path(root)/'ksr_step/calls').glob('*.json')}
            for node in nodes:
                own=[calls[c] for c in node['own_call_ids']]
                if node['node_role']=='probe':
                    self.assertEqual(node['children'],[]) # Errors on probes never trigger forks.
                    self.assertEqual([c['selected_profile'] for c in own],['strong']+['weak']*(len(own)-1))
                    self.assertEqual(node['y'],1)
                    self.assertIn('strong',own[-1]['state']['previous_actions'])
                else:
                    self.assertTrue(all(c['selected_profile']=='weak' for c in own))
                    self.assertEqual(node['y'],0)
                if node['children']:
                    children=[by_id[c] for c in node['children']]
                    first=[calls[c['own_call_ids'][0]] for c in children]
                    self.assertEqual(first[0]['state'],first[1]['state'])
                    self.assertEqual(first[0]['step_idx'],first[1]['step_idx'])
            self.assertEqual(sorted({s['step_idx'] for s in samples}),[2,4,6])
            for sample in samples:
                self.assertEqual(sample['base_profile'],'weak')
                self.assertEqual(sample['continuation_profile'],'weak')
                self.assertEqual(sample['label_method'],'single_step_then_weak_terminal_score')
                self.assertEqual(sample['action_scope'],'single_step')
                self.assertIsNone(sample['selection_probability'])
                self.assertTrue(sample['branch_enumerated'])
                self.assertTrue(sample['paired_actions_complete'])
                self.assertAlmostEqual(sample['remaining_api_cost'],(8-sample['step_idx'])*0.02)
            plan=json.loads((Path(root)/'ksr_step/plan.json').read_text())
            self.assertFalse(plan['root_branching'])
            self.assertFalse(plan['probe_branching'])
            self.assertEqual(plan['base_profile'],'weak')
            self.assertEqual(set(plan['settings']['rules']),{'error_window','repeat_window','min_branch_gap'})
            traces=[json.loads(l) for l in (export/'traces.jsonl').read_text().splitlines()]
            self.assertEqual(len(traces),4)
            self.assertEqual(sorted(t['y'] for t in traces),[0,1,1,1])
            self.assertTrue(all(t['steps']==8 for t in traces))
            self.assertEqual(summary['trace_step_export']['unique_exported_calls'],20)

    def test_single_step_without_trigger_has_only_weak_root_and_no_samples(self):
        class ProgressEnv(Env):
            def step(self,command):
                self.i+=1
                # Constant native score must not trigger KSR, even after 6 steps.
                return {'observation':'exploring','score':0,'done':self.i>=8}
        with tempfile.TemporaryDirectory() as root:
            summary,tree,samples,requests=self.run_collect(root,mode='ksr_step',factory=ProgressEnv,
                rules=Rules(error_window=1,repeat_window=20,min_branch_gap=2))
            self.assertEqual(requests,['weak']*8)
            self.assertEqual(len(tree['nodes']),1)
            self.assertEqual(samples,[])
            self.assertEqual(summary['complete_pairs'],0)

    def test_single_step_node_and_depth_limits_keep_running_weak(self):
        for limits,count in [({'max_nodes':1},0),({'max_depth':0},0),
                             ({'max_nodes':3,'max_depth':10},1),({'max_nodes':20,'max_depth':2},2)]:
            with self.subTest(limits=limits),tempfile.TemporaryDirectory() as root:
                summary,tree,samples,requests=self.run_collect(root,mode='ksr_step',factory=RecoveryEnv,**limits)
                self.assertEqual(summary['complete_pairs'],count)
                self.assertEqual(requests.count('strong'),count)
                self.assertEqual(len(tree['nodes']),1+2*count)
                main_leaf=next(n for n in tree['nodes'] if n['node_role']=='mainline' and not n['children'])
                self.assertEqual(main_leaf['status'],'EnvironmentDone')
                self.assertEqual(len(main_leaf['prefix_call_ids']+main_leaf['own_call_ids']),8)

    def test_single_step_failed_probe_does_not_poison_weak_return(self):
        class FailedProbe(RecoveryEnv):
            def step(self,command):
                if command=='strong':raise RuntimeError('probe action failed')
                return super().step(command)
        with tempfile.TemporaryDirectory() as root:
            summary,tree,samples,requests=self.run_collect(root,mode='ksr_step',factory=FailedProbe)
            self.assertEqual(summary['failed_nodes'],1)
            self.assertEqual(tree['nodes'][0]['y'],0)
            self.assertEqual(len(samples),2)
            strong=next(s for s in samples if s['action_profile']=='strong')
            weak=next(s for s in samples if s['action_profile']=='weak')
            self.assertIsNone(strong['y']);self.assertIsNone(strong['remaining_api_cost'])
            self.assertEqual(weak['y'],0)
            self.assertFalse(any(s['paired_actions_complete'] for s in samples))
            self.assertAlmostEqual(summary['total_api_cost'],0.18) # Failed call is still billed.

    def test_single_step_replay_failure_keeps_probe_but_invalidates_mainline(self):
        class BadReplay(RecoveryEnv):
            made=0
            def __init__(self,**kw):
                super().__init__(**kw);BadReplay.made+=1;self.instance=BadReplay.made
            def load_task(self,*args):
                result=super().load_task(*args)
                if self.instance>1:result['observation']='different initial state'
                return result
        with tempfile.TemporaryDirectory() as root:
            summary,tree,samples,requests=self.run_collect(root,mode='ksr_step',factory=BadReplay)
            self.assertEqual(len(requests),8)
            self.assertEqual(summary['failed_nodes'],1)
            self.assertIsNone(tree['nodes'][0]['y'])
            self.assertEqual(len(samples),1)
            self.assertEqual(samples[0]['y'],1)
            self.assertFalse(samples[0]['paired_actions_complete'])

    def test_single_step_budget_labels_follow_each_rollouts_completion(self):
        for budget,expected in [({'total_cost_limit':0.17},{'strong':1,'weak':None}),
                                ({'max_steps':3},{'strong':0.5,'weak':0.5})]:
            with self.subTest(budget=budget),tempfile.TemporaryDirectory() as root:
                summary,tree,samples,requests=self.run_collect(root,mode='ksr_step',factory=RecoveryEnv,**budget)
                self.assertEqual({s['action_profile']:s['y'] for s in samples},expected)
                self.assertEqual(summary['complete_pairs'],int(all(y is not None for y in expected.values())))

    def test_global_budget_invalidates_incomplete_returns(self):
        with tempfile.TemporaryDirectory() as root:
            summary,tree,samples,requests=self.run_collect(root,total_cost_limit=0.03)
            self.assertEqual(len(requests),2)
            self.assertAlmostEqual(summary['total_api_cost'],0.04)
            self.assertTrue(all(x['y'] is None for x in samples))
            self.assertFalse(any(x['paired_actions_complete'] for x in samples))

    def test_path_limit_is_valid_budgeted_terminal(self):
        with tempfile.TemporaryDirectory() as root:
            summary,tree,samples,requests=self.run_collect(root,path_cost_limit=0.03)
            self.assertEqual(len(requests),4)
            self.assertTrue(all(n['status']=='path_budget' for n in tree['nodes']))
            self.assertTrue(all(x['y']==0.5 for x in samples))

    def test_replay_mismatch_does_not_call_model_or_fake_label(self):
        class BadReplay(Env):
            made=0
            def __init__(self,**kw):
                super().__init__(**kw);BadReplay.made+=1;self.instance=BadReplay.made
            def load_task(self,*args):
                r=super().load_task(*args)
                if self.instance>1:r['observation']='different initial'
                return r
        with tempfile.TemporaryDirectory() as root:
            summary,tree,samples,requests=self.run_collect(root,factory=BadReplay)
            self.assertEqual(len(requests),4) # live switched child survives replay failure
            self.assertEqual(summary['failed_nodes'],2)
            self.assertEqual(tree['nodes'][0]['y'],None)
            live=[n for n in tree['nodes'] if n['state_source']=='live_continuation']
            self.assertEqual(len(live),1)
            self.assertEqual(live[0]['status'],'EnvironmentDone')
            self.assertEqual(live[0]['replay'],[])

    def test_live_continuation_reuses_episode_and_halves_restores(self):
        instances=[]
        class Tracked(Env):
            def __init__(self,**kw):
                super().__init__(**kw);self.actions=[];self.closed=0;instances.append(self)
            def step(self,command):
                self.actions.append(command);return super().step(command)
            def close(self):self.closed+=1
        with tempfile.TemporaryDirectory() as root:
            summary,tree,samples,requests=self.run_collect(root,factory=Tracked)
            self.assertEqual(len(instances),4) # 2 roots + 2 unchanged children, previously 6
            self.assertEqual(instances[0].actions,['strong','strong','weak','weak'])
            self.assertEqual(instances[1].actions,['weak','weak','strong','strong'])
            self.assertTrue(all(ep.closed==1 for ep in instances))
            by_id={n['id']:n for n in tree['nodes']}
            for n in tree['nodes'][2:]:
                parent=by_id[n['parent_id']]
                if n['profile']!=parent['profile']:
                    self.assertEqual(n['state_source'],'live_continuation')
                    self.assertEqual(n['replay'],[])
                else:
                    self.assertEqual(n['state_source'],'replay')
                    self.assertTrue(n['replay'][0]['matched'])

    def test_leaf_step_export_preserves_shared_prefix_and_leaf_labels(self):
        with tempfile.TemporaryDirectory() as root:
            output=Path(root)/'steps'
            summary,tree,samples,requests=self.run_collect(root,step_output_dir=output)
            traces=[json.loads(l) for l in (output/'traces.jsonl').read_text().splitlines()]
            steps=[json.loads(l) for l in (output/'steps.jsonl').read_text().splitlines()]
            self.assertEqual(len(traces),4);self.assertEqual(len(steps),16)
            self.assertEqual(len({s['sample_id'] for s in steps}),16)
            self.assertEqual(len({s['call_id'] for s in steps}),12)
            for trace in traces:
                rows=[s for s in steps if s['trace_id']==trace['trace_id']]
                self.assertEqual([s['step_idx'] for s in rows],[0,1,2,3])
                self.assertTrue(all(s['trace_complete'] and s['y']==trace['y'] for s in rows))
            same_call=[s for s in steps if s['call_id']=='c00000000']
            self.assertEqual({s['y'] for s in same_call},{0,1})
            self.assertTrue(all(s['call_multiplicity']==2 for s in same_call))
            self.assertEqual(samples[0]['y'],0.5) # fork label remains expectation
            self.assertAlmostEqual(summary['total_api_cost'],0.24)

    def test_nested_switches_keep_exact_sibling_state_and_node_limits(self):
        class LongEnv(Env):
            def step(self,command):
                self.i+=1
                return {'observation':'No known action matches that input.',
                        'score':0,'done':self.i>=8}
        with tempfile.TemporaryDirectory() as root:
            summary,tree,samples,requests=self.run_collect(root,factory=LongEnv,max_nodes=14,max_depth=2)
            nodes=tree['nodes'];by_id={n['id']:n for n in nodes}
            self.assertEqual(len(nodes),14)
            self.assertEqual(max(n['depth'] for n in nodes),2)
            self.assertEqual(summary['failed_nodes'],0)
            for parent in nodes:
                if not parent['children']:continue
                children=[by_id[c] for c in parent['children']]
                self.assertEqual(sum(c['state_source']=='live_continuation' for c in children),1)
                calls=[json.loads((Path(root)/'ksr/calls'/f"{c['own_call_ids'][0]}.json").read_text()) for c in children]
                self.assertEqual(calls[0]['state'],calls[1]['state'])
                self.assertEqual(calls[0]['step_idx'],calls[1]['step_idx'])

    def test_step_limit_export_keeps_budgeted_label_but_not_complete_flag(self):
        with tempfile.TemporaryDirectory() as root:
            output=Path(root)/'steps'
            self.run_collect(root,step_output_dir=output,max_steps=3)
            rows=[json.loads(l) for l in (output/'steps.jsonl').read_text().splitlines()]
            self.assertTrue(rows)
            self.assertTrue(all(s['labels_valid'] and s['y']==0.5 and not s['trace_complete'] for s in rows))

    def test_export_incomplete_trace_has_null_labels(self):
        with tempfile.TemporaryDirectory() as root:
            output=Path(root)/'steps'
            summary,tree,samples,requests=self.run_collect(root,step_output_dir=output,total_cost_limit=0.05)
            rows=[json.loads(l) for l in (output/'steps.jsonl').read_text().splitlines()]
            self.assertTrue(rows)
            self.assertTrue(all(s['y'] is None and not s['trace_complete'] for s in rows))

    def test_export_existing_directory_rejected_before_calls(self):
        with tempfile.TemporaryDirectory() as root:
            output=Path(root)/'steps';output.mkdir();(output/'keep').write_text('keep')
            with self.assertRaises(FileExistsError):self.run_collect(root,step_output_dir=output)
            self.assertFalse((Path(root)/'ksr').exists())
            self.assertEqual((output/'keep').read_text(),'keep')

    def test_api_timeout_marks_unknown_cost_and_stops(self):
        class Broken:
            async def complete(self,*args):raise TimeoutError('unknown upstream billing')
        with tempfile.TemporaryDirectory() as root:
            summary,tree,samples,requests=self.run_collect(root,provider=lambda *a,**k:Broken())
            self.assertFalse(summary['cost_complete']);self.assertEqual(summary['api_attempts'],1)
            self.assertEqual(summary['total_api_cost'],0);self.assertEqual(samples,[])

    def test_environment_error_retains_billed_cost(self):
        class BrokenEnv(Env):
            def step(self,command):raise RuntimeError('simulator error')
        with tempfile.TemporaryDirectory() as root:
            summary,tree,samples,requests=self.run_collect(root,factory=BrokenEnv)
            self.assertAlmostEqual(summary['total_api_cost'],0.04)
            self.assertTrue(summary['cost_complete']);self.assertTrue(all(s['y'] is None for s in samples))

    def test_task_selection_is_instance_based(self):
        with tempfile.TemporaryDirectory() as root:
            for name,success in [('strong',True),('weak',False)]:
                p=Path(root)/name;p.mkdir()
                (p/'one.traj.json').write_text(json.dumps({'info':{**TASK,'task_id':'other-spelling','labels_valid':True,'success':success}}))
            tasks,evidence=select_ksr_tasks([TASK],root,'strong','weak')
            self.assertEqual(tasks,[TASK]);self.assertEqual(len(evidence),1)
            self.assertEqual(select_ksr_tasks([{**TASK,'variation_idx':1}],root,'strong','weak')[0],[])

    def test_task_budget_counts_both_roots(self):
        with tempfile.TemporaryDirectory() as root:
            summary,tree,samples,requests=self.run_collect(root,task_cost_limit=0.05)
            self.assertEqual(len(requests),3)
            self.assertAlmostEqual(summary['total_api_cost'],0.06)
            self.assertTrue(all(s['y'] is None for s in samples))
            self.assertEqual(summary['jobs'][0]['status'],'partial')

    def test_nested_return_uses_fork_weights_not_flat_leaf_mean(self):
        nodes=[
            {'id':'a','status':'split','children':['b','c'],'own_call_ids':[]},
            {'id':'b','status':'Submitted','labels_valid':True,'final_score':100,'own_call_ids':[]},
            {'id':'c','status':'split','children':['d','e'],'own_call_ids':[]},
            {'id':'d','status':'Submitted','labels_valid':True,'final_score':-100,'own_call_ids':[]},
            {'id':'e','status':'Submitted','labels_valid':True,'final_score':100,'own_call_ids':[]},
        ]
        label_tree(nodes,{},'ksr')
        self.assertEqual(nodes[0]['y'],0.75) # not 2/3 (flat leaf mean)

    def test_no_branch_before_minimum_gap(self):
        rule=Rules(min_branch_gap=3)
        s={'format_error':True,'score_before':0,'score_after':0}
        self.assertIsNone(rule.detect([s,s]));self.assertEqual(rule.detect([s,s,s])['rule'],'consecutive_errors')

    def test_rules_only_use_errors_and_repeated_commands(self):
        rule=Rules(error_window=1,repeat_window=2,min_branch_gap=2)
        steps=[{'format_error':False,'command':str(i),'observation':'ok',
                'score_before':0,'score_after':0} for i in range(10)]
        self.assertIsNone(rule.detect(steps))
        steps[-1]['command']=steps[-2]['command']
        self.assertEqual(rule.detect(steps)['rule'],'repeated_command')
        steps[-1]['format_error']=True
        self.assertEqual(rule.detect(steps)['rule'],'consecutive_errors')

    def test_single_step_cli_plan_and_execution_dispatch(self):
        with tempfile.TemporaryDirectory() as root:
            tasks=Path(root)/'tasks.json';tasks.write_text(json.dumps([TASK]))
            baseline=Path(root)/'baseline'
            for profile,success in [('strong',True),('weak',False)]:
                directory=baseline/profile;directory.mkdir(parents=True)
                (directory/'trace.traj.json').write_text(json.dumps({'info':{
                    **TASK,'labels_valid':True,'success':success}}))
            args=['--mode','ksr_step','--tasks-root',str(tasks),'--baseline-root',str(baseline),
                  '--strong','strong','--weak','weak','--output-dir',str(Path(root)/'out'),
                  '--error-window','1','--repeat-window','2','--min-branch-gap','2']
            result={'cost_complete':True,'failed_nodes':0,'stop_reason':'tasks_finished'}
            with patch.object(cli,'load_config',return_value=config(root)), \
                 patch.object(cli,'load_prices',return_value=PRICING), \
                 patch.object(cli,'_validate_api_profiles'), \
                 patch.object(cli,'doctor',return_value={'scienceworld_ready':True}), \
                 patch.object(cli,'collect_ksr_step',new_callable=AsyncMock,return_value=result) as collect:
                stdout=io.StringIO()
                with redirect_stdout(stdout):self.assertEqual(cli.main(args+['--dry-run']),0)
                collect.assert_not_called()
                plan=json.loads(stdout.getvalue())
                self.assertEqual(plan['selected_tasks'],1)
                self.assertEqual(plan['base_profile'],'weak')
                self.assertFalse(plan['root_branching'])
                self.assertFalse((Path(root)/'out').exists())
                with redirect_stdout(io.StringIO()):self.assertEqual(cli.main(args+['--execute']),0)
                collect.assert_awaited_once()
                settings=collect.call_args.args[3]
                self.assertEqual(settings.mode,'ksr_step')
                self.assertEqual(settings.rules,Rules(error_window=1,repeat_window=2,min_branch_gap=2))

    def test_until_budget_repeats_both_modes_with_unique_run_ids(self):
        for mode,budget in [('ksr',0.25),('ksr_step',0.13),('random',0.09)]:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as root:
                summary,tree,samples,requests=self.run_collect(root,mode=mode,until_budget=True,total_cost_limit=budget)
                self.assertEqual(summary['stop_reason'],'budget_exhausted')
                self.assertGreaterEqual(summary['total_api_cost'],budget)
                self.assertLess(summary['total_api_cost'],budget+0.020001)
                self.assertEqual([j['run_id'] for j in summary['jobs']],[0,1])
                self.assertIsNone(summary['planned_jobs'])
                self.assertIsNone(summary['unstarted_jobs'])

    def test_until_budget_visits_all_tasks_before_repeat(self):
        with tempfile.TemporaryDirectory() as root:
            requests=[]
            tasks=[TASK,{**TASK,'task_id':'task1'}]
            settings=Settings(mode='random',strong='strong',weak='weak',until_budget=True,total_cost_limit=0.17)
            summary=asyncio.run(collect_random(tasks,config(root),PRICING,settings,Path(root)/'out',
                env_factory=Env,provider_factory=lambda p,**kw:Provider(p,requests)))
            self.assertEqual([(j['task_id'],j['run_id']) for j in summary['jobs']],
                             [('task0',0),('task1',0),('task0',1)])

    def test_until_budget_stops_when_round_has_no_billed_progress(self):
        class Unavailable(Env):
            def load_task(self,*args):raise RuntimeError('simulator unavailable')
        with tempfile.TemporaryDirectory() as root:
            summary,tree,samples,requests=self.run_collect(root,until_budget=True,factory=Unavailable)
            self.assertEqual(summary['stop_reason'],'no_cost_progress')
            self.assertEqual(summary['started_jobs'],1)
            self.assertEqual(summary['total_api_cost'],0)

    def test_until_budget_unknown_cost_stops_repetition(self):
        class Broken:
            async def complete(self,*args):raise TimeoutError('unknown billing')
        with tempfile.TemporaryDirectory() as root:
            summary,tree,samples,requests=self.run_collect(root,until_budget=True,provider=lambda *a,**kw:Broken())
            self.assertEqual(summary['stop_reason'],'unknown_api_cost')
            self.assertEqual(summary['api_attempts'],1)
            self.assertEqual(summary['started_jobs'],1)

if __name__=='__main__':unittest.main()
