"""Offline collection tests: no API credentials, Docker, or environment downloads."""
import asyncio
from contextlib import redirect_stdout
from dataclasses import replace
import io
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from mixed_pilot_harness.branch_sampling.core import Collector, Settings
from mixed_pilot_harness.inference.cli import execute, plan
from mixed_pilot_harness.step_collection.cli import parser, prepare
from mixed_pilot_harness.step_collection.ksr import validate_config, root_profiles, task_schedule, detect
from mixed_pilot_harness.step_collection.labels import samples
from mixed_pilot_harness.router.profiles import inline_config, snapshot
from mixed_pilot_harness.types import Message
from mixed_pilot_harness.collection.episodes import Transition
import test_inference as fixtures


class LongEpisode(fixtures.Episode):
    def __init__(self):self.score=0
    async def advance(self,response):
        self.steps+=1
        self.score = 100 if response.text=='a' else 0
        # Easy and difficult heuristics are independently configurable.
        output='successfully completed; command same'
        return Transition([Message('user',output)], {'observation':output,'command':response.text},
                          done=self.steps==7,status='EnvironmentDone')


class Tests(unittest.TestCase):
    setUp=fixtures.Tests.setUp

    def ksr_config(self,**overrides):
        raw=dict(strong_model='a',weak_model='b',default_start='weak',
            strong_first_tasks=['two'],weak_first_tasks=['one'],
            KSR_strong2weak=dict(signals=['easy_progress'],min_branch_gap=2,progress_window=2),
            KSR_weak2strong=dict(signals=['stalled_progress'],min_branch_gap=2,stall_window=2))
        raw.update(overrides)
        return validate_config(raw,self.config.profiles)

    def collect(self,*,strategy='ksr_branch1',**overrides):
        # fixtures' config attribute deliberately contains HarnessConfig.
        config=Tests.ksr_config(self)
        settings=Settings(mode=strategy,strong='a',weak='b',model_pool=['a','b'],max_steps=10,
            max_depth=2,max_nodes=7,total_cost_limit=100,path_cost_limit=100,task_cost_limit=100,
            stop_on_unknown_cost=False,ksr_config=config if strategy=='ksr_branch1' else {})
        if strategy=='router':
            cfg=inline_config(dict(engine='random',models=['a','b'],possibility={'a':0,'b':1}),self.config.profiles,self.root)
            settings=replace(settings,router=snapshot(cfg,'random-router'))
        settings=replace(settings,**overrides)
        self.args.collect_steps=True
        calls=[]
        with redirect_stdout(io.StringIO()):
            report=asyncio.run(execute(self.args,self.tasks,self.config,self.prices,settings,
                episode_factory=lambda *a:LongEpisode(),provider_factory=lambda p,**kw:fixtures.Provider(p,calls)))
        tree=json.loads((self.args.output_dir/'tree.json').read_text())
        rows=[json.loads(s) for s in (self.args.output_dir/'training/samples.jsonl').read_text().splitlines()]
        ledger={p.stem:json.loads(p.read_text()) for p in (self.args.output_dir/'calls').glob('*.json')}
        return report,tree,rows,ledger,calls

    def test_both_root_directions_single_step_switch_and_recursive_probe_branches(self):
        report,tree,rows,ledger,calls=self.collect()
        nodes=tree['nodes']
        self.assertEqual([n['profile'] for n in nodes if n['parent_id'] is None],['a','b'])
        for n in nodes:
            actual=[ledger[c]['selected_profile'] for c in n['own_call_ids']]
            self.assertEqual(actual,[n['profile']]+[n['base_profile']]*(len(actual)-1))
            if n['children']:
                children=[next(c for c in nodes if c['id']==x) for x in n['children']]
                self.assertEqual({c['profile'] for c in children},{'a','b'})
                self.assertEqual(children[0]['prefix_call_ids'],children[1]['prefix_call_ids'])
        self.assertTrue(any(n['node_role']=='probe' and n['children'] for n in nodes))
        self.assertTrue(any(n['node_role']=='mainline' and n['parent_id'] and n['children'] for n in nodes))
        self.assertEqual(len(rows),len(ledger))
        self.assertEqual(len({r['sample_id'] for r in rows}),len(rows))
        self.assertTrue(all(r['labels_valid'] for r in rows))
        self.assertAlmostEqual(report['total_api_cost'],len(ledger)*.02)
        self.assertTrue(all(r['remaining_cost']>=r['cost'] for r in rows))

    def test_random_uses_profile_weights_and_has_one_label_per_call(self):
        report,tree,rows,ledger,calls=self.collect(strategy='router')
        self.assertEqual(calls,['b']*14)
        self.assertEqual(len(rows),14)
        self.assertTrue(all(r['selection_probability']==1 for r in rows))
        self.assertAlmostEqual(rows[0]['remaining_cost'],.14)
        self.assertAlmostEqual(report['cost_by_profile']['b'],.28)

    def test_overlapping_lists_run_strong_group_then_weak_with_independent_trees(self):
        cfg=Tests.ksr_config(self,strong_first_tasks=['*','one'],weak_first_tasks=['*'])
        report,tree,rows,ledger,_=self.collect(ksr_config=cfg,runs=2)
        expected=[(t,r,p) for r in range(2) for p in ['a','b'] for t in ['one','two']]
        self.assertEqual([(j['task_id'],j['run_id'],j['base_profile']) for j in tree['jobs']],expected)
        self.assertEqual((report['total_tasks'],report['planned_jobs'],report['unstarted_jobs']),(2,8,0))
        nodes={n['id']:n for n in tree['nodes']}
        for n in nodes.values():
            if n['parent_id']:
                self.assertEqual(n['base_profile'],nodes[n['parent_id']]['base_profile'])
                self.assertEqual(n['branch_reason']['direction'],
                    'KSR_strong2weak' if n['base_profile']=='a' else 'KSR_weak2strong')
            else:
                self.assertEqual(n['profile'],n['base_profile'])
                self.assertEqual(n['prefix_call_ids'],[])
        self.assertEqual(len(rows),len(ledger))
        self.assertTrue(all(r['labels_valid'] for r in rows))
        self.assertEqual(len({r['sample_id'] for r in rows}),len(rows))
        self.assertAlmostEqual(sum(j['api_cost'] for j in tree['jobs']),report['total_api_cost'])

    def test_overlap_partial_resume_only_restarts_failed_direction(self):
        self.tasks=self.tasks[:1]
        cfg=Tests.ksr_config(self,strong_first_tasks=['one'],weak_first_tasks=['one'])
        _,tree,_,ledger,_=self.collect(ksr_config=cfg,max_depth=0)
        strong,weak=tree['jobs']
        strong['status']='partial'
        weak_id=weak['root_ids'][0]
        old_strong_id=strong['root_ids'][0]
        saved_weak=next(n for n in tree['nodes'] if n['id']==weak_id)
        tree_path=self.args.output_dir/'tree.json'
        tree_path.write_text(json.dumps(tree))
        self.args.resume_partial=True
        report,tree,rows,new_ledger,calls=self.collect(ksr_config=cfg,max_depth=0)
        self.assertEqual(calls,['a']*7)
        self.assertEqual(next(n for n in tree['nodes'] if n['id']==weak_id),saved_weak)
        self.assertTrue(next(n for n in tree['nodes'] if n['id']==old_strong_id)['superseded'])
        self.assertTrue(all(r['node_id']!=old_strong_id for r in rows))
        self.assertEqual(len(new_ledger),len(ledger)+7)
        self.assertEqual([j.get('attempt',0) for j in tree['jobs']],[1,0])
        self.assertAlmostEqual(tree['jobs'][0]['api_cost'],.28)
        self.assertAlmostEqual(tree['jobs'][1]['api_cost'],.14)
        self.assertAlmostEqual(report['total_api_cost'],.42)

    def test_overlap_resume_interrupted_weak_preserves_completed_strong(self):
        self.tasks=self.tasks[:1]
        cfg=Tests.ksr_config(self,strong_first_tasks=['one'],weak_first_tasks=['one'])
        original=Collector.one_call
        async def interrupt(c,*args):
            result=await original(c,*args)
            if len(c.calls)==8:raise KeyboardInterrupt()
            return result
        with patch.object(Collector,'one_call',interrupt),self.assertRaises(KeyboardInterrupt):
            self.collect(ksr_config=cfg,max_depth=0)
        self.args.resume=True
        report,tree,rows,ledger,calls=self.collect(ksr_config=cfg,max_depth=0)
        self.assertEqual(calls,['b']*6)
        self.assertEqual(len(ledger),14)
        self.assertTrue(all(j['status']=='complete' for j in tree['jobs']))
        self.assertTrue(all(r['labels_valid'] for r in rows))
        self.assertAlmostEqual(report['total_api_cost'],.28)
        self.assertTrue(all(abs(j['api_cost']-.14)<1e-9 for j in tree['jobs']))

    def test_overlap_budget_repeats_both_groups_and_legacy_resume(self):
        self.tasks=self.tasks[:1]
        cfg=Tests.ksr_config(self,strong_first_tasks=['one'],weak_first_tasks=['one'])
        report,tree,_,_,_=self.collect(ksr_config=cfg,max_depth=0,until_budget=True,total_cost_limit=.4)
        self.assertEqual([(j['run_id'],j['base_profile']) for j in tree['jobs']],[(0,'a'),(0,'b'),(1,'a')])
        self.assertEqual(report['stop_reason'],'budget_exhausted')
        self.args.output_dir=self.root/'legacy'
        _,tree,_,_,_=self.collect(max_depth=0)
        for j in tree['jobs']:j.pop('base_profile')
        (self.args.output_dir/'tree.json').write_text(json.dumps(tree))
        self.args.resume=True
        report,tree,_,_,calls=self.collect(max_depth=0)
        self.assertEqual(calls,[])
        self.assertEqual(tree['jobs'][0]['base_profile'],'b')
        self.assertAlmostEqual(report['total_api_cost'],.14)

    def test_resume_preserves_tree_and_does_not_rebill_replay(self):
        original=Collector.one_call
        async def interrupt(c,*args):
            result=await original(c,*args)
            if len(c.calls)==4:raise KeyboardInterrupt()
            return result
        with patch.object(Collector,'one_call',interrupt),self.assertRaises(KeyboardInterrupt):self.collect()
        self.args.resume=True
        report,tree,rows,ledger,calls=self.collect()
        self.assertEqual(len(calls),len(ledger)-4)
        self.assertTrue(all(j['status']=='complete' for j in tree['jobs']))
        self.assertAlmostEqual(report['total_api_cost'],len(ledger)*.02)

    def test_budget_repeats_and_partial_rows_have_no_labels(self):
        report,tree,rows,ledger,calls=self.collect(until_budget=True,total_cost_limit=.8)
        self.assertEqual(report['stop_reason'],'budget_exhausted')
        self.assertLessEqual(report['total_api_cost'],.82+1e-9)
        self.assertTrue(any(not r['labels_valid'] for r in rows))
        report,tree,rows,ledger,calls=self._fresh_random_budget()
        self.assertTrue(any(j['run_id']>0 for j in tree['jobs']))

    def _fresh_random_budget(self):
        self.args.output_dir=self.root/'repeat'
        return self.collect(strategy='router',until_budget=True,total_cost_limit=.4)

    def test_config_and_rules_validate_without_private_outcomes(self):
        cfg=Tests.ksr_config(self)
        self.assertEqual(root_profiles(cfg,{'task_id':'one'}),['b'])
        self.assertEqual(root_profiles(cfg,{'task_id':'two'}),['a'])
        self.assertEqual(root_profiles({**cfg,'strong_first_tasks':['*'],'weak_first_tasks':['one']},
                                       {'task_id':'one'}),['a','b'])
        self.assertEqual(root_profiles(cfg,{'task_id':'unlisted'}),['b'])
        self.assertEqual([(t['task_id'],p) for t,p in task_schedule(cfg,self.tasks)], [('two','a'),('one','b')])
        with self.assertRaises(ValueError):
            validate_config({**cfg,'KSR_weak2strong':{'signals':['unknown']}},self.config.profiles)
        steps=[dict(call_id=str(i),format_error=False,observation='successfully done',score_after=0,score_before=0) for i in range(2)]
        self.assertEqual(detect(cfg,'a',steps)['direction'],'KSR_strong2weak')
        self.assertIsNone(detect(cfg,'a',[{**s,'format_error':True} for s in steps]))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional MTRouter torch dependency not installed')
    def test_mtrouter_dataset_loads_collection_labels_and_cost_to_go(self):
        self.tasks[0]['benchmark']='swe'
        _,_,rows,_,_=self.collect()
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'MTRouter/src'))
        self.addCleanup(lambda:sys.path.remove(str(Path(__file__).resolve().parents[1]/'MTRouter/src')))
        from mixed_pilot_mtrouter.audit import audit_directories
        from mixed_pilot_mtrouter.research.training.dataset import TrajectoryDataset
        directory=self.args.output_dir/'training'
        with redirect_stdout(io.StringIO()):
            report=audit_directories([directory],['a','b'])
            dataset=TrajectoryDataset(directory,['a','b'])
        self.assertEqual(report['samples'],len(rows))
        self.assertEqual(len(dataset),len(rows))
        by_id={s['sample_id']:s for s in dataset.samples}
        for row in rows:
            self.assertEqual(by_id[row['sample_id']]['final_score'],row['y'])
            self.assertEqual(by_id[row['sample_id']]['remaining_cost'],row['remaining_cost'])
            self.assertEqual(by_id[row['sample_id']]['benchmark'],row['benchmark'])

    def test_cli_run_budget_and_random_profile_default(self):
        args=prepare(parser().parse_args(['--dataset','tasks','--mode','budget','--budget','2','--output-dir','out']))
        self.assertEqual((args.mode,args.stop_mode,args.router_profile),('random-router','budget-mode','random-router'))
        args=prepare(parser().parse_args(['--sampling-strategy','KSR-brunch-1','--output-dir','out']))
        self.assertEqual(args.mode,'ksr_branch1')

    def test_return_is_recursive_mean_not_leaf_count_weighting_and_missing_child_is_invalid(self):
        nodes=[];ledger={}
        for i,(parent,children,score,cost) in enumerate([
            (None,[1,2],None,.01),(0,[],100,.02),(0,[3,4],None,.03),
            (2,[],0,.04),(2,[],100,.05)]):
            cid=f'c{i}'
            nodes.append(dict(id=str(i),parent_id=str(parent) if parent is not None else None,
                children=list(map(str,children)),task_id='one',run_id=0,own_call_ids=[cid],prefix_call_ids=[],
                final_score=score,labels_valid=score is not None,base_profile='b',branch_point_id=str(parent)))
            ledger[cid]=dict(call_id=cid,step_idx=i,selected_profile='a',requested_model='actual-model',
                cost=cost,cost_currency='USD',state={'history':[]})
        task=[dict(task_id='one',benchmark='swe',split='train')]
        rows=samples(nodes,ledger,task,'ksr_branch1')
        self.assertAlmostEqual(rows[0]['y'],.75)
        self.assertAlmostEqual(rows[0]['remaining_cost'],.0575)
        self.assertEqual(rows[0]['action_model'],'a')
        ledger['c4']['cost_complete']=False
        rows=samples(nodes,ledger,task,'ksr_branch1')
        self.assertFalse(rows[0]['labels_valid'])
        self.assertIsNone(rows[0]['remaining_cost'])
        ledger['c4']['cost_complete']=True
        nodes[-1]['labels_valid']=False
        rows=samples(nodes,ledger,task,'ksr_branch1')
        self.assertFalse(rows[0]['labels_valid'])
        self.assertIsNone(rows[0]['y'])
        self.assertIsNone(rows[0]['remaining_cost'])
        self.assertTrue(rows[1]['labels_valid'])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional MTRouter torch dependency not installed')
    def test_root_collection_directory_does_not_duplicate_mtrouter_training_rows(self):
        _,_,rows,_,_=self.collect()
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'MTRouter/src'))
        self.addCleanup(lambda:sys.path.remove(str(Path(__file__).resolve().parents[1]/'MTRouter/src')))
        from mixed_pilot_mtrouter.audit import audit_directories
        from mixed_pilot_mtrouter.research.training.dataset import TrajectoryDataset
        with redirect_stdout(io.StringIO()):
            audit=audit_directories([self.args.output_dir],['a','b'])
            dataset=TrajectoryDataset(self.args.output_dir,['a','b'])
        self.assertEqual(audit['trajectories'],0)
        self.assertEqual(len(dataset),len(rows))
        self.assertEqual(audit['samples'],len(rows))

    def test_plan_reads_config_and_checks_task_assignment_before_calls(self):
        cfg=self.root/'KSR_config.json';cfg.write_text(json.dumps(self.ksr_config()))
        (self.root/'responses.json').write_text('[]')
        for p in self.config.profiles.values():p.responses_file='responses.json'
        args=prepare(parser().parse_args(['--dataset',str(self.root),'--sampling-strategy','KSR-brunch-1',
            '--ksr-config',str(cfg),'--output-dir',str(self.root/'planned'),'--dry-run']))
        with patch('mixed_pilot_harness.inference.cli.load_config',return_value=self.config), \
             patch('mixed_pilot_harness.inference.cli.load_dataset',return_value=self.tasks), \
             patch('mixed_pilot_harness.inference.cli.load_prices',return_value=self.prices):
            _,_,_,settings=plan(args)
            self.assertEqual(settings.model_pool,['a','b'])
            self.assertEqual(settings.ksr_config,self.ksr_config())
            args.stop_mode='budget-mode'
            with self.assertRaisesRegex(ValueError,'requires --budget'):plan(args)
            args.budget=1
            self.assertTrue(plan(args)[3].until_budget)
            self.assertFalse(plan(args)[3].stop_on_unknown_cost)
            args.stop_on_unknown_cost=True
            self.assertTrue(plan(args)[3].stop_on_unknown_cost)
            overlapping=self.ksr_config(strong_first_tasks=['*'])
            cfg.write_text(json.dumps(overlapping))
            tasks,_,_,settings=plan(args)
            self.assertEqual([(t['task_id'],p) for t,p in task_schedule(settings.ksr_config,tasks)],
                             [('one','a'),('two','a'),('one','b')])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional MTRouter torch dependency not installed')
    def test_mtrouter_rejects_mixed_currency_and_ignores_unknown_cost(self):
        _,_,rows,_,_=self.collect()
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'MTRouter/src'))
        self.addCleanup(lambda:sys.path.remove(str(Path(__file__).resolve().parents[1]/'MTRouter/src')))
        from mixed_pilot_mtrouter.audit import audit_directories
        from mixed_pilot_mtrouter.research.training.dataset import TrajectoryDataset
        directory=self.args.output_dir/'training'
        rows[0]['cost_complete']=False
        def save():
            (directory/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        save()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(audit_directories([directory],['a','b'])['samples'],len(rows)-1)
            self.assertEqual(len(TrajectoryDataset(directory,['a','b'])),len(rows)-1)
        rows[1]['cost_currency']='USD';save()
        with self.assertRaisesRegex(ValueError,'Mixed currencies'):
            audit_directories([directory],['a','b'])
