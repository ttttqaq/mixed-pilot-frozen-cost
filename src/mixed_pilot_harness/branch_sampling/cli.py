"""KSR trees, weak-base single-step probes, or random sampling; dry-run by default."""
from __future__ import annotations
import argparse
import asyncio
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys

from ..config import load_config
from ..collection.cli import _validate_api_profiles, doctor
from ..collection.tasks import load_tasks, select_tasks
from ..collection.pricing import load_prices
from .core import Settings, collect_ksr, collect_ksr_step, collect_random, sampling_policy
from .rules import Rules
from .selection import select_ksr_tasks


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['ksr','ksr_step','random'],required=True,
                   help='ksr: persistent model segments; ksr_step: one-step probes on a weak mainline; random: per-step routing')
    p.add_argument('--tasks-root',type=Path,default=Path('/Users/liangjianfeng/Data/scienceworld_task'))
    p.add_argument('--baseline-root',type=Path,default=Path('run/collection/baselines-scienceworld-3models-2runs/scienceworld'))
    p.add_argument('--config',type=Path,default=Path('configs/collection.providers.json'))
    p.add_argument('--pricing',type=Path,default=Path('configs/collection.prices.json'))
    p.add_argument('--strong',default='deepseek-flash');p.add_argument('--weak',default='qwen-flash-2025-07-28')
    p.add_argument('--split',choices=['all','train','dev','test'],default='all')
    p.add_argument('--task-filter',default='*');p.add_argument('--limit',type=int)
    p.add_argument('--seed',type=int,default=42)
    repeat=p.add_mutually_exclusive_group()
    repeat.add_argument('--runs',type=int,default=1)
    repeat.add_argument('--until-budget',action='store_true',help='Repeat all selected tasks in rounds until the total budget is reached')
    p.add_argument('--step-limit',type=int,default=50)
    p.add_argument('--total-cost-limit',type=float,default=10.0)
    p.add_argument('--task-cost-limit',type=float,default=1.0)
    p.add_argument('--path-cost-limit',type=float,default=0.5)
    p.add_argument('--max-nodes',type=int,default=14,help='Maximum nodes per task/run (ksr_step: one root + two per fork)')
    p.add_argument('--max-depth',type=int,default=3,help='Maximum fork depth (ksr_step: maximum forks along the weak mainline)')
    p.add_argument('--replay-attempts',type=int,default=3)
    p.add_argument('--error-window',type=int,default=2);p.add_argument('--repeat-window',type=int,default=3)
    p.add_argument('--min-branch-gap',type=int,default=3)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--step-output-dir',type=Path,help='Optional separate new directory for leaf traces and every step of each trace')
    p.add_argument('--execute',action='store_true',help='Make real model calls; without this flag, only print the plan')
    p.add_argument('--dry-run',action='store_true',help='Print the plan (also the default)')
    a=p.parse_args(argv)
    try:
        if a.execute and a.dry_run:raise ValueError('choose --execute or --dry-run')
        if a.step_output_dir is not None:
            main=a.output_dir.resolve();step=a.step_output_dir.resolve()
            if step==main or step in main.parents or main in step.parents:
                raise ValueError('--step-output-dir and --output-dir must be separate, non-nested directories')
            if step.exists():raise ValueError('--step-output-dir already exists: '+str(step))
        if a.strong==a.weak:raise ValueError('strong and weak must differ')
        for name in ['runs','step_limit','replay_attempts','error_window','repeat_window','min_branch_gap']:
            if getattr(a,name)<1:raise ValueError(name+' must be positive')
        min_nodes=1 if a.mode=='ksr_step' else 2
        if a.max_depth<0 or a.max_nodes<min_nodes:raise ValueError(f'max_depth >= 0 and max_nodes >= {min_nodes} required')
        if a.limit is not None and a.limit<1:raise ValueError('limit must be positive')
        for name in ['total_cost_limit','task_cost_limit','path_cost_limit']:
            if not math.isfinite(getattr(a,name)) or getattr(a,name)<=0:raise ValueError(name+' must be finite and positive')
        config=load_config(a.config);_validate_api_profiles(config,[a.strong,a.weak],live=a.execute)
        pricing=load_prices(a.pricing,[a.strong,a.weak])
        tasks=select_tasks(load_tasks(a.tasks_root),split=a.split,pattern=a.task_filter,seed=a.seed)
        evidence=[];source_count=len(tasks)
        if a.mode in {'ksr','ksr_step'}:tasks,evidence=select_ksr_tasks(tasks,a.baseline_root,a.strong,a.weak)
        selected_count=len(tasks)
        if a.limit:tasks=tasks[:a.limit]
        if not tasks:raise ValueError('no matching tasks')
        evidence=[x for x in evidence if x['task_id'] in {t['task_id'] for t in tasks}]
        settings=Settings(mode=a.mode,strong=a.strong,weak=a.weak,seed=a.seed,runs=a.runs,until_budget=a.until_budget,
            max_steps=a.step_limit,total_cost_limit=a.total_cost_limit,task_cost_limit=a.task_cost_limit,
            path_cost_limit=a.path_cost_limit,max_nodes=a.max_nodes,max_depth=a.max_depth,
            replay_attempts=a.replay_attempts,rules=Rules(error_window=a.error_window,
                repeat_window=a.repeat_window,min_branch_gap=a.min_branch_gap))
        plan={'dry_run':not a.execute,'source_tasks':source_count,'eligible_tasks':selected_count,
              'selected_tasks':len(tasks),'task_ids':[t['task_id'] for t in tasks],
              'settings':asdict(settings),'currency':pricing['currency'],'selection_evidence':evidence,
              'output_dir':str(a.output_dir.resolve()),
              'step_output_dir':str(a.step_output_dir.resolve()) if a.step_output_dir is not None else None,
              **sampling_policy(settings)}
        print(json.dumps(plan,ensure_ascii=False,indent=2))
        if not a.execute:return 0
        java=Path(sys.prefix)/'lib/jvm'
        if not os.environ.get('JAVA_HOME') and (java/'bin/java').exists():os.environ['JAVA_HOME']=str(java)
        if os.environ.get('JAVA_HOME'):os.environ['PATH']=os.environ['JAVA_HOME']+'/bin'+os.pathsep+os.environ.get('PATH','')
        if not doctor()['scienceworld_ready']:raise ValueError('ScienceWorld/Java unavailable; use the opensquilla environment')
        fn={'ksr':collect_ksr,'ksr_step':collect_ksr_step,'random':collect_random}[a.mode]
        summary=asyncio.run(fn(tasks,config,pricing,settings,a.output_dir,evidence=evidence,step_output_dir=a.step_output_dir))
        print(json.dumps(summary,ensure_ascii=False,indent=2))
        return 0 if (summary['cost_complete'] and not summary['failed_nodes']
                     and (not a.until_budget or summary['stop_reason']=='budget_exhausted')) else 1
    except (ValueError,OSError,ImportError) as exc:
        print('branch-sampling: '+str(exc),file=sys.stderr);return 2

if __name__=='__main__':sys.exit(main())
