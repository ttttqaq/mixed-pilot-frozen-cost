#!/usr/bin/env python3
"""Replay recorded agent replies through the actual ScienceWorld episode adapter.

No model API calls. Step numbers are one-based; state t is BEFORE action t.
"""
from __future__ import annotations
import argparse
import asyncio
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mixed_pilot_harness.collection.episodes import ScienceWorldEpisode
from mixed_pilot_harness.collection.upstream.scienceworld import ScienceWorldEnvironment
from mixed_pilot_harness.types import Message, ProviderResponse, ToolCall


def compare(expected, actual):
    return {'equal':expected == actual, 'expected':expected, 'actual':actual}


def response_for(step):
    return ProviderResponse(text=step['response'],
                            tool_calls=[ToolCall(**x) for x in step.get('tool_calls',[])])


async def replay(data, t, probe=False, env_factory=ScienceWorldEnvironment):
    if not 1 <= t <= len(data['steps']): raise ValueError('step outside trace')
    info=data['info']; config=info['config']
    report={'step':t,'state_timing':'before action t','prefix_checks':[],
            'scope':'observable agent state; original simulator snapshot is unavailable',
            'accounting':'previous_actions/cost/budget rebuilt from recorded metadata, not re-measured'}
    episode=ScienceWorldEpisode(info,config['prompt_config'],env_factory)
    try:
        messages=episode.start();system=episode.system;spent=0.0;actions=[]
        def state():
            return {'history':[asdict(m) for m in messages],'system':system,
                    'previous_actions':list(actions),'spent_cost':spent,
                    'remaining_budget':config['settings']['cost_limit']-spent}
        report['initial_state']=compare(data['steps'][0]['state'],state())
        if not report['initial_state']['equal']:
            report['status']='initial_mismatch';return report
        for i,s in enumerate(data['steps'][:t-1],1):
            before=compare(s['state'],state())
            response=response_for(s)
            messages.append(Message('assistant',response.text,tool_calls=response.tool_calls))
            result=await episode.advance(response)
            messages.extend(result.messages)
            spent+=s['cost'];actions.append(s['selected_profile'])
            checks={'step':i,'state_before_equal':before['equal'],
                    'command':compare(s.get('command'),result.record.get('command')),
                    'observation':compare(s.get('observation'),result.record.get('observation')),
                    'format_error':compare(s.get('format_error',False),result.format_error),
                    'replayed_native_score':episode.score,'replayed_done':result.done}
            if isinstance(checks['observation']['expected'],str) and isinstance(checks['observation']['actual'],str):
                checks['observation_lines_equal_ignoring_order'] = sorted(checks['observation']['expected'].splitlines()) == sorted(checks['observation']['actual'].splitlines())
            report['prefix_checks'].append(checks)
            print(f"step {i}: command={checks['command']['equal']} observation={checks['observation']['equal']}",flush=True)
            if not before['equal'] or not all(checks[k]['equal'] for k in ['command','observation','format_error']) or result.done:
                report.update(status='prefix_mismatch',first_divergence=i,
                              original_state=s['state'],replayed_state_before=before['actual'])
                if i != t-1:
                    return report
        original=data['steps'][t-1]['state']; actual=state()
        report.update(original_state=original,replayed_state=actual,
                      state_fields_equal={k:original.get(k)==actual.get(k) for k in set(original)|set(actual)},
                      replayed_native_score=episode.score)
        report['status']='match' if original==actual else 'state_mismatch'
        if probe and report['status']=='match':
            s=data['steps'][t-1];r=await episode.advance(response_for(s))
            report['step_t_probe']={k:compare(s.get(k),r.record.get(k)) for k in ['command','observation']}
            report['step_t_probe']['replayed_done']=r.done
            report['step_t_probe']['replayed_native_score']=episode.score
        return report
    finally:
        episode.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('run/collection/baselines-scienceworld-3models-2runs/scienceworld'))
    p.add_argument('--trace',type=Path)
    p.add_argument('--step',type=int)
    p.add_argument('--seed',type=int,default=20260921)
    p.add_argument('--probe-step',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rng=random.Random(a.seed)
    # Mirror the collection wrapper's conda Java discovery.
    java_home=Path(sys.prefix)/'lib/jvm'
    if not os.environ.get('JAVA_HOME') and (java_home/'bin/java').exists():
        os.environ['JAVA_HOME']=str(java_home)
        os.environ['PATH']=str(java_home/'bin')+os.pathsep+os.environ.get('PATH','')
    candidates=sorted([*a.root.glob('deepseek-flash/*.traj.json'),*a.root.glob('qwen-flash-2025-07-28/*.traj.json')])
    path=a.trace or rng.choice(candidates)
    data=json.loads(path.read_text());t=a.step if a.step is not None else rng.randint(2,len(data['steps']))
    report={'trace':str(path.resolve()),'trace_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'seed':a.seed,'candidate_traces':len(candidates),'task_name':data['info']['task_name'],
            'model':path.parent.name,'run_id':data['info']['run_id'],'trace_steps':len(data['steps']),
            'recorded_scienceworld_version':data['info']['config']['scienceworld_version'],
            'installed_scienceworld_version':version('scienceworld'),'selected_step':t}
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    report['status']='running';a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    try:
        if report['recorded_scienceworld_version']!=report['installed_scienceworld_version']:
            raise ValueError('ScienceWorld version mismatch')
        report.update(asyncio.run(replay(data,t,a.probe_step)))
    except Exception as exc:
        report.update(status='error',error=f'{type(exc).__name__}: {exc}')
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Result:',report['status'],'report:',a.output,flush=True)
    if report['status']!='match' or any(not report.get('step_t_probe',{}).get(k,{'equal':True})['equal'] for k in ['command','observation']):
        return 1
    return 0

if __name__=='__main__':sys.exit(main())
