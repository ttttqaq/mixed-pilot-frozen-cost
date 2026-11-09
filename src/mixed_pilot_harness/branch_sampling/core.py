"""Switch-first live continuation, immutable prefixes and a unique-call ledger."""
from __future__ import annotations
import asyncio
from collections import deque
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import random
from importlib.metadata import version, PackageNotFoundError

from ..collection.collector import atomic_json, utcnow, load_prompt_config
from ..collection.episodes import ScienceWorldEpisode
from ..collection.pricing import estimate_cost
from ..collection.upstream.scienceworld import ScienceWorldEnvironment
from ..providers import create_provider
from ..types import Message, ProviderResponse, ProviderError, ToolCall, Usage
from .rules import Rules


@dataclass
class Settings:
    mode: str = 'ksr'
    strong: str = 'deepseek-flash'
    weak: str = 'qwen-flash-2025-07-28'
    seed: int = 42
    runs: int = 1
    until_budget: bool = False
    max_steps: int = 50
    total_cost_limit: float = 10.0
    task_cost_limit: float = 1.0
    path_cost_limit: float = 0.5
    max_nodes: int = 14
    max_depth: int = 3
    replay_attempts: int = 3
    retry_wait: float = 0.0
    trace_delay: float = 0.0
    rules: Rules = field(default_factory=Rules)
    model_pool: list[str] = field(default_factory=list)
    stop_on_unknown_cost: bool = True
    router: dict = field(default_factory=dict)
    ksr_config: dict = field(default_factory=dict)


def sampling_policy(settings):
    if settings.mode == 'ksr_branch1':
        return {'scheduling':'alternate_single_step_live_then_base_fifo_replay',
                'root_branching':False,'action_scope':'single_step','probe_branching':True}
    if settings.mode == 'ksr_step':
        return {'scheduling':'strong_probe_live_continuation_fifo_mainline_replay',
                'base_profile':settings.weak,'root_branching':False,
                'action_scope':'single_step','continuation_profile':settings.weak,
                'probe_branching':False}
    return {'scheduling':'switch_first_live_continuation_fifo_replay'}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def provider_response(step):
    return ProviderResponse(text=step['response'],stop_reason=step.get('stop_reason','end_turn'),
                            reasoning_content=step.get('reasoning_content'), reasoning_owner=step.get('reasoning_owner'),
                            tool_calls=[ToolCall(**c) for c in step.get('tool_calls',[])])


def label_tree(nodes, calls, mode):
    """Label legacy forks by averaging; single-step forks follow the weak spine.

    A failed probe must not invalidate an otherwise complete weak continuation.
    Legacy KSR still requires BOTH children, avoiding surviving-leaf bias.
    """
    nodes=[n for n in nodes if not n.get('superseded')]
    by_id={n['id']:n for n in nodes}
    def value(n):
        if n['status']=='split':
            children=[by_id[c] for c in n['children']]
            if mode=='ksr_step':
                children=[c for c in children if c['node_role']=='mainline']
            vals=[value(c) for c in children]
            y=sum(vals)/len(vals) if vals and all(v is not None for v in vals) else None
        else:
            y=n.get('label_value', (n['final_score']+100)/200) if n.get('labels_valid') else None
        n['y']=y
        n['return_valid']=y is not None
        if mode=='ksr_step':
            # Costs cover the sampled call and its weak-policy suffix, not probes
            # at later decision points and not the inherited prefix.
            n['remaining_api_cost']=None if y is None or any(not calls[c].get('cost_complete',True) or calls[c].get('cost') is None for c in n['own_call_ids']) or (n['status']=='split' and children[0]['remaining_api_cost'] is None) else (
                sum(calls[c]['cost'] for c in n['own_call_ids']) +
                (children[0]['remaining_api_cost'] if n['status']=='split' else 0))
        return y
    for n in nodes:
        value(n)
    result=[]
    for n in nodes:
        if mode=='ksr_step' and n['parent_id'] is None:
            continue # The weak-only initial segment is not a routing comparison.
        ids=n['own_call_ids'] if mode in {'random', 'single-model','router'} else n['own_call_ids'][:1]
        for cid in ids:
            s=calls[cid]
            result.append({'sample_id':cid,'task_id':n['task_id'],'run_id':n['run_id'],
                'node_id':n['id'],'parent_id':n['parent_id'],'branch_point_id':n['branch_point_id'],
                'step_idx':s['step_idx'],'state':s['state'],'action_profile':s['selected_profile'],
                'action_model':s['requested_model'],'response':s.get('response'),
                'observation':s.get('observation'),'cost':s.get('cost'),
                'cost_complete':s.get('cost') is not None and s.get('cost_complete',True),
                'y':n['y'],'labels_valid':n['y'] is not None,
                'label_method':'normalized_terminal_score' if mode in {'random','single-model','router'} else 'uniform_future_forks_expected_terminal_score',
                'selection_probability':s['selection_probability'],
                'branch_enumerated':mode in {'ksr','ksr_step'},
                'candidate_count': n.get('candidate_count', 2)})
            if mode=='ksr_step':
                result[-1].update(label_method='single_step_then_weak_terminal_score',
                    base_profile=n['base_profile'],continuation_profile=n['base_profile'],
                    action_scope='single_step',remaining_api_cost=n['remaining_api_cost'])
    by_point={}
    for sample in result:by_point.setdefault(sample['branch_point_id'],[]).append(sample)
    for sample in result:
        paired=by_point[sample['branch_point_id']]
        sample['paired_actions_complete']=(mode in {'ksr','ksr_step'} and len(paired)==sample['candidate_count'] and all(x['labels_valid'] for x in paired))
    return result


def export_trace_steps(output, nodes, calls):
    """Expand each leaf path; terminal labels here differ from fork-average labels.

    Shared call IDs repeat across paths, but must never be billed again.
    Interrupted/failed paths are included with null labels, not silently dropped.
    """
    leaves=[n for n in nodes if not n['children'] and not n.get('superseded')]
    copies={}
    for n in leaves:
        for cid in n['prefix_call_ids']+n['own_call_ids']:
            copies[cid]=copies.get(cid,0)+1
    count=valid_count=0
    with (output/'traces.jsonl').open('w') as traces, (output/'steps.jsonl').open('w') as steps:
        for n in leaves:
            ids=n['prefix_call_ids']+n['own_call_ids']
            valid=bool(n.get('labels_valid'))
            cost_complete=all(calls[c].get('cost') is not None and calls[c].get('cost_complete',True) for c in ids) and n.get('evaluation',{}).get('cost_complete',True)
            y=n.get('label_value', (n['final_score']+100)/200) if valid else None
            common={'trace_id':n['id'],'leaf_node_id':n['id'],'task_id':n['task_id'],
                    'base_profile':n.get('base_profile'),
                    'run_id':n['run_id'],'trace_status':n['status'],
                    'trace_complete':n['status'] in {'EnvironmentDone','Submitted'},
                    'labels_valid':valid,'cost_complete':cost_complete,'y':y,'label_method':'normalized_leaf_terminal_score'}
            traces.write(json.dumps({**common,'call_ids':ids,'steps':len(ids),
                'final_score':n['final_score'],
                'cost_currency':n.get('config_snapshot',{}).get('pricing',{}).get('currency'),
                'path_api_cost':sum(calls[c].get('cost') or 0 for c in ids)},ensure_ascii=False)+'\n')
            for cid in ids:
                s=calls[cid]
                row={**common,'sample_id':n['id']+':'+cid,'call_id':cid,
                     'node_id':s['node_id'],'step_idx':s['step_idx'],'state':s['state'],
                     'action_profile':s['selected_profile'],'action_model':s['requested_model'],
                     'response':s.get('response'),'observation':s.get('observation'),
                     'call_status':s['status'],'score_before':s['score_before'],
                     'score_after':s.get('score_after'),'done':s.get('done'),
                     'cost':s.get('cost'),'call_multiplicity':copies[cid],
                     'selection_probability':s['selection_probability']}
                steps.write(json.dumps(row,ensure_ascii=False)+'\n')
                count+=1;valid_count+=valid
    summary={'schema':'leaf-trace-steps-v1','traces':len(leaves),'steps':count,
             'valid_steps':valid_count,'unique_exported_calls':len(copies),
             'cost_note':'Repeated prefix costs are references; use construct summary for total billing.'}
    atomic_json(output/'summary.json',summary)
    return summary


class Collector:
    def __init__(self, tasks, config, pricing, settings, output, *,
                 env_factory=ScienceWorldEnvironment, provider_factory=create_provider, evidence=None,
                 step_output_dir=None, episode_factory=None, evaluator=None, resume=False, resume_partial=False,
                 pricing_profiles=None):
        if not tasks or settings.mode not in {'ksr','ksr_step','ksr_branch1','random','single-model','router'} or (not settings.model_pool and settings.strong==settings.weak):
            raise ValueError('nonempty tasks, valid mode, distinct strong/weak profiles required')
        if any(not math.isfinite(x) or x<=0 for x in (settings.total_cost_limit,settings.task_cost_limit,settings.path_cost_limit)):
            raise ValueError('budget thresholds must be finite and positive')
        if settings.max_nodes<(1 if settings.mode in {'ksr_step','ksr_branch1','single-model','random','router'} else 2) or settings.max_depth<0 or min(settings.max_steps,settings.runs,settings.replay_attempts,*asdict(settings.rules).values())<1:
            raise ValueError('invalid step/tree/replay/rule limits')
        self.pool = settings.model_pool or [settings.strong, settings.weak]
        if settings.mode=='ksr_branch1':
            from ..step_collection.ksr import validate_config
            if validate_config(settings.ksr_config,config.profiles)!=settings.ksr_config or self.pool!=[settings.ksr_config['strong_model'],settings.ksr_config['weak_model']]:
                raise ValueError('KSR settings must use a validated config and its ordered model pool')
        self.pricing_profiles = set(self.pool) | set(pricing_profiles or [])
        if any(not math.isfinite(x) or x < 0 for x in (settings.retry_wait, settings.trace_delay)):
            raise ValueError('retry wait and trace delay must be finite and nonnegative')
        for profile in self.pool:
            config.profile(profile)
            if profile not in pricing['profiles']:raise ValueError('missing pricing: '+profile)
        self.tasks=tasks;self.config=config;self.pricing=pricing;self.s=settings;self.out=Path(output)
        from ..step_collection.ksr import task_schedule
        self.schedule = (task_schedule(settings.ksr_config, tasks) if settings.mode=='ksr_branch1'
                         else [(task, None) for task in tasks])
        self.step_out=Path(step_output_dir).resolve() if step_output_dir is not None else None
        if self.step_out is not None:
            main=self.out.resolve()
            if self.step_out==main or self.step_out in main.parents or main in self.step_out.parents:
                raise ValueError('step_output_dir and output must be separate, non-nested directories')
        self.env_factory=env_factory;self.provider_factory=provider_factory
        self.episode_factory=episode_factory;self.evaluator=evaluator
        self.prompt=load_prompt_config();self.providers={};self.calls={};self.nodes=[];self.jobs=[]
        self.total_cost=0.0;self.cost_complete=True;self.abort=False
        self.stop_reason=None
        self.started=False
        self.resume=resume or resume_partial
        self.resume_partial=resume_partial
        self.evidence=evidence or []
        self.rng=random.Random(settings.seed)
        self.router_engine = None
        if settings.mode=='router' and settings.router.get('models')!=self.pool:
            raise ValueError('Router pool and inference model_pool must match')

    def event(self, event, **data):
        with (self.out/'events.jsonl').open('a') as f:
            f.write(json.dumps({'event':event,'timestamp':utcnow(),**data},ensure_ascii=False)+'\n')
            f.flush();os.fsync(f.fileno())

    def config_snapshot(self):
        return {'settings': asdict(self.s), 'pricing': self.pricing,
                'prompt_config': self.prompt,
                'profiles': {name: {key: getattr(self.config.profile(name), key)
                    for key in ('kind', 'model', 'base_url', 'endpoint', 'max_tokens',
                                'temperature', 'timeout', 'extra_body', 'context_window', 'max_input_tokens')}
                    for name in self.pool}}

    def currency_totals(self):
        totals = {}
        for c in self.calls.values():
            currency = c.get('cost_currency', self.pricing['currency'])
            totals[currency] = totals.get(currency, 0.0) + (c.get('cost') or 0)
        for n in self.nodes:
            currency = n.get('config_snapshot', {}).get('pricing', self.pricing)['currency']
            totals[currency] = totals.get(currency, 0.0) + n.get('evaluation', {}).get('cost', 0)
        return totals

    def profile_costs(self):
        """Unique agent calls plus evaluator calls; currencies never mix."""
        totals = {}
        def add(profile, currency, record, kind):
            if not profile:
                return  # Routing failed before any model was selected.
            row = totals.setdefault(profile, {}).setdefault(currency, {
                'agent_cost': 0.0, 'evaluation_cost': 0.0, 'total_known_cost': 0.0,
                'cost_complete': True, 'uncertain_calls': 0, 'api_attempts': 0})
            cost = record.get('cost') or 0.0
            complete = record.get('cost') is not None and record.get('cost_complete', True)
            row[kind] += cost
            row['total_known_cost'] += cost
            row['cost_complete'] &= complete
            row['uncertain_calls'] += not complete
            row['api_attempts'] += len(record['api_attempts']) if 'api_attempts' in record else record.get('api_calls', 1)
        for c in self.calls.values():
            add(c.get('selected_profile'), c.get('cost_currency', self.pricing['currency']), c, 'agent_cost')
        for n in self.nodes:
            e = n.get('evaluation', {})
            add(e.get('selected_profile'), n.get('config_snapshot', {}).get('pricing', self.pricing)['currency'], e, 'evaluation_cost')
        return totals

    def summary(self):
        return {'mode':self.s.mode,'total_api_cost':self.total_cost,'currency':self.pricing['currency'],
                'cost_by_currency':self.currency_totals(),
                'cost_by_profile':{p:sum(c.get('cost') or 0 for c in self.calls.values()
                    if c.get('selected_profile')==p and c.get('cost_currency',self.pricing['currency'])==self.pricing['currency'])
                    for p in dict.fromkeys([*self.pool, *[c['selected_profile'] for c in self.calls.values() if c.get('selected_profile')]])},
                'profile_costs':self.profile_costs(),
                'cost_by_profile_scope':'known agent costs in current currency; profile_costs includes evaluators and historical currencies',
                'uncertain_calls':sum(c.get('cost') is None or not c.get('cost_complete',True) for c in self.calls.values()),
                'transport_attempts':sum(len(c['api_attempts']) if 'api_attempts' in c else 1 for c in self.calls.values()),
                'total_api_cost_scope':'current currency only; see cost_by_currency for all historical costs',
                'unknown_cost_policy':'stop' if self.s.stop_on_unknown_cost else 'continue_known_costs_only',
                'until_budget':self.s.until_budget,'stop_reason':self.stop_reason,
                'total_tasks':len(self.tasks),
                'planned_jobs':None if self.s.until_budget else len(self.schedule)*self.s.runs,
                'started_jobs':len(self.jobs),
                'unstarted_jobs':None if self.s.until_budget else len(self.schedule)*self.s.runs-len(self.jobs),
                'cost_basis':'token_estimate','cost_complete':self.cost_complete and all(
                    c.get('cost') is not None and c.get('cost_complete',True) for c in self.calls.values()),
                'unpriced_calls':sum(s.get('cost') is None for s in self.calls.values()),
                'api_attempts':len(self.calls),'api_responses':sum('response' in s for s in self.calls.values()),
                'nodes':len(self.nodes),'jobs':[{k:v for k,v in j.items() if k!='initial_history'} for j in self.jobs],
                'failed_nodes':sum(not n.get('superseded') and n['status'] in {'initialization_error','replay_failed','environment_error','api_or_accounting_error','provider_error'} for n in self.nodes),
                'superseded_nodes':sum(bool(n.get('superseded')) for n in self.nodes),
                'total_cost_limit':self.s.total_cost_limit,
                'budget_semantics':'stop before next call when threshold reached; one in-flight call may overshoot'}

    def save(self):
        atomic_json(self.out/'tree.json',{'nodes':self.nodes,'jobs':self.jobs,'rng_state':self.rng.getstate()})
        atomic_json(self.out/'summary.json',self.summary())

    def new_node(self, task, run, profile, parent, prefix, depth, reason, point):
        n={'id':f'n{len(self.nodes):06d}','task_id':task['task_id'],'run_id':run,
           'profile':profile,'parent_id':parent,'prefix_call_ids':list(prefix),'own_call_ids':[],
            'depth':depth,'branch_reason':reason,'branch_point_id':point,'children':[],
           'status':'pending','labels_valid':False,'final_score':None,'replay':[],
            'state_source':'replay' if prefix else 'initialization', 'candidate_count':len(self.pool)}
        n['config_snapshot'] = self.config_snapshot()
        if self.s.mode=='ksr_branch1':
            base = next(p['base_profile'] for p in self.nodes if p['id']==parent) if parent else profile
            n.update(base_profile=base, node_role='probe' if profile!=base else 'mainline')
        n['attempt']=next((j.get('attempt',0) for j in self.jobs
                          if self.matches_job(j, task, run, n.get('base_profile'))),0)
        if self.s.mode=='ksr_step':
            n.update(base_profile=self.s.weak,
                     node_role='probe' if profile==self.s.strong else 'mainline')
        self.nodes.append(n);return n

    async def restore(self, task, prefix, initial, node):
        """Exact observations/native scores only; ordering mismatches retry, never normalize."""
        previous=max((r['attempt'] for r in node['replay']),default=0)
        for attempt in range(previous+1,previous+self.s.replay_attempts+1):
            if attempt > previous + 1 and self.s.retry_wait:
                await asyncio.sleep(self.s.retry_wait)
            ep=None
            stage='initialization'
            try:
                ep=(self.episode_factory(task, node, attempt) if self.episode_factory else
                    ScienceWorldEpisode(task,self.prompt,self.env_factory))
                history=ep.start()
                stage='replay'
                expected=[asdict(Message(**m)) for m in initial] if initial is not None else None
                if expected is not None and [asdict(m) for m in history]!=expected:
                    raise ValueError('initial_observation_mismatch')
                for cid in prefix:
                    s=self.calls[cid];r=provider_response(s)
                    history.append(r.assistant_message())
                    transition=await ep.advance(r);history.extend(transition.messages)
                    if (transition.record!=s['transition_record'] or transition.format_error!=s['format_error']
                        or ep.score!=s['score_after'] or transition.done!=s['done']):
                        raise ValueError('prefix_mismatch:'+cid)
                    if transition.done and not (self.resume and cid==prefix[-1] and cid in node['own_call_ids']):
                        raise ValueError('prefix_already_terminal')
                node['replay'].append({'attempt':attempt,'matched':True,'prefix_steps':len(prefix)})
                return ep,history
            except Exception as exc:
                if ep is not None:ep.close()
                node['replay'].append({'attempt':attempt,'matched':False,'stage':stage,'error':str(exc)})
        return None,None

    def budget_status(self, job, prefix):
        if self.abort:return 'collection_aborted'
        if self.total_cost>=self.s.total_cost_limit:return 'global_budget'
        if job['api_cost']>=self.s.task_cost_limit:return 'task_budget'
        if sum(self.calls[cid]['cost'] for cid in prefix)>=self.s.path_cost_limit:return 'path_budget'
        if len(prefix)>=self.s.max_steps:return 'step_limit'
        return None

    async def one_call(self, ep, history, profile, prefix, node, job):
        profile = profile or self.pool[0]
        cid=f'c{len(self.calls):08d}';p=self.config.profile(profile)
        spent=sum(self.calls[i]['cost'] for i in prefix)
        s={'call_id':cid,'node_id':node['id'],'step_idx':len(prefix),
           'selected_profile':profile,'requested_model':p.model,'started_at':utcnow(),
           'selection_probability':1/len(self.pool) if self.s.mode=='random' else 1.0 if self.s.mode=='single-model' else None,
           'state':{'history':[asdict(m) for m in history],'system':ep.system,
                    'previous_actions':[self.calls[i]['selected_profile'] for i in prefix],
                    'spent_cost':spent,'remaining_budget':self.s.path_cost_limit-spent,
                    'tools':[asdict(t) for t in ep.tools]},
           'cost_currency':self.pricing['currency'],
           'pricing_snapshot':self.pricing['profiles'][profile],
           'provider_snapshot':self.config_snapshot()['profiles'][profile],
           'score_before':ep.score,'cost':None,'status':'in_flight'}
        self.calls[cid]=s;atomic_json(self.out/'calls'/f'{cid}.json',s)
        api_started=False
        try:
            if self.s.mode=='router':
                from ..router.input_adapter import build_router_request
                from ..router.types import RouterDecision
                from ..router.provider import route_request
                s.update(selected_profile=None,requested_model=None,pricing_snapshot={},provider_snapshot={})
                previous=[RouterDecision(**self.calls[c]['router_decision']) for c in prefix
                          if self.calls[c].get('router_decision')][-5:]
                request=build_router_request(history,ep.tools,ep.system,previous)
                try:
                    decision=await route_request(self.router_engine,request)
                    profile=decision.tier
                    if profile not in self.pool:
                        raise ValueError('Router selected a profile outside the configured pool')
                except Exception as exc:
                    raise ProviderError(f'Router selection failed: {exc}',cost_known_zero=True) from exc
                p=self.config.profile(profile)
                s.update(selected_profile=profile,requested_model=p.model,
                    pricing_snapshot=self.pricing['profiles'][profile],
                    provider_snapshot=self.config_snapshot()['profiles'][profile],
                    selection_probability=decision.metadata.get('selection_probability'),
                    router_profile=self.s.router['profile'],router_decision=asdict(decision))
                self.event('router_decision',call_id=cid,profile=profile,decision=asdict(decision))
            atomic_json(self.out/'calls'/f'{cid}.json',s)
            # Persist selection and RNG before the selected provider is called.
            self.event('api_call_started',call_id=cid,profile=profile);self.save()
            if profile not in self.providers:
                self.providers[profile]=self.provider_factory(p,config_dir=self.config.path.parent,profiles=self.config.profiles)
            api_started=True
            response=await self.providers[profile].complete(history,ep.tools,ep.system)
            s.update(response=response.text,stop_reason=response.stop_reason,tool_calls=[asdict(c) for c in response.tool_calls],usage=asdict(response.usage),returned_model=response.model,
                     reasoning_content=response.reasoning_content,reasoning_owner=response.reasoning_owner,context_management=response.context_management,
                     cost_complete=response.cost_complete,api_attempts=response.api_attempts)
            # Persist response before price calculation/environment action.
            atomic_json(self.out/'calls'/f'{cid}.json',s)
            cost,details=estimate_cost(response.usage,self.pricing['profiles'][profile],s['started_at'])
            s.update(cost=cost,cost_details=details)
            self.total_cost+=cost;job['api_cost']+=cost
            if not response.cost_complete:
                self.cost_complete=False
                self.abort=self.s.stop_on_unknown_cost
            self.event('api_call_cost',call_id=cid,profile=profile,cost=cost,total_api_cost=self.total_cost,cost_complete=response.cost_complete,profile_costs=self.profile_costs())
            self.save()
        except BaseException as exc:
            known_zero=not api_started or getattr(exc,'cost_known_zero',False)
            status='provider_error' if known_zero else 'api_or_accounting_error'
            s.update(status=status,error=f'{type(exc).__name__}: {exc}',cost=0.0 if known_zero else s.get('cost'))
            s['cost_complete']=known_zero
            s['api_attempts']=getattr(exc,'api_attempts',s.get('api_attempts',[]))
            s['context_management']=getattr(exc,'context_management',s.get('context_management',{}))
            if not known_zero:
                self.cost_complete=False
                self.abort=self.s.stop_on_unknown_cost
            node['status']=status
            atomic_json(self.out/'calls'/f'{cid}.json',s);self.save()
            if not isinstance(exc,Exception):raise
            return None
        # Once billed, errors below must never lose this call's cost.
        node['own_call_ids'].append(cid)
        history.append(response.assistant_message())
        try:
            result=await ep.advance(response);history.extend(result.messages)
            s.update(result.record);s.update(transition_record=result.record,format_error=result.format_error,
                done=result.done,transition_status=result.status,score_after=ep.score,status='completed')
            if response.tool_calls:
                s.setdefault('command',json.dumps([{'name':c.name,'arguments':c.arguments} for c in response.tool_calls],sort_keys=True))
                s.setdefault('observation','\n'.join(m.content for m in result.messages))
        except Exception as exc:
            s.update(status='environment_error',error=str(exc));node['status']='environment_error'
            atomic_json(self.out/'calls'/f'{cid}.json',s);self.save();return None
        atomic_json(self.out/'calls'/f'{cid}.json',s);self.save();return s

    def matches_job(self, record, task, run, base_profile=None):
        return (record['task_id']==task['task_id'] and record['run_id']==run
                and (self.s.mode!='ksr_branch1' or record.get('base_profile')==base_profile))

    async def run_job(self, task, run, base_profile=None):
        if self.s.mode=='ksr_branch1' and base_profile not in self.pool:
            raise ValueError('KSR job requires a starting profile from its model pool')
        job=next((j for j in self.jobs if self.matches_job(j,task,run,base_profile)),None)
        restart=bool(job is not None and job['status']=='partial' and self.resume_partial)
        if restart:
            if self.budget_status(job,[]):
                return
            job.setdefault('previous_attempts',[]).append({k:v for k,v in job.items() if k!='previous_attempts'})
            for n in self.nodes:
                if self.matches_job(n,task,run,base_profile):
                    n['superseded']=True
            # Costs remain cumulative across attempts; only the environment and
            # trajectory start over. IDs keep growing so old artifacts survive.
            job.update(status='running',root_ids=[],attempt=job.get('attempt',0)+1)
            job.pop('initial_history',None)
        if job is not None and job['status']!='running':
            return
        existing=job is not None and not restart
        if job is None:
            job={'task_id':task['task_id'],'run_id':run,'api_cost':0.0,'api_cost_currency':self.pricing['currency'],'root_ids':[],'status':'running'}
            if self.s.mode=='ksr_branch1':job['base_profile']=base_profile
            self.jobs.append(job)
        print(f"{self.s.mode}: {task['task_id']} run={run}" + (f" base={base_profile}" if base_profile else ''),flush=True)
        from ..step_collection.ksr import detect as detect_ksr
        models=(self.pool if self.s.mode=='ksr' else
                [self.pool[0]] if self.s.mode=='single-model' else
                [self.s.weak] if self.s.mode=='ksr_step' else
                [base_profile] if self.s.mode=='ksr_branch1' else [None])
        queue=deque();jobnodes=[];initial=None
        if existing:
            jobnodes=[n for n in self.nodes if self.matches_job(n,task,run,base_profile) and not n.get('superseded')]
            queue.extend(sorted((n for n in jobnodes if n['status'] in {'running','pending'} or
                                 (self.evaluator and n['status'] in {'Submitted','EnvironmentDone','step_limit','path_budget','EmptyResponseLimit'} and 'evaluation' not in n)),
                                key=lambda n:n['status']!='running'))
            initial=job.get('initial_history')
        for profile in ([] if existing else models):
            point=None if self.s.mode in {'ksr_step','ksr_branch1'} else f'{task["task_id"]}:{run}:root'
            reason={'rule':'initialization' if self.s.mode in {'ksr_step','ksr_branch1'} else 'root'}
            n=self.new_node(task,run,profile,None,[],0,reason,point)
            queue.append(n);jobnodes.append(n);job['root_ids'].append(n['id'])
        self.save()
        while queue:
            n=queue.popleft();prefix=list(n['prefix_call_ids'])
            budget=self.budget_status(job,prefix)
            if budget:n['status']=budget;self.save();continue
            ep,history=await self.restore(task,prefix+n['own_call_ids'],initial,n)
            if ep is None:
                n['status']='initialization_error' if all(r.get('stage')=='initialization' for r in n['replay'][-self.s.replay_attempts:]) else 'replay_failed'
                self.save();continue
            if initial is None:initial=[asdict(m) for m in history];job['initial_history']=initial
            n['status']='running'
            try:
                while True:
                    path=prefix+n['own_call_ids'];budget=self.budget_status(job,path)
                    if n['own_call_ids'] and self.calls[n['own_call_ids'][-1]].get('done'):
                        last=self.calls[n['own_call_ids'][-1]]
                        n.update(status=last['transition_status'],labels_valid=True,final_score=ep.score)
                        break
                    if budget:
                        n['status']=budget
                        if budget in {'path_budget','step_limit'}:
                            n.update(labels_valid=True,final_score=ep.score)
                        break
                    own=[self.calls[i] for i in n['own_call_ids']]
                    may_branch=(self.s.mode=='ksr' or
                                (self.s.mode=='ksr_step' and n['node_role']=='mainline'))
                    reason=(detect_ksr(self.s.ksr_config,n['base_profile'],own)
                            if self.s.mode=='ksr_branch1' else self.s.rules.detect(own) if may_branch else None)
                    if reason and n['depth']<self.s.max_depth and len(jobnodes)+len(self.pool)<=self.s.max_nodes:
                        n.update(status='split',split_before_step=len(path)+1,split_reason=reason)
                        point=f'{n["id"]}:step{len(path)+1}'
                        continued=None
                        for profile in self.pool:
                            child=self.new_node(task,run,profile,n['id'],path,n['depth']+1,reason,point)
                            n['children'].append(child['id']);jobnodes.append(child)
                            if profile==(n['base_profile'] if self.s.mode=='ksr_branch1' else n['profile']) or continued is not None:
                                queue.append(child)
                            else:
                                child['state_source']='live_continuation'
                                continued=child
                        # The switched child owns this exact live episode/history.
                        # The unchanged sibling retains an immutable prefix for later replay.
                        self.save()
                        print(f"  {n['id']} {n['profile']} split -> live {continued['id']}",flush=True)
                        n=continued;prefix=list(n['prefix_call_ids']);n['status']='running'
                        continue
                    if reason:
                        n.setdefault('suppressed_forks',[]).append({'step':len(path)+1,'reason':reason,'cause':'tree_limit'})
                    if self.s.mode=='random':
                        profile=self.rng.choice(self.pool)
                    elif self.s.mode=='ksr_branch1':
                        profile=n['base_profile'] if own else n['profile']
                    elif self.s.mode=='ksr_step' and own:
                        profile=self.s.weak # Only the first call of a probe uses strong.
                    else:
                        profile=n['profile']
                    s=await self.one_call(ep,history,profile,path,n,job)
                    if s is None:break
                    recent=[self.calls[c] for c in prefix+n['own_call_ids']][-3:]
                    if len(recent)==3 and all(not c.get('response','').strip() and not c.get('tool_calls') for c in recent):
                        n.update(status='EmptyResponseLimit',labels_valid=True,final_score=0.0,success=False)
                        break
                    if s['done']:
                        n.update(status=s['transition_status'],labels_valid=True,final_score=ep.score);break
            finally:
                try:
                    if self.evaluator and n['status'] not in {'split','running','api_or_accounting_error','provider_error','environment_error'}:
                        await self.evaluator(self, ep, n, job)
                finally:
                    ep.close();self.save()
                print(f"  {n['id']} {n['profile'] or self.s.mode} {n['status']} calls={len(n['own_call_ids'])} total_cost={self.total_cost:.6f}",flush=True)
            if n['labels_valid'] and not n['children'] and self.s.trace_delay:
                await asyncio.sleep(self.s.trace_delay)
        complete={'split','EnvironmentDone','Submitted','path_budget','step_limit','EmptyResponseLimit'}
        job['status']='complete' if all(n['status'] in complete for n in jobnodes) else 'partial'
        self.save()

    async def run(self):
        # Exclusive output creation prevents accidental reruns/spending and mixed configs.
        if self.resume:
            self.load_checkpoint()
        elif self.out.exists():raise ValueError('output already exists: '+str(self.out))
        if self.s.mode=='router':
            from ..router.profiles import from_snapshot, snapshot
            from ..router.provider import _build_engine
            from ..router.random import RandomRouterEngine
            cfg=from_snapshot(self.s.router,self.config.profiles)
            if snapshot(cfg,self.s.router['profile'])!=self.s.router:
                raise ValueError('Router checkpoint/config changed since planning')
            try:
                self.router_engine=(RandomRouterEngine(cfg,self.rng) if cfg.engine=='random' else _build_engine(cfg))
            except Exception as exc:
                raise ValueError(f'Router initialization failed: {exc}') from exc
        if self.step_out is not None:
            self.step_out.mkdir(parents=True,exist_ok=self.resume)
        if not self.resume:
            self.out.mkdir(parents=True,exist_ok=False);(self.out/'calls').mkdir()
        self.started=True
        try: env_version=version('scienceworld')
        except PackageNotFoundError: env_version='not-installed'
        if not self.resume:
            atomic_json(self.out/'plan.json',{'settings':asdict(self.s),'tasks':self.tasks,
            'selection_evidence':self.evidence,'pricing':self.pricing,'prompt_config':self.prompt,
            'schema':'branch-sampling-v3' if self.s.mode=='ksr_step' else 'branch-sampling-v2',
            'scienceworld_version':env_version, 'state_equivalence':'exact_observations_and_native_scores_only',
            **sampling_policy(self.s),
            'step_output_dir':str(self.step_out) if self.step_out is not None else None,
            'profiles':{n:{k:getattr(self.config.profile(n),k) for k in ('kind','model','base_url','endpoint','max_tokens','temperature','timeout','extra_body','context_window','max_input_tokens')} for n in self.pool}})
        self.save()
        try:
            if self.s.until_budget:
                run=0
                while not self.abort and self.total_cost<self.s.total_cost_limit:
                    before=self.total_cost
                    pending=any(not any(self.matches_job(j,t,run,base) and j['status']!='running'
                                        and not (self.resume_partial and j['status']=='partial')
                                        for j in self.jobs) for t,base in self.schedule)
                    # Cover every selected task before starting another repeat.
                    for task,base in self.schedule:
                        if self.abort or self.total_cost>=self.s.total_cost_limit:break
                        await self.run_job(task,run,**({'base_profile':base} if base is not None else {}))
                    if not self.abort and self.total_cost==before and pending:
                        self.stop_reason='no_cost_progress'
                        break
                    run+=1
            else:
                for run in range(self.s.runs):
                    for task,base in self.schedule:
                        if self.abort or self.total_cost>=self.s.total_cost_limit:break
                        await self.run_job(task,run,**({'base_profile':base} if base is not None else {}))
            if self.abort:self.stop_reason='unknown_api_cost'
            elif self.total_cost>=self.s.total_cost_limit:self.stop_reason='budget_exhausted'
            elif self.stop_reason is None:self.stop_reason='tasks_finished'
        finally:
            if self.stop_reason is None:self.stop_reason='interrupted'
            if self.s.mode=='ksr_branch1':
                from ..step_collection.labels import samples as step_labels
                samples=step_labels(self.nodes,self.calls,self.tasks,self.s.mode)
            else:
                samples=label_tree(self.nodes,self.calls,self.s.mode)
            with (self.out/'samples.jsonl').open('w') as f:
                for s in samples:f.write(json.dumps(s,ensure_ascii=False)+'\n')
            self.save()
            report=self.summary();report.update(samples=len(samples),valid_samples=sum(s['labels_valid'] for s in samples),
                complete_pairs=len({s['branch_point_id'] for s in samples if s.get('paired_actions_complete')}))
            if self.step_out is not None:
                report['trace_step_export']={'output_dir':str(self.step_out),
                    **export_trace_steps(self.step_out,self.nodes,self.calls)}
            atomic_json(self.out/'summary.json',report)
        return report

    def load_checkpoint(self):
        saved=json.loads((self.out/'plan.json').read_text())
        current=asdict(self.s)
        # Recovery/timing controls may change without invalidating saved actions,
        # labels or billing. Keep all sampling and budget checks strict.
        recovery_keys={'replay_attempts','retry_wait','trace_delay','stop_on_unknown_cost'}
        if self.resume_partial:
            recovery_keys.add('max_steps')
        tasks_changed = saved['tasks'] != self.tasks
        if tasks_changed and self.resume_partial:
            old_tasks = {t['task_id']:t for t in saved['tasks']}
            new_tasks = {t['task_id']:t for t in self.tasks}
            if len(old_tasks)!=len(saved['tasks']) or len(new_tasks)!=len(self.tasks):
                raise ValueError('Resume requires unique task IDs')
            if any(new_tasks.get(task_id)!=task for task_id,task in old_tasks.items()):
                raise ValueError('Partial resume only allows adding tasks; existing tasks must remain unchanged')
        elif tasks_changed:
            raise ValueError('Resume requires the original tasks/settings; use --resume-partial to add tasks')
        if any(current.get(k)!=v for k,v in saved['settings'].items() if k not in recovery_keys) or saved['settings'].get('router',{})!=self.s.router or saved['settings'].get('ksr_config',{})!=self.s.ksr_config:
            raise ValueError('Resume requires the original tasks/settings')
        if saved['prompt_config']!=self.prompt:
            raise ValueError('Resume prompt configuration changed')
        previous_path = self.out / 'resume-config.json'
        previous = json.loads(previous_path.read_text()) if previous_path.exists() else saved
        step_limit_changed = self.resume_partial and previous['settings']['max_steps'] != self.s.max_steps
        previous_profiles = {name: {'max_input_tokens': None, **p} for name, p in previous['profiles'].items()}
        profile_changed = previous_profiles != self.config_snapshot()['profiles']
        currency_changed = previous['pricing']['currency'] != self.pricing['currency']
        if (profile_changed or currency_changed) and not self.resume_partial:
            raise ValueError('Provider/currency changes require --resume-partial to restart unfinished tasks')
        if currency_changed and (self.s.until_budget or min(self.s.total_cost_limit,
                self.s.task_cost_limit, self.s.path_cost_limit) < 1e100):
            raise ValueError('Cannot change currency in a budgeted run without an explicit exchange rate')
        tree=json.loads((self.out/'tree.json').read_text())
        self.nodes,self.jobs=tree['nodes'],tree['jobs']
        if self.s.mode=='ksr_branch1':
            # Older checkpoints had one starting profile per task and omitted
            # it from jobs. Nodes already recorded the unambiguous base profile.
            from ..step_collection.ksr import root_profiles
            task_map={t['task_id']:t for t in self.tasks}
            for j in self.jobs:
                if 'base_profile' not in j:
                    bases={n['base_profile'] for n in self.nodes
                           if n['task_id']==j['task_id'] and n['run_id']==j['run_id']}
                    if not bases:bases=set(root_profiles(self.s.ksr_config,task_map[j['task_id']]))
                    if len(bases)!=1:raise ValueError('Ambiguous starting profile in legacy KSR checkpoint')
                    j['base_profile']=bases.pop()
        self.calls={p.stem:json.loads(p.read_text()) for p in sorted((self.out/'calls').glob('*.json'))}
        if list(self.calls)!=[f'c{i:08d}' for i in range(len(self.calls))]:
            raise ValueError('Incomplete call ledger: refusing to reuse call IDs')
        legacy_snapshot = {k: saved[k] for k in ('settings', 'profiles', 'pricing', 'prompt_config')}
        for n in self.nodes:
            n.setdefault('config_snapshot', legacy_snapshot)
        for c in self.calls.values():
            c.setdefault('cost_currency', saved['pricing']['currency'])
            c.setdefault('pricing_snapshot', saved['pricing']['profiles'].get(c['selected_profile'],{}))
            c.setdefault('provider_snapshot', saved['profiles'].get(c['selected_profile'],{}))
            if c.get('cost') is None and 'response' in c and 'usage' in c:
                # A response may have been persisted immediately before a crash
                # in accounting. Reprice its saved usage without a second API call.
                try:
                    cost,details=estimate_cost(Usage(**c['usage']),c['pricing_snapshot'],c['started_at'])
                except (ValueError,TypeError,KeyError):
                    continue
                c.update(cost=cost,cost_details=details,cost_recovered_from_saved_usage=True)
                atomic_json(self.out/'calls'/(c['call_id']+'.json'),c)
        if 'rng_state' in tree:
            def tuples(x): return tuple(map(tuples,x)) if isinstance(x,list) else x
            self.rng.setstate(tuples(tree['rng_state']))
        elif self.s.mode=='random':
            for _ in self.calls:self.rng.choice(self.pool)
        elif self.s.mode=='router' and self.s.router['engine']=='random':
            raise ValueError('Random router resume requires saved RNG state')
        self.total_cost=self.currency_totals().get(self.pricing['currency'], 0.0)
        self.cost_complete=all(c.get('cost') is not None and c.get('cost_complete',True) for c in self.calls.values()) and all(
            n.get('evaluation',{}).get('cost_complete',True) for n in self.nodes)
        if not self.cost_complete and self.s.stop_on_unknown_cost:
            raise ValueError('Cannot resume a budgeted run with unknown API cost; reconcile the ledger first')
        for n in self.nodes:
            owned=[c for c in self.calls.values() if c['node_id']==n['id']]
            # A crash can occur between persisting the call and persisting the tree.
            for c in owned:
                if c['status']=='completed' and c['call_id'] not in n['own_call_ids']:
                    n['own_call_ids'].append(c['call_id'])
            if n['status']=='running' and any(c['status']!='completed' for c in owned):
                n.update(status='interrupted_call',labels_valid=False,final_score=None)
        for j in self.jobs:
            j['api_cost_currency']=self.pricing['currency']
            j['api_cost']=sum(c.get('cost') or 0 for c in self.calls.values()
                if c['cost_currency']==self.pricing['currency'] and any(n['id']==c['node_id'] and self.matches_job(n,j,j['run_id'],j.get('base_profile')) for n in self.nodes)) + sum(
                n.get('evaluation',{}).get('cost',0) for n in self.nodes if self.matches_job(n,j,j['run_id'],j.get('base_profile')) and n['config_snapshot']['pricing']['currency']==self.pricing['currency'])
        if profile_changed or currency_changed or step_limit_changed:
            for j in self.jobs:
                if j['status'] == 'running':
                    j['status'] = 'partial'
        if tasks_changed:
            # Persist before starting new jobs so interruptions and later
            # resumes/reevaluation retain the complete task inventory.
            saved.setdefault('task_extensions', []).append({
                'timestamp':utcnow(), 'previous_total':len(saved['tasks']),
                'total_tasks':len(self.tasks),
                'added_task_ids':[t['task_id'] for t in self.tasks if t['task_id'] not in old_tasks]})
            saved['tasks'] = self.tasks
            atomic_json(self.out/'plan.json', saved)
        atomic_json(previous_path, self.config_snapshot())
        self.event('resume_configuration', pricing=self.pricing,
                   profiles=self.config_snapshot()['profiles'])
        self.event('resume_recovery_settings',
                   settings={k:current[k] for k in sorted(recovery_keys)})


async def collect_ksr(tasks,config,pricing,settings,output,**kwargs):
    if settings.mode!='ksr':raise ValueError('collect_ksr requires mode=ksr')
    return await Collector(tasks,config,pricing,settings,output,**kwargs).run()


async def collect_ksr_step(tasks,config,pricing,settings,output,**kwargs):
    """Weak mainline with one strong call per non-recursive counterfactual probe."""
    if settings.mode!='ksr_step':raise ValueError('collect_ksr_step requires mode=ksr_step')
    return await Collector(tasks,config,pricing,settings,output,**kwargs).run()


async def collect_random(tasks,config,pricing,settings,output,**kwargs):
    if settings.mode!='random':raise ValueError('collect_random requires mode=random')
    return await Collector(tasks,config,pricing,settings,output,**kwargs).run()
