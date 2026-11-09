#!/usr/bin/env python3
"""Stage the October 2 merge; never publish or delete source runs automatically."""
import copy,json,sys,shutil,subprocess
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mixed_pilot_harness.inference.dataset import load_dataset
from mixed_pilot_harness.inference.report import InferenceCollector,export
from mixed_pilot_harness.branch_sampling.core import label_tree
B=Path(__file__).resolve().parents[1]
OLD=B/'run_0929/swe-neo-deepseek-flash-0929'
NEW=B/'run_1002/swe-neo-deepseek-flash-replacements22'
OUT=OLD.with_name(OLD.name+'.merge-staging')
AUDIT=B/'run_0929/merge-audit-1002'
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def lines(p,rows):
 with p.open('w') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def main():
 assert not (OUT/'MERGE_REPORT.json').exists()
 oldtree,newtree=read(OLD/'tree.json'),read(NEW/'tree.json')
 op,np=read(OLD/'plan.json'),read(NEW/'plan.json')
 assert not ({t['task_id'] for t in op['tasks']}&{t['task_id'] for t in np['tasks']})
 nm={n['id']:f"n{216+i:06d}" for i,n in enumerate(newtree['nodes'])}
 cm={p.stem:f"c{4975+i:08d}" for i,p in enumerate(sorted((NEW/'calls').glob('*.json')))}
 ids={**nm,**cm}
 # Preserve exact model input/output/history strings. Only structural references change.
 opaque={'state','history','initial_history','response','observation','tool_results','tool_calls','reasoning_content','messages','content','output','arguments'}
 def remap(v,key=''):
  if key in opaque:return v
  if isinstance(v,dict):return {k:remap(x,k) for k,x in v.items()}
  if isinstance(v,list):return [remap(x,key) for x in v]
  if isinstance(v,str):
   if v in ids:return ids[v]
   if v.startswith(str(NEW)):
    v=v.replace(str(NEW),str(OLD),1)
    for a,b in nm.items():v=v.replace('/'+a+'/', '/'+b+'/')
   return v
  return v
 nodes=oldtree['nodes']+[remap(n) for n in newtree['nodes']]
 jobs=oldtree['jobs']+[remap(j) for j in newtree['jobs']]
 tasks=load_dataset(Path('/Users/liangjianfeng/Data/Router/SWE-neo-data'),'train')
 assert len(tasks)==100 and {t['task_id'] for t in tasks}=={n['task_id'] for n in nodes}
 originaltasks={t['task_id']:t for t in op['tasks']+np['tasks']}
 for t in tasks:assert t['content_hashes']==originaltasks[t['task_id']]['content_hashes']
 repairs={p.stem:read(p) for p in (AUDIT/'reevaluation').glob('n*.json')}
 assert len(repairs)==6
 repairs[nm['n000021']]=read(AUDIT/'harness-submodule-fix/native-eval/result.json')
 assert all(r['status'] in {'passed','failed'} for r in repairs.values()),[(k,r['status']) for k,r in repairs.items()]
 for n in nodes:
  if n['id'] not in repairs:continue
  r=repairs[n['id']];success=r['status']=='passed';score=100. if success else 0.
  n['original_evaluation']=n['evaluation'];n['original_status']=n['status']
  n.update(status='reevaluated',labels_valid=True,final_score=score,success=success,label_value=float(success))
  n['evaluation']={'status':'scored','method':'swe_native','score':score,'success':success,'cost':0.,'cost_complete':True,'api_calls':0,'native_evaluation':r}
  n['reevaluation_evidence']=str(AUDIT/('harness-submodule-fix/native-eval' if n['id']==nm['n000021'] else 'reevaluation/'+n['id']))
 for j in jobs:j['status']='complete'
 for src,dst in nm.items():
  p=NEW/'workspaces'/src
  if p.exists():
   assert not (OUT/'workspaces'/dst).exists()
   subprocess.run(['cp','-cR',str(p),str(OUT/'workspaces'/dst)],check=True)
 for p in sorted((NEW/'calls').glob('*.json')):write(OUT/'calls'/(cm[p.stem]+'.json'),remap(read(p)))
 calls={p.stem:read(p) for p in sorted((OUT/'calls').glob('*.json'))}
 for n in nodes:
  assert all(c in calls for c in n['prefix_call_ids']+n['own_call_ids'])
  assert all(calls[c]['node_id']==n['id'] for c in n['own_call_ids'])
 samples=label_tree(nodes,calls,'single-model');lines(OUT/'samples.jsonl',samples)
 plan=copy.deepcopy(op);plan['tasks']=tasks;plan['step_output_dir']=None
 write(OUT/'plan.json',plan);write(OUT/'tree.json',{**oldtree,'nodes':nodes,'jobs':jobs})
 with (OUT/'events.jsonl').open('w') as f:
  for src in (OLD,NEW):
   for line in (src/'events.jsonl').open():
    v=json.loads(line)
    if src==NEW:v=remap(v)
    v['source_run']=str(src)
    f.write(json.dumps(v,ensure_ascii=False)+'\n')
 leaves=[n for n in nodes if not n['children'] and not n.get('superseded')]
 assert len(leaves)==100 and all(n['labels_valid'] for n in leaves)
 shutil.rmtree(OUT/'traces');shutil.rmtree(OUT/'leaf-data')
 settings=SimpleNamespace(**plan['settings']);settings.path_cost_limit=getattr(settings,'path_cost_limit',1e100)
 collector=InferenceCollector.__new__(InferenceCollector)
 collector.out=OUT;collector.tasks=tasks;collector.nodes=leaves;collector.calls=calls;collector.s=settings;collector.pricing=plan['pricing']
 export(collector)
 collector.nodes=nodes;collector.jobs=jobs;collector.total_cost=sum(c.get('cost') or 0 for c in calls.values());collector.cost_complete=all(c.get('cost') is not None for c in calls.values());collector.stop_reason='tasks_finished'
 summary=collector.summary();summary.update(samples=len(samples),valid_samples=sum(s['labels_valid'] for s in samples),planned_jobs=100,started_jobs=100,unstarted_jobs=0,complete_pairs=0,cost_by_profile={'deepseek-flash':collector.total_cost})
 write(OUT/'summary.json',summary)
 # Keep original patch/evaluation intact. These explicit pointers select repaired evidence.
 for n in leaves:
  workspace=OUT/'workspaces'/n['id']
  if n['id'] in repairs:
   write(workspace/'ACTIVE_EVALUATION.json',{'evidence':n['reevaluation_evidence'],'original_preserved':True})
   replay=sorted(workspace.glob('replay-*/prediction.patch'))[-1].parent
   (replay/'native-eval').rename(replay/'native-eval.original-before-merge')
   subprocess.run(['cp','-cR',n['reevaluation_evidence'],str(replay/'native-eval')],check=True)
 repaired=OUT/'workspaces'/nm['n000021']/'prediction-reexported.patch'
 shutil.copy2(AUDIT/'harness-submodule-fix/prediction-reexported.patch',repaired)
 originalpatch=OUT/'workspaces'/nm['n000021']/'replay-1/prediction.patch'
 originalpatch.rename(originalpatch.with_name('prediction.original-before-reexport.patch'))
 shutil.copy2(repaired,originalpatch)
 report={'status':'staged','tasks':100,'final_traces':100,'historical_nodes':len(nodes)-100,'passed':summary['passed'],'failed':100-summary['passed'],'unscored':summary['unscored_traces'],'new_api_cost':0,'known_retained_call_cost':collector.total_cost,'node_id_map':nm,'call_id_map':cm,'repaired_nodes':list(repairs),'source_runs':[str(OLD),str(NEW)],'historical_paths_note':'Model messages and tool outputs are byte-preserved and may mention original paths. Sources are archived before cleanup. Original patches remain unchanged; ACTIVE_EVALUATION.json points to reevaluation evidence.','resume_supported':False}
 (OUT/'SUMMARY.md').write_text(f"# Merged 100 tasks\n\nScored: 100; passed: {summary['passed']}; failed: {100-summary['passed']}; unscored: 0.\n\nFinal trajectories: traces/ (100 files). Historical {len(nodes)-100} attempts remain in tree/calls and pre-merge backups.\n\nKnown retained-call cost: {collector.total_cost:.8f} CNY. No new model calls. Historical missing prices remain unknown.\n\nDo not resume this merged checkpoint. See MERGE_REPORT.json for ID mappings and provenance.\n")
 assert len(list((OUT/'traces').glob('*.traj.json')))==100
 for n in leaves:
  tr=read(OUT/'traces'/(n['id']+'.traj.json'));source=NEW if n['id'] in nm.values() else OLD
  sourceid=next((k for k,v in nm.items() if v==n['id']),n['id'])
  original=read(source/'traces'/(sourceid+'.traj.json'))
  assert tr['messages']==original['messages'],n['id']
  assert tr['info']['labels_valid'] and tr['info']['task_id']==n['task_id']
  assert len(tr['steps'])==len(original['steps'])
  for a,b in zip(tr['steps'],original['steps']):
   for k in ('state','response','tool_results','reasoning_content','cost','usage'):assert a.get(k)==b.get(k),(n['id'],k)
 write(OUT/'MERGE_REPORT.json',report)
 print(json.dumps({k:v for k,v in report.items() if k not in {'call_id_map','node_id_map'}},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
