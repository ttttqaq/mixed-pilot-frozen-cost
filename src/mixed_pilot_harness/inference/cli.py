"""One configurable dataset inference entry point; no experiment-specific model defaults."""
from __future__ import annotations
import argparse
import asyncio
import fcntl
import json
import math
import os
from pathlib import Path
import sys
from urllib.parse import urlparse

from ..branch_sampling.core import Settings
from ..branch_sampling.rules import Rules
from ..collection.cli import doctor
from ..collection.pricing import load_prices
from ..config import load_config
from ..providers import HTTP_RETRY_POLICY
from .dataset import load_dataset
from .episodes import factory, evaluator
from .report import InferenceCollector, export


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset', type=Path, help='Dataset root: train.json/test.json or tasks/*/manifest.json')
    p.add_argument('--split', choices=['train','test','all','train-test'], default='train', help='all/train-test means train + test, excludes dev')
    p.add_argument('--mode', choices=['single-model','random','ksr','ksr-branch','router','mtrouter','random-router'])
    p.add_argument('--models', nargs='+', help='Exact profile names in harness.json')
    p.add_argument('--router-profile', help='Router profile in harness.json; --models optionally overrides its pool')
    p.add_argument('--router-possibility', nargs='+', metavar='PROFILE=WEIGHT',
                   help='RandomRouter relative weights for every pool profile; default uniform')
    runtime = p.add_mutually_exclusive_group()
    runtime.add_argument('--router-checkpoint', type=Path, help='Local MTRouter checkpoint; overrides profile service_url')
    runtime.add_argument('--router-service-url', help='External MTRouter scoring service; overrides local checkpoint')
    p.add_argument('--router-device', choices=['cpu','cuda','mps'])
    p.add_argument('--router-model-names', nargs='+', help='Checkpoint model names, in the same order as pool profiles')
    p.add_argument('--router-embedding-url', help='Override embedding endpoint for a vLLM checkpoint')
    p.add_argument('--router-model-metadata', type=Path, help='Relocated checkpoint model metadata YAML')
    p.add_argument('--stop-mode', choices=['run-mode','budget-mode','budget-model'], default='run-mode')
    p.add_argument('--runs', type=int, default=1, help='Dataset traversal rounds (run-mode only)')
    p.add_argument('--budget', type=float, help='Total cost threshold in pricing currency; required in budget-mode')
    p.add_argument('--stop-on-unknown-cost', action='store_true',
                   help='Stop on uncertain API charges; default continues using known costs only (unknown portion contributes 0)')
    p.add_argument('--output-dir', type=Path, required=True, help='New output directory; refuses overwrite')
    p.add_argument('--config', type=Path, default=Path('harness.json'))
    p.add_argument('--pricing', type=Path, default=Path('configs/inference.prices.json'))
    p.add_argument('--step-limit', type=int, default=50)
    p.add_argument('--trace-budget', type=float, help='Optional per-path cost threshold')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--limit', type=int, help='Optional task subset for debugging')
    p.add_argument('--max-nodes', type=int, default=14, help='KSR nodes per task/round, including roots')
    p.add_argument('--max-depth', type=int, default=3)
    retry = p.add_mutually_exclusive_group()
    retry.add_argument('--replay-attempts', type=int, default=3, help='Legacy recovery attempt count, including first attempt')
    retry.add_argument('--max-retries', type=int, help='Extra retries for initialization/replay and transient API errors; 0 disables retries (default: 2)')
    p.add_argument('--retry-wait', type=float, default=5.0, help='Seconds between retries')
    p.add_argument('--error-window', type=int, default=2)
    p.add_argument('--repeat-window', type=int, default=3)
    p.add_argument('--min-branch-gap', type=int, default=3)
    p.add_argument('--hle-evaluator', choices=['judge','exact'], default='judge')
    p.add_argument('--judge-profile', help='Required for HLE judge mode; also included in budget')
    p.add_argument('--tool-timeout', type=float, default=300, help='Default shell timeout in seconds')
    p.add_argument('--max-tool-timeout', type=float, default=3600, help='SWE shell ceiling for model-requested timeouts')
    p.add_argument('--cache', type=Path, default=Path('run/inference-cache'), help='SWE native runtime cache')
    p.add_argument('--eval-timeout', type=int, default=3600)
    p.add_argument('--pull-timeout', type=int, default=1800)
    p.add_argument('--cpus', type=int, default=4)
    p.add_argument('--memory', default='8g')
    p.add_argument('--dry-run', action='store_true', help='Validate tasks/config/prices only; no models, Docker, Java or downloads')
    maintenance = p.add_mutually_exclusive_group()
    maintenance.add_argument('--resume', action='store_true', help='Resume original output with identical inference options; completed jobs are skipped')
    maintenance.add_argument('--resume-partial', action='store_true', help='Resume and restart partial jobs from scratch once; retain old attempts and their costs')
    maintenance.add_argument('--eval-only', type=Path, metavar='RUN', help='Re-evaluate saved SWE patches into a NEW output; no LLM calls')
    maintenance.add_argument('--reparse-only', type=Path, metavar='RUN', help='Reclassify saved SWE test logs into a NEW output; no Docker or LLM calls')
    maintenance.add_argument('--precheck', action='store_true', help='Run empty/gold SWE evaluation baselines; no LLM calls')
    return p


def plan(args):
    if args.max_retries is not None:
        if args.max_retries < 0:
            raise ValueError('--max-retries must be nonnegative')
        args.replay_attempts = args.max_retries + 1
    if not math.isfinite(args.retry_wait) or args.retry_wait < 0:
        raise ValueError('--retry-wait must be finite and nonnegative')
    if args.dataset is None or args.mode is None:
        raise ValueError('Inference requires --dataset, --mode and --models')
    config = load_config(args.config)
    ksr_config = {}
    if args.mode=='ksr_branch1':
        from ..step_collection.ksr import load_config as load_ksr
        ksr_config=load_ksr(args.ksr_config,config.profiles)
        args.models=[ksr_config['strong_model'],ksr_config['weak_model']]
    from .routing import configure
    router = configure(args, config)
    if not args.models:
        raise ValueError('Inference requires --models or a router profile model pool')
    args.mode = 'ksr' if args.mode == 'ksr-branch' else args.mode
    if len(set(args.models)) != len(args.models):
        raise ValueError('--models contains duplicate profiles')
    if (args.mode == 'single-model' and len(args.models) != 1) or (args.mode != 'single-model' and len(args.models) < 2):
        raise ValueError('single-model requires one profile; random/KSR require at least two')
    until = args.stop_mode != 'run-mode'
    if until and args.budget is None:
        raise ValueError('budget-mode requires --budget')
    if until and args.runs != 1:
        raise ValueError('--runs is for run-mode only')
    for name in ['budget','trace_budget','tool_timeout','max_tool_timeout']:
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f'--{name.replace("_","-")} must be finite and positive')
    for name in ['runs','step_limit','replay_attempts','error_window','repeat_window','min_branch_gap','cpus','eval_timeout','pull_timeout']:
        if getattr(args,name) <= 0:
            raise ValueError(f'{name} must be positive')
    if args.max_depth < 0 or args.max_nodes < (len(args.models) if args.mode == 'ksr' else 1):
        raise ValueError('max-depth must be nonnegative and max-nodes must cover all KSR roots')
    if args.limit is not None and args.limit <= 0:
        raise ValueError('--limit must be positive')
    if args.tool_timeout > args.max_tool_timeout:
        raise ValueError('--tool-timeout must not exceed --max-tool-timeout')
    if args.output_dir.exists() and not (args.resume or args.resume_partial):
        raise ValueError('Output already exists; choose a new --output-dir')
    tasks = load_dataset(args.dataset, args.split)
    if args.limit:
        tasks = tasks[:args.limit]
    names = list(args.models)
    if any(t['benchmark']=='hle' for t in tasks) and args.hle_evaluator=='judge':
        if not args.judge_profile:
            raise ValueError('HLE requires --judge-profile or --hle-evaluator exact')
        names += [args.judge_profile]
    # Validate separately: aliases are allowed and retain the requested profile identity.
    for name in dict.fromkeys(names):
        profile = config.profile(name)
        if profile.kind not in {'openai','openai-compatible','ollama','openrouter','anthropic','gemini','scripted'}:
            raise ValueError(f'Expected a concrete model profile, got {profile.kind}: {name}')
        if profile.max_tokens <= 0 or not math.isfinite(profile.timeout) or profile.timeout <= 0:
            raise ValueError(f'Invalid max_tokens/timeout for {name}')
        if profile.context_window is not None and (type(profile.context_window) is not int or profile.context_window <= profile.max_tokens):
            raise ValueError(f'context_window must be an integer larger than max_tokens: {name}')
        if profile.kind != 'scripted':
            endpoint = urlparse(profile.endpoint or profile.base_url)
            if endpoint.scheme not in {'http','https'} or not endpoint.hostname or endpoint.username or endpoint.password:
                raise ValueError(f'Invalid endpoint for {name}; credentials must be in environment variables')
            if not args.dry_run and not profile.api_key_optional and not os.environ.get(profile.api_key_env):
                raise ValueError(f'Missing credential variable: {profile.api_key_env} ({name})')
        else:
            responses = config.path.parent / profile.responses_file
            if not isinstance(json.loads(responses.read_text()), list):
                raise ValueError(f'Invalid scripted responses: {responses}')
        if {'model','messages','tools','tool_choice','stream','max_tokens','temperature'}.intersection(profile.extra_body):
            raise ValueError(f'extra_body overrides core request fields: {name}')
    prices = load_prices(args.pricing, list(dict.fromkeys(names)))
    settings = Settings(mode=args.mode, model_pool=args.models, strong=args.models[0], weak=args.models[-1],
        seed=args.seed, runs=args.runs, until_budget=until, max_steps=args.step_limit,
        total_cost_limit=args.budget or 1e100, task_cost_limit=1e100,
        path_cost_limit=args.trace_budget or args.budget or 1e100,
        max_nodes=args.max_nodes, max_depth=args.max_depth, replay_attempts=args.replay_attempts,
        retry_wait=args.retry_wait, trace_delay=3.0,
        router=router, ksr_config=ksr_config,
        rules=Rules(args.error_window,args.repeat_window,args.min_branch_gap),
        stop_on_unknown_cost=args.stop_on_unknown_cost)
    return tasks, config, prices, settings


async def execute(args, tasks, config, prices, settings, *, episode_factory=None, provider_factory=None):
    kwargs = {'episode_factory': episode_factory or factory(args,args.output_dir),
              'evaluator': evaluator(args,tasks), 'resume': args.resume,
              'resume_partial': args.resume_partial,
              'pricing_profiles': ([args.judge_profile] if args.hle_evaluator == 'judge'
                  and args.judge_profile and any(t['benchmark'] == 'hle' for t in tasks) else [])}
    if provider_factory:
        kwargs['provider_factory'] = provider_factory
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    # A second resume process must not append calls to the same ledger.
    with args.output_dir.with_name('.' + args.output_dir.name + '.infer.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another inference process owns this output directory') from exc
        collector = InferenceCollector(tasks,config,prices,settings,args.output_dir,**kwargs)
        token = HTTP_RETRY_POLICY.set((settings.replay_attempts - 1, settings.retry_wait))
        result = None
        try:
            result = await collector.run()
            return result
        finally:
            HTTP_RETRY_POLICY.reset(token)
            if collector.started:
                export(collector)
                if getattr(args,'collect_steps',False):
                    from ..step_collection.labels import export as export_training
                    training=export_training(collector)
                    if result is not None:
                        result['step_collection']=training
                    report_path=args.output_dir/'summary.json'
                    report=json.loads(report_path.read_text())
                    report['step_collection']=training
                    from ..collection.collector import atomic_json
                    atomic_json(report_path,report)
                    with (args.output_dir/'SUMMARY.md').open('a') as stream:
                        stream.write(f'\nTraining samples: {training["valid_samples"]}/{training["samples"]} valid; `training/samples.jsonl`.\n')
                print((args.output_dir/'SUMMARY.md').read_text(), flush=True)


def main(argv=None):
    return run(parser().parse_args(argv))


def run(args):
    args.output_dir = args.output_dir.resolve()
    args.cache = args.cache.resolve()
    try:
        if args.eval_only or args.reparse_only or args.precheck:
            from .maintenance import evaluate_saved, precheck
            if args.dry_run and (args.eval_only or args.reparse_only):
                raise ValueError('Use --reparse-only for offline log review; --dry-run applies to inference/precheck')
            report = precheck(args) if args.precheck else evaluate_saved(args, logs_only=bool(args.reparse_only))
            print(json.dumps({k:v for k,v in report.items() if k!='results'}, ensure_ascii=False, indent=2))
            return int(any(r['status'] not in {'passed','failed','ready','static_checked'} for r in report['results']))
        tasks, config, prices, settings = plan(args)
        from ..step_collection.ksr import task_schedule
        jobs_per_round = len(task_schedule(settings.ksr_config, tasks)) if settings.mode=='ksr_branch1' else len(tasks)
        print(json.dumps({'dry_run':args.dry_run,'tasks':len(tasks),
            'jobs_per_round':jobs_per_round,
            'splits':{s:sum(t['split']==s for t in tasks) for s in ['train','test']},
            'mode':args.mode, 'router':settings.router or None, 'ksr_config':settings.ksr_config or None, 'models':{n:{'model':config.profile(n).model,'extra_body':config.profile(n).extra_body,
                                        'max_tokens':config.profile(n).max_tokens} for n in args.models},
            'stop_mode':args.stop_mode,'rounds':None if settings.until_budget else args.runs,
            'budget':args.budget,'currency':prices['currency'],'output_dir':str(args.output_dir)},
            ensure_ascii=False,indent=2))
        if args.dry_run:
            return 0
        if any(t['benchmark']=='scienceworld' for t in tasks):
            java=Path(sys.prefix)/'lib/jvm'
            if not (java/'bin/java').exists():
                java=Path(sys.base_prefix)/'lib/jvm'
            if not os.environ.get('JAVA_HOME') and (java/'bin/java').exists():
                os.environ['JAVA_HOME']=str(java)
            if os.environ.get('JAVA_HOME'):
                os.environ['PATH']=os.environ['JAVA_HOME']+'/bin'+os.pathsep+os.environ.get('PATH','')
            if not doctor()['scienceworld_ready']:
                raise ValueError('ScienceWorld/Java unavailable; activate opensquilla')
            if any(t.get('env_step_limit') for t in tasks if t['benchmark'] == 'scienceworld'):
                from importlib.metadata import version
                if version('scienceworld') != '1.2.3':
                    raise ValueError('VirtualWorld requires scienceworld==1.2.3')
        if any(t['benchmark']=='alfworld' for t in tasks):
            from importlib.metadata import version
            if (version('alfworld') != '0.4.2' or version('textworld') != '1.7.0'
                    or version('fast-downward-textworld') != '20.6.4'):
                raise ValueError('ALFWorld requires alfworld==0.4.2, textworld==1.7.0 and fast-downward-textworld==20.6.4')
            from textworld.envs.pddl import PddlEnv
            env = PddlEnv()
            env.close()
        report = asyncio.run(execute(args,tasks,config,prices,settings))
        # A successful retry can leave earlier charges unknown; that is an
        # accounting caveat, not a failed run. Explicit strict stops still fail.
        if report['failed_nodes'] or report['unscored_traces'] or report['stop_reason']=='unknown_api_cost':
            return 1
        return 0 if not settings.until_budget or report['stop_reason']=='budget_exhausted' else 1
    except (ValueError,OSError,ImportError) as exc:
        print(f'infer: {exc}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
