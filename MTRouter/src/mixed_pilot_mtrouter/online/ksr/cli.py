"""Independent entry point for KSR cold-start Reference Teacher training."""
import asyncio
import json
from pathlib import Path
import yaml

from ..cli import parser, api_credentials, prepare
from .config import load_config, validate_config
from .recovery import method_fingerprint, prepare_runtime_repair


def main(argv=None):
    p = parser()
    p.add_argument('--credentials-config', type=Path, help='Read only api_keys from an existing YAML at runtime')
    p.add_argument('--reuse-cold-start', type=Path, help='Reuse a completed compatible run directory or cache.json')
    p.add_argument('--cold-only', action='store_true', help='Collect/train/evaluate cold start, then exit; --resume continues online')
    p.add_argument('--runtime-repair-from', help='With --resume, verify an explicit ancestor source hash and record the reviewed runtime repair')
    args = p.parse_args(argv)
    try:
        if args.runtime_repair_from and not args.resume:
            raise ValueError('--runtime-repair-from requires --resume')
        cfg = load_config(args.config, args.set)
        if args.credentials_config:
            keys = yaml.safe_load(args.credentials_config.read_text()).get('api_keys', {})
            cfg['api_keys'] = {k: cfg['api_keys'].get(k) or keys.get(k, '') for k in ('dashscope','zhipu','newcli')}
        if args.reuse_cold_start:
            cfg['cold_start']['reuse_from'] = str(args.reuse_cold_start.expanduser().resolve())
        if args.output_dir: cfg['output_dir'] = str(args.output_dir.expanduser().resolve())
        factories = (None, None)
        if args.smoke:
            from ..smoke import smoke_config, smoke_factories
            cfg = smoke_config(cfg)
            cfg['cold_start'].update(trajectory_steps=20, checkpoint_steps=5)
            factories = smoke_factories(cfg)
        elif cfg['encoder']['backend'] == 'hash':
            raise ValueError('Hash features are only for --smoke')
        if args.evaluate_checkpoint:
            cfg['teacher']['enabled'] = False
            cfg['evaluation']['enabled'] = True
        validate_config(cfg)
        with api_credentials(cfg, args):
            train, valid, manifest, harness, prices, references = prepare(cfg,
                check_credentials=not args.dry_run, simulation=args.smoke)
            from mixed_pilot_harness.step_collection.ksr import load_config as load_ksr
            rules = load_ksr(cfg['cold_start']['ksr_config'], harness.profiles)
            if set((rules['strong_model'],rules['weak_model'])) != set(cfg['model_pool']):
                raise ValueError('KSR and Router model pools differ')
            base_fingerprint = manifest['runtime_fingerprint']
            manifest['runtime_fingerprint'] = method_fingerprint(base_fingerprint, rules, cfg['project_root'])
            repair = (prepare_runtime_repair(cfg, manifest, base_fingerprint, rules, args.runtime_repair_from)
                      if args.runtime_repair_from else None)
            if args.dry_run:
                print(json.dumps({'status':'dry_run', 'train_tasks':len(train), 'evaluation_tasks':len(valid),
                    'cold_start':cfg['cold_start'], 'online':cfg['online'], 'decision':cfg['decision'],
                    'training_objective':'separate_Y_C', 'rollout':cfg['rollout'], 'budget':cfg['budget'],
                    'output_dir':cfg['output_dir'], 'step_semantics':'unique Actor call with environment transition'}, indent=2))
                return 0
            import torch
            if args.smoke: torch.set_num_threads(1)
            from .trainer import KSRTrainer
            from .learning import SeparateLearner
            learner = SeparateLearner.load(args.evaluate_checkpoint, {'encoder':{'device':cfg['encoder']['device']}}) if args.evaluate_checkpoint else None
            if learner and (learner.model_names != cfg['model_pool'] or learner.config['rollout'] != cfg['rollout']
                            or learner.config['decision'] != cfg['decision']):
                raise ValueError('Evaluation model pool, limits and beta must match checkpoint')
            t = KSRTrainer(cfg, train, valid, manifest, harness, prices, resume=args.resume,
                episode_factory=factories[0], provider_factory=factories[1], simulation=args.smoke,
                references=references, learner=learner)
            if repair:
                from ..teacher import append_jsonl
                append_jsonl(t.out/'runtime_repairs.jsonl', repair)
                t.event('runtime_repair_verified', **repair)
            if args.evaluate_checkpoint:
                asyncio.run(t._evaluate(learner.metadata.get('completed_iterations',0)))
                result = t._summary('evaluation_only')
            elif args.cold_only:
                if not args.resume: t._save(0)
                if not t.cold_done: asyncio.run(t._cold_start())
                result = t._summary('cold_start_completed')
            else:
                result = asyncio.run(t.run())
            print(json.dumps(result, indent=2))
            return 0 if result['stop_reason'] in {'iterations_completed','cold_start_completed','evaluation_only'} else 2
    except (ValueError, FileNotFoundError) as exc:
        print('ksr-reference-router: '+str(exc))
        return 2
