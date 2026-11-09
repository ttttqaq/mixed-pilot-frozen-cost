"""Online experiment with historical cold cost and round-boundary budget stopping."""
import asyncio
import hashlib
import json
from pathlib import Path
import yaml

from ..cli import parser, api_credentials, prepare
from ..ksr.recovery import method_fingerprint as ksr_fingerprint
from .config import load_config, validate_config


def method_fingerprint(base, rules, root):
    digest = hashlib.sha256(ksr_fingerprint(base, rules, root).encode())
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main(argv=None):
    p = parser()
    p.description = __doc__
    p.add_argument('--credentials-config', type=Path, help='Read only api_keys at runtime')
    p.add_argument('--reuse-cold-start', type=Path, help='Reuse a compatible cache produced by this method')
    p.add_argument('--init-cold-checkpoint', type=Path, help='Start a NEW run from a completed original KSR cold checkpoint')
    p.add_argument('--budget-cold-source', type=Path, required=True, help='Historical cold run charged to the soft budget')
    p.add_argument('--cold-only', action='store_true', help='Initialize/collect/train/evaluate cold start only')
    args = p.parse_args(argv)
    try:
        cfg = load_config(args.config, args.set)
        if args.credentials_config:
            keys = yaml.safe_load(args.credentials_config.read_text()).get('api_keys', {})
            cfg['api_keys'] = {k: cfg['api_keys'].get(k) or keys.get(k, '') for k in ('dashscope', 'zhipu', 'newcli')}
        if args.reuse_cold_start:
            cfg['cold_start']['reuse_from'] = str(args.reuse_cold_start.expanduser().resolve())
        if args.init_cold_checkpoint:
            cfg['initialization']['cold_checkpoint'] = str(args.init_cold_checkpoint.expanduser().resolve())
        if args.evaluate_checkpoint and cfg['initialization']['cold_checkpoint']:
            raise ValueError('Choose checkpoint evaluation or cold initialization')
        if args.output_dir:
            cfg['output_dir'] = str(args.output_dir.expanduser().resolve())
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
        if cfg['evaluation']['after_cold_start']:
            raise ValueError('Online experiment starts directly; evaluation.after_cold_start must be false')
        if not cfg['initialization']['cold_checkpoint']:
            raise ValueError('Online experiment requires --init-cold-checkpoint')
        if args.cold_only or args.reuse_cold_start or args.evaluate_checkpoint:
            raise ValueError('Use the original entry for cold-only, cache reuse or checkpoint evaluation')
        if not Path(cfg['initialization']['cold_checkpoint']).is_relative_to(args.budget_cold_source.expanduser().resolve()):
            raise ValueError('Historical cold budget source must contain the initialization checkpoint')
        cfg['experiment_budget'] = {'historical_run': str(args.budget_cold_source.expanduser().resolve()),
                                    'threshold': cfg['budget']['total']}
        with api_credentials(cfg, args):
            train, valid, manifest, harness, prices, references = prepare(cfg,
                check_credentials=not args.dry_run, simulation=args.smoke)
            from mixed_pilot_harness.step_collection.ksr import load_config as load_ksr
            rules = load_ksr(cfg['cold_start']['ksr_config'], harness.profiles)
            if {rules['strong_model'], rules['weak_model']} != set(cfg['model_pool']):
                raise ValueError('KSR and Router model pools differ')
            manifest['runtime_fingerprint'] = method_fingerprint(manifest['runtime_fingerprint'], rules, cfg['project_root'])
            if args.dry_run:
                initial = cfg['initialization']['cold_checkpoint']
                if initial and not args.resume and not Path(initial).is_file():
                    raise ValueError('Initial cold checkpoint does not exist')
                print(json.dumps({'status': 'dry_run', 'train_tasks': len(train), 'evaluation_tasks': len(valid),
                    'cold_start': cfg['cold_start'], 'initialization': cfg['initialization'], 'online': cfg['online'],
                    'decision': cfg['decision'], 'experiment_budget': cfg['experiment_budget'], 'training_objective': 'joint_Y_C_cold_then_Y_only',
                    'cost_prediction': 'immutable_cold_ensemble', 'output_dir': cfg['output_dir']}, indent=2))
                return 0
            import torch
            if args.smoke:
                torch.set_num_threads(1)
            from .learning import FrozenCostLearner
            from .round_trainer import RoundBudgetTrainer
            learner = None
            if args.evaluate_checkpoint:
                learner = FrozenCostLearner.load(args.evaluate_checkpoint, {'encoder': {'device': cfg['encoder']['device']}})
                if (learner.model_names != cfg['model_pool'] or learner.config['rollout'] != cfg['rollout'] or
                        learner.config['decision'] != cfg['decision'] or learner.frozen_cost_router is None):
                    raise ValueError('Evaluation requires a frozen C checkpoint with matching model pool, limits and beta')
            elif cfg['initialization']['cold_checkpoint'] and not args.resume:
                from .initialization import load_cold_checkpoint
                learner = load_cold_checkpoint(cfg, manifest)
            trainer = RoundBudgetTrainer(cfg, train, valid, manifest, harness, prices, resume=args.resume,
                episode_factory=factories[0], provider_factory=factories[1], simulation=args.smoke,
                references=references, learner=learner)
            if args.evaluate_checkpoint:
                asyncio.run(trainer._evaluate(learner.metadata.get('completed_iterations', 0)))
                result = trainer._summary('evaluation_only')
            elif args.cold_only:
                if not args.resume:
                    trainer._save(0)
                if not trainer.cold_done:
                    asyncio.run(trainer._cold_start())
                result = trainer._summary('cold_start_completed')
            else:
                result = asyncio.run(trainer.run())
            print(json.dumps(result, indent=2))
            return 0 if result['stop_reason'] in {'iterations_completed', 'budget_round_completed'} else 2
    except (ValueError, FileNotFoundError) as exc:
        print('frozen-cost-online: '+str(exc))
        return 2
