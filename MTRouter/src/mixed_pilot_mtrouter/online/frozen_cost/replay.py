"""Re-fit historical KSR snapshots without collecting or revealing future labels."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from ..cli import parser, api_credentials, prepare
from ..data import assert_training_rows
from ..ksr.cold import labels
from ..ksr.config import load_config, validate_config
from ..ksr.learning import SeparateLearner
from ..ksr.recovery import method_fingerprint as ksr_fingerprint
from ..ksr.trainer import KSRTrainer
from ..runtime import _atomic, UnknownCostError
from ..teacher import append_jsonl

SCHEMA = 'ksr-historical-snapshot-replay-v1'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state().clone(),
            'cuda': [x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def numpy_json():
    name, keys, pos, gaussian, cached = np.random.get_state()
    return [name, keys.tolist(), pos, gaussian, cached]


def restore_numpy(value):
    np.random.set_state((value[0], np.array(value[1], dtype=np.uint32), *value[2:]))


def source_manifest(source_run, config, manifest, *, simulation=False):
    """Check source data identity and record every exact historical snapshot hash."""
    source = Path(source_run).expanduser().resolve()
    old = json.loads((source/'config.resolved.json').read_text())
    saved = json.loads((source/'split_manifest.json').read_text())
    for key in ('split_fingerprint', 'reference_fingerprint'):
        if not saved.get(key) or saved.get(key) != manifest.get(key):
            raise ValueError('Replay source data/reference mismatch: '+key)
    for key in ('harness_config', 'pricing', 'model_metadata'):
        value = saved.get('runtime_artifact_hashes', {}).get(key)
        if not value or value != manifest.get('runtime_artifact_hashes', {}).get(key):
            raise ValueError('Replay source artifact mismatch: '+key)
    for key in ('model_pool', 'strong_model', 'rollout'):
        if old[key] != config[key]:
            raise ValueError('Replay source configuration mismatch: '+key)
    if old['budget']['cost_reference'] != config['budget']['cost_reference']:
        raise ValueError('Replay source cost units changed')
    if old['decision'] != config['decision']:
        raise ValueError('Replay source beta differs from evaluation beta')
    if not simulation:
        for key in ('model', 'optimization'):
            if old[key] != config[key]:
                raise ValueError('Replay must retain original cold training '+key)
        for key in ('backend', 'model_name', 'max_tokens', 'hash_dim'):
            if old['encoder'][key] != config['encoder'][key]:
                raise ValueError('Replay source encoder mismatch: '+key)
    for key in ('strategy', 'trajectory_steps', 'checkpoint_steps', 'max_depth', 'max_nodes'):
        if old['cold_start'][key] != config['cold_start'][key]:
            raise ValueError('Replay source cold setting mismatch: '+key)
    if sha256(old['cold_start']['ksr_config']) != sha256(config['cold_start']['ksr_config']):
        raise ValueError('Replay source KSR rules changed')
    count = old['cold_start']['trajectory_steps']
    interval = old['cold_start']['checkpoint_steps']
    steps = list(range(interval, count+1, interval))
    if not steps or steps[-1] != count:
        steps.append(count)
    snapshots = {}
    previous_calls, fingerprint = set(), None
    for step in steps:
        path = source/'cold_start/checkpoints'/('step_%06d.json'%step)
        data = json.loads(path.read_text())
        if len(data['calls']) != step or not previous_calls.issubset(data['calls']):
            raise ValueError('Historical snapshot count/prefix mismatch at '+str(step))
        if fingerprint is not None and data['fingerprint'] != fingerprint:
            raise ValueError('Historical snapshots have different cache identities')
        fingerprint, previous_calls = data['fingerprint'], set(data['calls'])
        snapshots[str(step)] = {'path': str(path), 'sha256': sha256(path)}
    return {'source_run': str(source), 'source_seed': old['seed'], 'replay_seed': config['seed'],
            'source_config_sha256': sha256(source/'config.resolved.json'),
            'source_manifest_sha256': sha256(source/'split_manifest.json'),
            'snapshots': snapshots, 'milestones': steps, 'cache_fingerprint': fingerprint,
            'label_semantics': 'each historical snapshot only; no future terminal evidence'}


def evaluation_spending(directory):
    known = reserved = 0.
    uncertain = []
    for path in sorted((Path(directory)/'calls').glob('c*.json')):
        row = json.loads(path.read_text())
        if row.get('currency') != 'CNY':
            raise ValueError('Replay evaluation ledger currency differs')
        for key in ('cost', 'reserved_cost'):
            value = row.get(key)
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
                raise ValueError('Replay evaluation has invalid recorded fees')
        known += float(row.get('cost') or 0.)
        reserved += float(row.get('reserved_cost') or 0.)
        if not row.get('cost_complete') and (row.get('status') == 'in_flight' or row.get('reserved_cost', 0.) <= 0):
            uncertain.append(row['call_id'])
    return {'known_cost': known, 'reserved_cost': reserved, 'budget_cost': known+reserved,
            'unbounded_calls': uncertain}


class ColdReplay:
    """Train once per old milestone; evaluate 300/500 in separate resumable dirs."""
    def __init__(self, config, train, valid, manifest, harness, prices, source, output,
                 *, eval_steps=(300, 500), resume=False, simulation=False,
                 episode_factory=None, provider_factory=None):
        self.config = deepcopy(config)
        self.config['output_dir'] = str(Path(output).expanduser().resolve())
        self.config['teacher']['enabled'] = False
        self.out = Path(self.config['output_dir'])
        self.train, self.valid, self.manifest = train, valid, deepcopy(manifest)
        self.harness, self.prices = harness, prices
        self.simulation, self.episode_factory, self.provider_factory = simulation, episode_factory, provider_factory
        self.eval_steps = tuple(sorted(set(int(x) for x in eval_steps)))
        self.source = source_manifest(source, self.config, manifest, simulation=simulation)
        if not set(self.eval_steps).issubset(self.source['milestones']):
            raise ValueError('Evaluation steps must be saved historical milestones')
        self.manifest['replay'] = {**self.source, 'evaluation_steps': list(self.eval_steps), 'schema': SCHEMA}
        self._lock = None
        if not simulation:
            import fcntl
            self.out.parent.mkdir(parents=True, exist_ok=True)
            self._lock = self.out.with_name(self.out.name+'.lock').open('a+')
            try:
                fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self._lock.close()
                raise ValueError('Another process owns this replay output')
        if resume:
            if json.loads((self.out/'config.resolved.json').read_text()) != self.config:
                raise ValueError('Replay resume requires identical configuration')
            if json.loads((self.out/'replay-manifest.json').read_text()) != self.manifest:
                raise ValueError('Replay source, data, prices or execution source changed')
            self.state = json.loads((self.out/'replay-state.json').read_text())
            self.learner = SeparateLearner.load(self.out/self.state['checkpoint'])
            restore_numpy(self.state['numpy_rng'])
        else:
            self.out.mkdir(parents=True, exist_ok=False)
            _atomic(self.out/'config.resolved.json', self.config)
            _atomic(self.out/'replay-manifest.json', self.manifest)
            np.random.seed(self.config['seed'])
            self.learner = SeparateLearner(self.config)
            self.state = {'trained_step': 0, 'checkpoint': 'checkpoints/step_000000.pt',
                          'evaluated_steps': [], 'reports': {}, 'numpy_rng': numpy_json()}
            self._save_training(0, {'samples': 0, 'updates': 0, 'objective': 'separate_Y_C'})

    def _save_training(self, step, report):
        name = 'checkpoints/step_%06d.pt'%step
        self.learner.save(self.out/name, {'completed_iterations': 0, 'collected_cold_steps': step,
            'objective': 'separate_Y_C', 'replay_schema': SCHEMA, 'seed': self.config['seed'],
            'source_snapshot': self.source['snapshots'].get(str(step)), 'simulation': self.simulation})
        self.state.update(trained_step=step, checkpoint=name, numpy_rng=numpy_json())
        _atomic(self.out/'replay-state.json', self.state)

    def summary(self, stop):
        fees = {str(step): evaluation_spending(self.out/(str(step)+'step')) for step in self.eval_steps}
        result = {'schema': SCHEMA, 'stop_reason': stop, 'source_run': self.source['source_run'],
            'source_seed': self.source['source_seed'], 'replay_seed': self.config['seed'],
            'trained_step': self.state['trained_step'], 'evaluated_steps': self.state['evaluated_steps'],
            'evaluations': self.state['reports'], 'online_iterations': 0, 'new_cold_actor_calls': 0,
            'new_teacher_calls': 0, 'evaluation_spending': fees,
            'known_cost': sum(f['known_cost'] for f in fees.values()),
            'reserved_cost': sum(f['reserved_cost'] for f in fees.values()),
            'budget_cost': sum(f['budget_cost'] for f in fees.values()),
            'budget_limit': self.config['budget']['total'], 'simulation': self.simulation}
        _atomic(self.out/'summary.json', result)
        return result

    async def evaluate(self, step):
        stage = self.out/(str(step)+'step')
        other = [evaluation_spending(self.out/(str(s)+'step')) for s in self.eval_steps if s != step]
        if any(x['unbounded_calls'] for x in other):
            raise UnknownCostError('Other replay evaluation contains unbounded fees')
        left = self.config['budget']['total'] - sum(x['budget_cost'] for x in other)
        if left <= 0:
            return None
        cfg = deepcopy(self.config)
        cfg['output_dir'] = str(stage)
        cfg['budget']['total'] = left
        cfg['evaluation']['enabled'] = True
        rng, cache = capture_rng(), OrderedDict(self.learner.features.cache)
        trainer = None
        try:
            existing = (stage/'state.json').is_file()
            # Before the first API call, a state checkpoint is committed below.
            if stage.exists() and not existing:
                if any((stage/'calls').glob('*.json')):
                    raise ValueError('Incomplete replay stage initialization contains calls')
                import shutil
                shutil.rmtree(stage)
            trainer = KSRTrainer(cfg, self.train, self.valid, self.manifest, self.harness, self.prices,
                resume=existing, learner=self.learner, simulation=self.simulation,
                episode_factory=self.episode_factory, provider_factory=self.provider_factory, references={})
            trainer.ksr_trained_steps = step
            trainer_rng = trainer.rng.getstate()
            trainer._save(0, cold_done=True)
            # Save an easily reusable checkpoint before paid evaluation starts.
            self.learner.save(stage/'cold.pt', {'completed_iterations': 0, 'collected_cold_steps': step,
                'objective': 'separate_Y_C', 'replay_schema': SCHEMA, 'seed': self.config['seed']})
            try:
                report = await trainer._evaluate(step)
            finally:
                trainer.rng.setstate(trainer_rng)
            trainer._save(0, cold_done=True)
            stage_summary = trainer._summary('evaluation_completed' if report and report['complete'] else 'evaluation_incomplete')
            stage_summary.update(replay_schema=SCHEMA, replay_step=step,
                                 source_run=self.source['source_run'], replay_seed=self.config['seed'],
                                 cold_trajectory_steps=step, cold_checkpoint_steps=step,
                                 new_cold_actor_calls=0, new_teacher_calls=0,
                                 source_snapshot=self.source['snapshots'][str(step)])
            _atomic(stage/'summary.json', stage_summary)
            return report
        finally:
            if trainer is not None and trainer._run_lock is not None:
                trainer._run_lock.close()
            self.learner.features.cache = cache
            restore_rng(rng)

    async def run(self):
        try:
            current = self.state['trained_step']
            # A crash after training but during evaluation resumes evaluation first.
            if current in self.eval_steps and current not in self.state['evaluated_steps']:
                if not await self._evaluate_and_commit(current):
                    return self.summary('evaluation_incomplete')
            for step in self.source['milestones']:
                if step <= self.state['trained_step']:
                    continue
                spec = self.source['snapshots'][str(step)]
                if sha256(spec['path']) != spec['sha256']:
                    raise ValueError('Historical snapshot changed during replay')
                data = json.loads(Path(spec['path']).read_text())
                all_rows, _ = labels(data)
                rows = [r for r in all_rows if r['labels_valid']]
                assert_training_rows(rows)
                report = self.learner.train_router(rows)
                report.update(step=step, snapshot_sha256=spec['sha256'],
                              collected_steps=len(data['calls']), valid_labels=len(rows),
                              unknown_labels=len(all_rows)-len(rows), seed=self.config['seed'])
                # Commit weights/RNG/progress before any paid evaluation.
                self._save_training(step, report)
                append_jsonl(self.out/'training_log.jsonl', report)
                print(json.dumps({'event': 'replay_fit', **report}), flush=True)
                if step in self.eval_steps and not await self._evaluate_and_commit(step):
                    return self.summary('evaluation_incomplete')
            return self.summary('cold_replay_completed')
        except BaseException:
            self.summary('interrupted')
            raise
        finally:
            if self._lock is not None:
                self._lock.close()

    async def _evaluate_and_commit(self, step):
        report = await self.evaluate(step)
        if report:
            self.state['reports'][str(step)] = report
        if not report or not report['complete']:
            _atomic(self.out/'replay-state.json', self.state)
            return False
        self.state['evaluated_steps'].append(step)
        _atomic(self.out/'replay-state.json', self.state)
        self.summary('running')
        return True


def main(argv=None):
    p = parser()
    p.description = __doc__
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--credentials-config', type=Path)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval-steps', default='300,500')
    p.add_argument('--budget', type=float, default=50.)
    args = p.parse_args(argv)
    try:
        if args.evaluate_checkpoint:
            raise ValueError('Replay accepts historical data snapshots, not --evaluate-checkpoint')
        cfg = load_config(args.config, args.set)
        cfg['seed'] = args.seed
        cfg['teacher']['enabled'] = False
        cfg['evaluation']['enabled'] = True
        cfg['budget']['total'] = args.budget
        if not args.output_dir:
            raise ValueError('Replay requires a new --output-dir')
        cfg['output_dir'] = str(args.output_dir.expanduser().resolve())
        if args.credentials_config:
            import yaml
            keys = yaml.safe_load(args.credentials_config.read_text()).get('api_keys', {})
            cfg['api_keys'] = {k: cfg['api_keys'].get(k) or keys.get(k, '') for k in ('dashscope', 'zhipu', 'newcli')}
        factories = (None, None)
        if args.smoke:
            from ..smoke import smoke_config, smoke_factories
            cfg = smoke_config(cfg)
            cfg['budget']['total'] = args.budget
            factories = smoke_factories(cfg)
        elif cfg['encoder']['backend'] == 'hash':
            raise ValueError('Hash features require --smoke')
        validate_config(cfg)
        eval_steps = tuple(int(x) for x in args.eval_steps.split(','))
        with api_credentials(cfg, args):
            train, valid, manifest, harness, prices, _ = prepare(cfg, check_credentials=not args.dry_run, simulation=args.smoke)
            # Compare reference provenance without enabling any Teacher request.
            from ..references import load_references
            load_references(cfg, train, manifest)
            from mixed_pilot_harness.step_collection.ksr import load_config as load_rules
            rules = load_rules(cfg['cold_start']['ksr_config'], harness.profiles)
            base = ksr_fingerprint(manifest['runtime_fingerprint'], rules, cfg['project_root'])
            manifest['runtime_fingerprint'] = hashlib.sha256((base+sha256(__file__)).encode()).hexdigest()
            source = source_manifest(args.source_run, cfg, manifest, simulation=args.smoke)
            if args.dry_run:
                print(json.dumps({'status': 'dry_run', 'schema': SCHEMA, 'source': source,
                                  'evaluation_steps': eval_steps, 'seed': args.seed,
                                  'evaluation_tasks': len(valid), 'new_cold_actor_calls': 0,
                                  'new_teacher_calls': 0, 'budget': args.budget}, indent=2))
                return 0
            if args.smoke:
                torch.set_num_threads(1)
            replay = ColdReplay(cfg, train, valid, manifest, harness, prices, args.source_run, cfg['output_dir'],
                eval_steps=eval_steps, resume=args.resume, simulation=args.smoke,
                episode_factory=factories[0], provider_factory=factories[1])
            result = asyncio.run(replay.run())
            print(json.dumps(result, indent=2))
            return 0 if result['stop_reason'] == 'cold_replay_completed' else 2
    except (ValueError, FileNotFoundError) as exc:
        print('ksr-cold-replay: '+str(exc))
        return 2
