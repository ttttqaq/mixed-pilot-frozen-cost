"""Explicit import of a completed cold checkpoint; never resume old online weights."""
import hashlib
import json
from pathlib import Path

from ..ksr.learning import SeparateLearner
from .learning import FrozenCostLearner


def load_cold_checkpoint(config, manifest):
    path = Path(config['initialization']['cold_checkpoint']).resolve()
    source_root = next((p for p in path.parents if (p/'split_manifest.json').is_file()
                        and (p/'config.resolved.json').is_file()), None)
    if source_root is None:
        raise ValueError('Cold checkpoint requires its original run manifest and resolved configuration')
    saved = json.loads((source_root/'split_manifest.json').read_text())
    for key in ('split_fingerprint', 'reference_fingerprint'):
        if not saved.get(key) or saved.get(key) != manifest.get(key):
            raise ValueError('Cold initialization data/reference mismatch: '+key)
    for key in ('harness_config', 'pricing', 'model_metadata'):
        if (not saved.get('runtime_artifact_hashes', {}).get(key) or
                saved['runtime_artifact_hashes'][key] != manifest.get('runtime_artifact_hashes', {}).get(key)):
            raise ValueError('Cold initialization runtime artifact mismatch: '+key)
    source = SeparateLearner.load(path, {'encoder': {'device': config['encoder']['device']}})
    meta = source.metadata
    if (meta.get('completed_iterations') != 0 or
            meta.get('collected_cold_steps') != config['cold_start']['trajectory_steps'] or
            meta.get('objective') != 'separate_Y_C'):
        raise ValueError('Require the completed KSR step checkpoint, not latest/best/online weights')
    for key in ('model_pool', 'strong_model', 'rollout', 'model'):
        if source.config[key] != config[key]:
            raise ValueError('Cold initialization configuration mismatch: '+key)
    for key in ('backend', 'model_name', 'max_tokens', 'hash_dim'):
        if source.config['encoder'][key] != config['encoder'][key]:
            raise ValueError('Cold initialization encoder mismatch: '+key)
    if source.cost_reference != config['budget']['cost_reference']:
        raise ValueError('Cold initialization cost unit mismatch')
    old = source.config['cold_start']
    for key in ('strategy', 'trajectory_steps', 'checkpoint_steps', 'max_depth', 'max_nodes'):
        if old[key] != config['cold_start'][key]:
            raise ValueError('Cold initialization collection mismatch: '+key)
    data_path = path.with_suffix('.json')
    data = json.loads(data_path.read_text())
    if len(data['calls']) != config['cold_start']['trajectory_steps'] or not data.get('complete'):
        raise ValueError('Cold checkpoint data snapshot is incomplete')
    if data['fingerprint'] != meta.get('cache_fingerprint'):
        raise ValueError('Cold checkpoint weights/data fingerprint mismatch')
    # Require the original rules file so imported weights retain their collection semantics.
    rules_path = Path(config['cold_start']['ksr_config'])
    old_rules = Path(old['ksr_config'])
    if not old_rules.is_file() or old_rules.read_bytes() != rules_path.read_bytes():
        raise ValueError('Cold initialization KSR rules differ or their source is unavailable')
    learner = FrozenCostLearner(config, _model_metadata=source.model_metadata)
    learner.router.load_state_dict(source.router.state_dict())
    learner.router_optimizer.load_state_dict(source.router_optimizer.state_dict())
    for group in learner.router_optimizer.param_groups:
        group['lr'] = config['optimization']['learning_rate']
        group['weight_decay'] = config['optimization']['weight_decay']
    learner.router_updates = source.router_updates
    learner.features.cache.update(source.features.cache)
    learner.freeze_cost()
    learner.cold_initialization = {'checkpoint': str(path),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'data_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
        'trajectory_steps': meta['collected_cold_steps'], 'cache_fingerprint': data['fingerprint'],
        'source_run': str(source_root), 'source_costs_imported': False}
    return learner
