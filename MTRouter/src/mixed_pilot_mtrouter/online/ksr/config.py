"""Independent configuration; the reference baseline remains available."""
from copy import deepcopy
from pathlib import Path
import yaml
from .. import config as legacy
from . import METHOD

DEFAULTS = deepcopy(legacy.DEFAULTS)
DEFAULTS.update(method=METHOD, output_dir='outputs/ksr-reference-router-run01')
DEFAULTS['cold_start'].update(strategy='ksr-brunch-1', trajectory_steps=500,
    checkpoint_steps=100, reuse_from=None, ksr_config='KSR_config.json', max_depth=3, max_nodes=14)
DEFAULTS['online']['iterations'] = 5
DEFAULTS['budget']['total'] = 80.
DEFAULTS['optimization'].update(pairwise_weight=0., anchor_weight=0., cost_loss_weight=1.)
DEFAULTS['decision'] = {'beta': .5}
DEFAULTS['audit'].update(reward_threshold=.1, cost_threshold=.05)
DEFAULTS['evaluation']['after_cold_start'] = True


def validate_config(config):
    base = deepcopy(config)
    if base.pop('method') != METHOD:
        raise ValueError('Unexpected method')
    decision = base.pop('decision')
    legacy._number(decision['beta'], 'decision.beta')
    cold = base['cold_start']
    if cold['strategy'] != 'ksr-brunch-1':
        raise ValueError('cold_start.strategy must be ksr-brunch-1')
    for key in ('trajectory_steps', 'checkpoint_steps', 'max_nodes'):
        legacy._number(cold.pop(key), 'cold_start.'+key, strict=True, integer=True)
    legacy._number(cold.pop('max_depth'), 'cold_start.max_depth', integer=True)
    legacy._string(cold.pop('ksr_config'), 'cold_start.ksr_config')
    reuse = cold.pop('reuse_from')
    if reuse is not None:
        legacy._string(reuse, 'cold_start.reuse_from')
    cold['strategy'] = 'uniform'
    for key in ('reward_threshold', 'cost_threshold'):
        legacy._number(base['audit'].pop(key), 'audit.'+key)
    if type(base['evaluation'].pop('after_cold_start')) is not bool:
        raise ValueError('evaluation.after_cold_start must be boolean')
    legacy.validate_config(base)
    if config['optimization']['pairwise_weight'] or config['optimization']['anchor_weight']:
        raise ValueError('Only separate Y/C losses and parameter regularization are supported')
    if config['optimization']['cost_loss_weight'] <= 0 or config['rollout']['reward'] != 'success':
        raise ValueError('Both Y(success) and C heads must be supervised')
    if config['rollout']['step_limit'] > 50 or config['rollout']['task_budget'] > 1.:
        raise ValueError('Limits are at most 50 Actor calls/path and 1 CNY/task')
    if abs(config['budget']['lambda_'] / config['budget']['cost_reference'] - decision['beta']) > 1e-10:
        raise ValueError('budget.lambda_/cost_reference must equal decision.beta')


def load_config(path, overrides=None, *, defaults=None, validator=validate_config):
    source = Path(path).expanduser().resolve()
    given = yaml.load(source.read_text(), Loader=legacy._UniqueKeyLoader) or {}
    cfg = legacy._merge(DEFAULTS if defaults is None else defaults, given)
    for spec in overrides or []:
        legacy._override(cfg, spec)
    # Beta is a decision parameter, not a training target coefficient.
    cfg['budget']['lambda_'] = cfg['decision']['beta'] * cfg['budget']['cost_reference']
    validator(cfg)
    root = Path(cfg['project_root']).expanduser()
    if not root.is_absolute():
        root = source.parent / root
    root = root.resolve()
    cfg['project_root'] = str(root)
    for dotted in ('harness_config', 'pricing', 'output_dir', 'dataset.root', 'model.model_metadata', 'cold_start.ksr_config'):
        obj = cfg
        keys = dotted.split('.')
        for key in keys[:-1]: obj = obj[key]
        obj[keys[-1]] = str((root / Path(obj[keys[-1]]).expanduser()).resolve())
    if cfg['cold_start']['reuse_from']:
        cfg['cold_start']['reuse_from'] = str((root / Path(cfg['cold_start']['reuse_from']).expanduser()).resolve())
    return cfg
