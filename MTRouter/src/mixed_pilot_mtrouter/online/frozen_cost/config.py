"""The cost-loss weight applies only to the unchanged joint cold-start fit."""
from copy import deepcopy
from pathlib import Path

from ..ksr import METHOD as KSR_METHOD
from ..ksr import config as ksr
from . import METHOD

DEFAULTS = deepcopy(ksr.DEFAULTS)
DEFAULTS.update(method=METHOD, output_dir='outputs/ksr-frozen-cost-router-run01')
DEFAULTS['initialization'] = {'cold_checkpoint': None}
DEFAULTS['cold_start']['finish_current_task'] = True


def validate_config(config):
    base = deepcopy(config)
    if base.get('method') != METHOD:
        raise ValueError('Require the frozen-cost Y-only method')
    base['method'] = KSR_METHOD
    if base['cold_start'].pop('finish_current_task') is not True:
        raise ValueError('This method finishes the current task after reaching the cold-step threshold')
    initial = base.pop('initialization')['cold_checkpoint']
    if initial is not None and (not isinstance(initial, str) or not initial.strip()):
        raise ValueError('initialization.cold_checkpoint must be a path or null')
    if initial and config['cold_start']['reuse_from']:
        raise ValueError('Choose cold checkpoint initialization or cache reuse, not both')
    ksr.validate_config(base)
    if not config['cold_start']['enabled']:
        raise ValueError('Frozen C requires a completed KSR cold start')


def load_config(path, overrides=None):
    config = ksr.load_config(path, overrides, defaults=DEFAULTS, validator=validate_config)
    initial = config['initialization']['cold_checkpoint']
    if initial:
        config['initialization']['cold_checkpoint'] = str((Path(config['project_root']) / Path(initial).expanduser()).resolve())
    return config
