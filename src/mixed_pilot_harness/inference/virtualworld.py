"""Native ALFWorld text interaction; no policy, gold plan or reference is model-visible."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..collection.episodes import ScienceWorldEpisode
from .alfworld_runtime import ALFWorldEnvironment


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_task(manifest, root):
    from .dataset import inside
    root = root.resolve()
    env = manifest['environment']
    if env.get('task_type') not in {'pick_and_place_simple', 'look_at_obj_in_light',
            'pick_clean_then_place_in_recep', 'pick_heat_then_place_in_recep',
            'pick_cool_then_place_in_recep', 'pick_two_obj_and_place'}:
        raise ValueError('Unknown ALFWorld task_type')
    game = inside(root, env.get('game_file', 'workspace/game.tw-pddl'))
    workspace = inside(root, manifest['workspace']['path'])
    if not game.is_file() or not game.is_relative_to(workspace):
        raise ValueError('ALFWorld game must exist inside the public workspace')
    if env.get('runtime', {}).get('textworld') != '1.7.0':
        raise ValueError('Unsupported ALFWorld TextWorld version')
    step_limit = env.get('step_limit', 200)
    if type(step_limit) is not int or step_limit <= 0:
        raise ValueError('ALFWorld step_limit must be positive')
    prompt = inside(root, manifest.get('prompt', 'prompt.md'))
    return {'task_id': manifest['id'], 'benchmark': 'alfworld',
            'task_name': env['task_type'], 'source_id': manifest['source']['upstream_id'],
            'split': manifest.get('routing_metadata', {}).get('router_split') or manifest['source']['split'],
            'source_split': manifest['source']['split'], 'source_revision': manifest['source']['revision'],
            'game_file': str(game), 'step_limit': step_limit,
            'content_hashes': {'game': fingerprint(game), 'prompt': fingerprint(prompt),
                               'manifest': fingerprint(root / 'manifest.json')}}


class ALFWorldEpisode(ScienceWorldEpisode):
    def __init__(self, task, prompt):
        prompt = dict(prompt)
        prompt.update(system_template='You are interacting with an ALFWorld household simulation. '
                      'Complete the objective using EXACTLY ONE command in a ```text``` code block. '
                      'Use ?all to inspect current admissible commands without taking a game step. '
                      'Use task completed to submit. Success requires the native environment won flag.',
                      instance_template='<task_description>{{task_description}}</task_description>\n'
                      '<initial_observation>{{initial_observation}}</initial_observation>')
        super().__init__(dict(task, variation_idx=0), prompt,
                         lambda **kwargs: ALFWorldEnvironment(task['game_file'], task['step_limit']))

    async def evaluate(self, **kwargs):
        return {'status': 'scored', 'method': 'alfworld_native_won', 'score': self.score,
                'success': self.score == 100.0, 'cost': 0.0, 'cost_complete': True, 'api_calls': 0}
