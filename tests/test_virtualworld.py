"""Regression checks for mixed dataset routing, identity isolation and resume hashes."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from mixed_pilot_harness.inference.dataset import load_dataset
from mixed_pilot_harness.inference.virtualworld import ALFWorldEpisode
from mixed_pilot_harness.collection.collector import load_prompt_config
from mixed_pilot_harness.types import ProviderResponse


class VirtualWorldTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def task(self, task_id='alf-one', split='train', source='train/group/trial'):
        directory = self.root / 'tasks' / task_id
        (directory / 'workspace').mkdir(parents=True)
        (directory / 'workspace/game.tw-pddl').write_text('{}')
        (directory / 'prompt.md').write_text('Public household objective')
        manifest = {'id': task_id, 'environment': {'benchmark': 'alfworld',
                      'task_type': 'pick_and_place_simple', 'step_limit': 200,
                      'runtime': {'textworld': '1.7.0'}},
                    'workspace': {'path': 'workspace'},
                    'source': {'dataset': 'ALFWorld', 'split': 'train', 'upstream_id': source,
                               'revision': '0.4.2'}, 'routing_metadata': {'router_split': split}}
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        return directory

    def test_load_uses_alf_adapter_and_fingerprints_changes(self):
        directory = self.task()
        first = load_dataset(self.root, 'train')[0]
        self.assertEqual(first['benchmark'], 'alfworld')
        self.assertEqual(first['task_id'], 'train--alf-one')
        self.assertNotIn('reference_path', first)
        (directory / 'workspace/game.tw-pddl').write_text('{"changed":true}')
        second = load_dataset(self.root, 'train')[0]
        self.assertNotEqual(first['content_hashes'], second['content_hashes'])

    def test_duplicate_instance_across_splits_rejected(self):
        self.task()
        self.task('alf-two', 'test')
        with self.assertRaisesRegex(ValueError, 'Duplicate native instance'):
            load_dataset(self.root, 'all')

    def test_game_path_escape_rejected(self):
        directory = self.task()
        p = directory / 'manifest.json'; m = json.loads(p.read_text())
        m['environment']['game_file'] = '../../../outside.tw-pddl'
        p.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, 'escapes'):
            load_dataset(self.root, 'train')

    def test_submission_does_not_claim_native_win(self):
        task = {'task_name': 'pick_and_place_simple',
                'game_file': 'unused', 'step_limit': 200}
        episode = ALFWorldEpisode(task, load_prompt_config())
        result = asyncio.run(episode.advance(ProviderResponse(text='```text\ntask completed\n```')))
        self.assertTrue(result.done)
        score = asyncio.run(episode.evaluate())
        self.assertFalse(score['success'])
        self.assertEqual(score['score'], 0)
        self.assertNotIn('science experiment', episode.system)


if __name__ == '__main__': unittest.main()
