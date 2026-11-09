"""Standalone native ALFWorld PDDL adapter, shared with exported task workspaces."""
from __future__ import annotations

class ALFWorldEnvironment:
    def __init__(self, game_file, step_limit=200):
        self.game_file = str(game_file)
        self.step_limit = step_limit
        self.env = None
        self.score = 0.0
        self.done = False
        self.moves = 0
        self.state = None

    def load_task(self, *_):
        self.close()
        import os
        from pathlib import Path
        # ALFWorld's import creates its cache; keep that side effect beside the game.
        os.environ.setdefault('ALFWORLD_DATA', str(Path(self.game_file).resolve().parent))
        import textworld
        from textworld.envs.pddl import PddlEnv
        from alfworld.agents.utils.misc import Demangler
        from importlib.metadata import version
        if (version('textworld') != '1.7.0' or version('alfworld') != '0.4.2'
                or version('fast-downward-textworld') != '20.6.4'):
            raise RuntimeError('VirtualWorld requires textworld==1.7.0, alfworld==0.4.2 and fast-downward-textworld==20.6.4')

        class Demangle(textworld.core.Wrapper):
            def load(self, gamefile):
                super().load(gamefile)
                demangler = Demangler(game_infos=self._entity_infos, shuffle=False)
                for info in self._entity_infos.values():
                    info.name = demangler.demangle_alfred_name(info.id)

        # This is the official ALFWorld demangling protocol with domain randomization off.
        self.env = Demangle(PddlEnv(textworld.EnvInfos(won=True, admissible_commands=True)))
        try:
            self.env.load(self.game_file)
            self.state = self.env.reset()
            self.score, self.done, self.moves = 0.0, False, 0
            objective = self.state.get('objective') or self.state.feedback.rpartition('Your task is to:')[-1].strip()
            return {'observation': self.state.feedback, 'task_description': objective,
                    'valid_actions': list(self.state.admissible_commands)}
        except BaseException:
            self.close()
            raise

    def step(self, command):
        if self.env is None:
            raise RuntimeError('Environment has not been loaded')
        if command.startswith('?'):
            return {'observation': '\n'.join(self.state.admissible_commands), 'reward': 0.0,
                    'score': self.score, 'done': self.done}
        if self.done:
            raise RuntimeError('Episode is already done')
        self.state, _, done = self.env.step(command)
        self.moves += 1
        # Reward and textual claims are insufficient: only the native won flag passes.
        self.score = 100.0 if self.state.won else 0.0
        self.done = bool(done or self.state.won or self.moves >= self.step_limit)
        return {'observation': self.state.feedback, 'reward': self.score,
                'score': self.score, 'done': self.done, 'won': bool(self.state.won),
                'valid_actions': list(self.state.admissible_commands)}

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
