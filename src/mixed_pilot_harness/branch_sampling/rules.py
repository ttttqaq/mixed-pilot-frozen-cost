"""Key-step detection from past observations only; no paid judge calls."""
from dataclasses import dataclass


@dataclass
class Rules:
    error_window: int = 2
    repeat_window: int = 3
    min_branch_gap: int = 3

    def detect(self, steps):
        if len(steps) < self.min_branch_gap:
            return None
        markers = ('no known action', 'not a valid', 'cannot', "can't", 'must specify',
                   "don't see", 'not found', 'not recognized', 'unknown command', '"ok": false')
        last = steps[-self.error_window:]
        if len(last) == self.error_window and all(s['format_error'] or any(
            m in s.get('observation','').lower() for m in markers) for s in last):
            return {'rule':'consecutive_errors','window':self.error_window}
        last = steps[-self.repeat_window:]
        if len(last) == self.repeat_window and len({s.get('command') for s in last}) == 1 and last[0].get('command'):
            return {'rule':'repeated_command','window':self.repeat_window}
        return None
