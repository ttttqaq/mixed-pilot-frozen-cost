"""Bounded retries; unavailable Teacher evidence falls back to real rollouts."""
import asyncio
import time

from ..runtime import InvalidTeacherEvidenceError, RecoverableRequestError
from ..teacher import FixedReferenceTeacher, append_jsonl


class ResilientReferenceTeacher(FixedReferenceTeacher):
    max_attempts = 3

    def begin_phase(self):
        super().begin_phase()
        self.transport_failed_in_phase = False

    async def get(self, state, policy_context):
        # Keep FixedReferenceTeacher's split guard even on the retry path.
        if state.get('split') != 'train' or str(state.get('task_id', '')).startswith(('valid--', 'test--')):
            raise ValueError('Fixed Teacher may only assess train states')
        if not self.config['teacher']['enabled']:
            return None
        key = self.key(state, policy_context)
        cached = self.cache.get(key)
        if getattr(self, 'transport_failed_in_phase', False) and (cached is None or cached.get('error')):
            append_jsonl(self.output/'teacher_retries.jsonl', {
                'key': key, 'action': 'real_rollout_fallback',
                'reason': 'phase_transport_circuit_open', 'request_sent': False,
                'policy_version': state['policy_version'], 'time': time.time()})
            return None
        for attempt in range(self.max_attempts):
            previous = set(self.runner.ledger.records)
            try:
                return await super().get(state, policy_context)
            except InvalidTeacherEvidenceError as exc:
                # Only invalidate the in-memory error entry. The original
                # replies, invalid estimates, valid estimates and fees stay on disk.
                cached = self.cache.get(key)
                if cached and cached.get('error'):
                    self.cache.pop(key)
                exhausted = attempt + 1 == self.max_attempts
                append_jsonl(self.output/'teacher_retries.jsonl', {
                    'key': key, 'attempt': attempt + 1, 'error': str(exc),
                    'call_id': getattr(exc, 'call_id', None),
                    'fee_reserved': bool(getattr(exc, 'fee_reserved', False)),
                    'action': 'real_rollout_fallback' if exhausted else 'retry_teacher',
                    'policy_version': state['policy_version'], 'time': time.time()})
                if exhausted:
                    # None already makes KSRTrainer audit the actual continuation;
                    # it is never converted to a synthetic zero/success target.
                    return None
                delay = min(30., float(self.config.get('resilience', {}).get('retry_delay', 0.)))
                if delay:
                    await asyncio.sleep(delay)
            except RecoverableRequestError as exc:
                # Runner already exhausted/classified transport retries and
                # bounded any uncertain fees. An unavailable assessor is missing
                # evidence; real Actor continuations can still provide labels.
                self.runner.ledger.check()
                self.transport_failed_in_phase = True
                append_jsonl(self.output/'teacher_retries.jsonl', {
                    'key': key, 'action': 'real_rollout_fallback',
                    'reason': 'teacher_transport_unavailable', 'error': str(exc),
                    'call_ids': sorted(set(self.runner.ledger.records)-previous),
                    'request_sent': bool(set(self.runner.ledger.records)-previous),
                    'policy_version': state['policy_version'], 'time': time.time()})
                return None
