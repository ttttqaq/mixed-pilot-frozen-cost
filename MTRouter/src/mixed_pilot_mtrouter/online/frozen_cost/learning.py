"""Keep the complete cold model for C; update a separate model using only Y."""
from copy import deepcopy
import hashlib

import torch
from torch import nn

from ..learning import LearningSystem, _finite
from ..ksr.learning import SeparateLearner
from . import METHOD


def tensor_fingerprint(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class FrozenCostLearner(LearningSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frozen_cost_router = None
        self.cost_fingerprint = None
        self.cold_initialization = None

    def freeze_cost(self):
        if self.frozen_cost_router is not None:
            return
        self.frozen_cost_router = deepcopy(self.router).eval()
        self.frozen_cost_router.requires_grad_(False)
        self.cost_fingerprint = tensor_fingerprint(self.frozen_cost_router.state_dict())
        # The live C layers are unused online; exclude even stale gradients.
        for net in self.router:
            net.cost.requires_grad_(False)
            for param in net.cost.parameters():
                param.grad = None

    def predict_actions(self, state):
        values = super().predict_actions(state)
        if self.frozen_cost_router is not None:
            self.frozen_cost_router.eval()
            with torch.no_grad():
                x = self.features.encode(state).to(self.device).unsqueeze(0).expand(len(self.model_names), -1)
                costs = torch.stack([net(x, list(range(len(self.model_names))))[1]
                                     for net in self.frozen_cost_router]) * self.cost_reference
                for i, name in enumerate(self.model_names):
                    values[name].update(cost=float(costs[:, i].mean()),
                                        cost_std=float(costs[:, i].std(unbiased=False)))
        return values

    def policy_fingerprint(self, epsilon=None):
        base = super().policy_fingerprint(epsilon)
        if self.frozen_cost_router is None:
            return base
        return 'policy-' + hashlib.sha256((base + ':frozen-cold-C:' + self.cost_fingerprint).encode()).hexdigest()

    def train_router(self, rows, pairs=None):
        if self.frozen_cost_router is None:
            return SeparateLearner.train_router(self, rows, pairs)
        # Online Y labels do not require a C field, even for Teacher evidence.
        valid = []
        for row in rows:
            if row.get('labels_valid') is False:
                continue
            if row['model'] not in self.model_to_idx:
                raise ValueError('Unknown router action '+str(row['model']))
            item = dict(row)
            item['reward'] = _finite(row['reward'], 'reward target', low=0, high=1)
            item['weight'] = _finite(row.get('weight', 1.), 'sample weight', low=0)
            if item['weight']:
                valid.append(item)
        limit = self.optimization['replay_limit']
        rows = valid[-limit:] if limit > 0 else valid
        if not rows:
            return {'samples': 0, 'pairs': len(pairs or []), 'updates': 0, 'loss': None,
                    'objective': 'Y_only_frozen_C', 'cost_loss': 0.}
        x = torch.stack([self.features.encode(r['state']) for r in rows]).to(self.device)
        actions = [self.model_to_idx[r['model']] for r in rows]
        y = torch.tensor([r['reward'] for r in rows], device=self.device)
        w = torch.tensor([r['weight'] for r in rows], device=self.device)
        sums = dict(loss=0., reward_loss=0., regularization=0.)
        updates, opt = 0, self.optimization
        self.router.train()
        for _ in range(opt['epochs']):
            for offset in range(0, len(rows), opt['batch_size']):
                terms, metrics = [], []
                for net in self.router:
                    idx = torch.randint(len(rows), (min(opt['batch_size'], len(rows)-offset),), device=self.device)
                    py, _ = net(x[idx], [actions[i] for i in idx.tolist()])
                    ly = self._weighted_loss(py, y[idx], w[idx])
                    reg = net.models.get_residual_l2_loss()
                    terms.append(ly + reg)
                    metrics.append((ly, reg))
                loss = torch.stack(terms).mean()
                self.router_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.router.parameters(), opt['grad_clip'])
                self.router_optimizer.step()
                sums['loss'] += loss.item()
                sums['reward_loss'] += torch.stack([m[0] for m in metrics]).mean().item()
                sums['regularization'] += torch.stack([m[1] for m in metrics]).mean().item()
                updates += 1
        self.router_updates += updates
        self.router.eval()
        return {**{k: v/updates for k, v in sums.items()}, 'samples': len(rows),
                'pairs': len(pairs or []), 'updates': updates, 'cost_loss': 0.,
                'objective': 'Y_only_frozen_C', 'utility_loss': False,
                'cost_model_fingerprint': self.cost_fingerprint,
                'cold_initialization': self.cold_initialization}

    def _checkpoint_extra(self):
        return {'frozen_cost_schema': METHOD,
                'frozen_cost_router': None if self.frozen_cost_router is None else self.frozen_cost_router.state_dict(),
                'cost_model_fingerprint': self.cost_fingerprint,
                'cold_initialization': self.cold_initialization}

    def _restore_checkpoint_extra(self, data):
        if data.get('frozen_cost_schema') != METHOD:
            raise ValueError('Checkpoint does not contain frozen-cost stage information')
        self.cold_initialization = data.get('cold_initialization')
        state = data['frozen_cost_router']
        if state is not None:
            self.freeze_cost()
            self.frozen_cost_router.load_state_dict(state)
            self.cost_fingerprint = tensor_fingerprint(state)
            if self.cost_fingerprint != data.get('cost_model_fingerprint'):
                raise ValueError('Frozen C checkpoint fingerprint mismatch')
        elif data.get('metadata', {}).get('cold_done') or data.get('metadata', {}).get('completed_iterations', 0):
            raise ValueError('Online checkpoint is missing its frozen C model')

    @classmethod
    def load(cls, path, config_override=None):
        obj = super().load(path, config_override)
        if obj.config.get('method') != METHOD:
            raise ValueError('Old runs require explicit cold-checkpoint initialization, not --resume')
        return obj
