"""Y and C supervision. Beta is deliberately absent from optimization."""
import torch
from torch import nn

from ..learning import LearningSystem

from . import METHOD


class SeparateLearner(LearningSystem):
    def train_router(self, rows, pairs=None):
        rows = self._rows(rows)
        if not rows:
            return {'samples': 0, 'pairs': len(pairs or []), 'updates': 0, 'loss': None}
        x = torch.stack([self.features.encode(r['state']) for r in rows]).to(self.device)
        actions = [self.model_to_idx[r['model']] for r in rows]
        y = torch.tensor([r['reward'] for r in rows], device=self.device)
        c = torch.tensor([r['cost'] / self.cost_reference for r in rows], device=self.device)
        w = torch.tensor([r['weight'] for r in rows], device=self.device)
        opt = self.optimization
        sums = dict(loss=0., reward_loss=0., cost_loss=0., regularization=0.)
        updates = 0
        self.router.train()
        for _ in range(opt['epochs']):
            for offset in range(0, len(rows), opt['batch_size']):
                terms, metrics = [], []
                for net in self.router:
                    idx = torch.randint(len(rows), (min(opt['batch_size'], len(rows)-offset),), device=self.device)
                    py, pc = net(x[idx], [actions[i] for i in idx.tolist()])
                    ly = self._weighted_loss(py, y[idx], w[idx])
                    lc = self._weighted_loss(pc, c[idx], w[idx])
                    reg = net.models.get_residual_l2_loss()
                    terms.append(ly + opt['cost_loss_weight'] * lc + reg)
                    metrics.append((ly, lc, reg))
                loss = torch.stack(terms).mean()
                self.router_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.router.parameters(), opt['grad_clip'])
                self.router_optimizer.step()
                sums['loss'] += loss.item()
                for i, key in enumerate(('reward_loss', 'cost_loss', 'regularization')):
                    sums[key] += torch.stack([m[i] for m in metrics]).mean().item()
                updates += 1
        self.router_updates += updates
        self.router.eval()
        return {**{k: v/updates for k, v in sums.items()}, 'samples': len(rows),
                'pairs': len(pairs or []), 'updates': updates, 'objective': 'separate_Y_C',
                'utility_loss': False}

    @classmethod
    def load(cls, path, config_override=None):
        result = super().load(path, config_override)
        if result.config.get('method') != METHOD:
            raise ValueError('Require a KSR Y/C checkpoint; old reference/option checkpoints cannot resume')
        return result
