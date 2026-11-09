"""TensorBoard is optional; upstream JSON training logs remain enabled."""
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:
        def __init__(self, **kwargs):
            print('TensorBoard unavailable; using the JSON training log only.')
        def add_scalar(self, *args, **kwargs): pass
        def add_histogram(self, *args, **kwargs): pass
        def add_text(self, *args, **kwargs): pass
        def close(self): pass
