"""Collect per-step terminal-return labels using random routing or KSR-brunch-1."""
from pathlib import Path
from ..inference import cli as inference


def parser():
    p = inference.parser()
    p.description = __doc__
    mode = next(a for a in p._actions if a.dest == 'mode')
    mode.choices = ['run','budget']
    mode.default = 'run'
    mode.help = 'run: traverse dataset for --runs rounds; budget: repeat until --budget'
    p.add_argument('--sampling-strategy', choices=['random','KSR-brunch-1','ksr-brunch-1','ksr-branch-1'], default='random')
    p.add_argument('--ksr-config',type=Path,default=Path('KSR_config.json'))
    return p


def prepare(args):
    # Keep infer's limits, retry, preflight and recovery controls in one parser.
    args.collect_steps = True
    if args.eval_only or args.reparse_only or args.precheck:
        raise ValueError('Use scripts/infer.py for evaluation maintenance')
    if args.mode == 'budget':
        args.stop_mode = 'budget-mode'
    if args.sampling_strategy == 'random':
        args.mode = 'random-router'
        args.router_profile = args.router_profile or 'random-router'
    else:
        if args.models or args.router_profile or args.router_possibility:
            raise ValueError('KSR model pool and starting models must be configured in --ksr-config')
        args.mode = 'ksr_branch1'
    return args


def main(argv=None):
    p = parser()
    try:
        args = prepare(p.parse_args(argv))
    except ValueError as exc:
        p.error(str(exc))
    return inference.run(args)
