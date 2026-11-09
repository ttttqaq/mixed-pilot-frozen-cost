"""CLI overrides and a persisted routing contract for dataset inference."""
from pathlib import Path
from ..router.profiles import inline_config, profile_config, snapshot


def configure(args, config):
    profile_name = args.router_profile
    if args.models and len(args.models)==1 and args.models[0] in config.profiles and config.profile(args.models[0]).kind=='router':
        if profile_name and profile_name != args.models[0]:
            raise ValueError('Conflicting router profiles')
        profile_name = args.models[0]
        args.models = None
    routed = args.mode in {'router','mtrouter','random-router'} or profile_name
    overrides = {'checkpoint':args.router_checkpoint, 'service_url':args.router_service_url,
        'device':args.router_device, 'model_names':args.router_model_names,
        'embedding_url':args.router_embedding_url, 'model_metadata':args.router_model_metadata}
    if getattr(args,'router_possibility',None) is not None:
        weights = {}
        for item in args.router_possibility:
            name, sep, value = item.partition('=')
            if not sep or not name or name in weights:
                raise ValueError('--router-possibility requires distinct PROFILE=WEIGHT entries')
            try:
                weights[name] = float(value)
            except ValueError as exc:
                raise ValueError('--router-possibility requires numeric weights') from exc
        overrides['possibility'] = weights
    if not routed:
        if any(v is not None for v in overrides.values()):
            raise ValueError('Router options require --mode router/mtrouter/random-router')
        return {}
    if args.mode in {'ksr','ksr-branch','random'}:
        raise ValueError('Use --mode router with --router-profile; KSR/random already select models')
    if profile_name:
        profile = config.profile(profile_name)
        if profile.kind != 'router': raise ValueError('--router-profile must reference kind=router')
        # Inline overrides can supply a missing/relocated checkpoint before validation.
        if profile.router_config:
            cfg = profile_config(profile, config.profiles, config.path.parent)
            if any(p.name not in config.profiles or p != config.profile(p.name) for p in cfg.tiers.values()):
                raise ValueError('infer router pools must reference unchanged harness profiles; move tier overrides into named profiles')
            raw = {k:v for k,v in snapshot(cfg,profile_name).items()
                   if k not in {'profile','checkpoint_sha256','model_metadata_sha256'}}
        else:
            raw = dict(profile.router)
    else:
        if args.mode == 'router': raise ValueError('--mode router requires --router-profile')
        raw = {'engine':'mtrouter' if args.mode=='mtrouter' else 'random'}
        profile_name = args.mode
    if args.models is not None: raw['models'] = list(args.models)
    if args.router_checkpoint is not None: raw.pop('service_url',None)
    if args.router_service_url is not None: raw.pop('checkpoint',None)
    for k,v in overrides.items():
        if v is not None:
            raw[k] = str(Path(v).expanduser().resolve()) if k in {'checkpoint','model_metadata'} else v
    raw['seed'] = args.seed
    cfg = inline_config(raw, config.profiles, config.path.parent)
    if args.mode=='mtrouter' and cfg.engine!='mtrouter' or args.mode=='random-router' and cfg.engine!='random':
        raise ValueError('Router mode and profile engine disagree')
    args.models = [p.name for p in cfg.tiers.values()]
    args.mode = 'router'
    return snapshot(cfg, profile_name)
