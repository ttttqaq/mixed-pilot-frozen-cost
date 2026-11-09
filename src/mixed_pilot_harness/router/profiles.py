"""Resolve profile pools without loading torch, encoders or calling a service."""
from dataclasses import replace
import hashlib
import math
from pathlib import Path
from urllib.parse import urlsplit

from .config import RouterConfig, RouterConfigError, load_router_config


def inline_config(raw, profiles, config_dir):
    allowed = {'engine','models','checkpoint','service_url','service_timeout','device',
               'seed','model_names','embedding_url','model_metadata','possibility'}
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise RouterConfigError('Unsupported inline router configuration fields')
    engine = raw.get('engine')
    if engine not in {'random','mtrouter'}:
        raise RouterConfigError('Inline router engine must be random or mtrouter')
    for key in ('checkpoint','model_metadata','service_url','embedding_url','device'):
        if key in raw and not isinstance(raw[key],str):
            raise RouterConfigError(f'Router {key} must be a string')
    pool = raw.get('models')
    if not isinstance(pool, list) or len(pool) < 2 or any(not isinstance(p,str) for p in pool) or len(set(pool)) != len(pool):
        raise RouterConfigError('Router models must contain at least two distinct profile names')
    tiers = {}
    for name in pool:
        if name not in profiles or profiles[name].kind == 'router':
            raise RouterConfigError(f'Router pool must reference concrete profiles: {name}')
        tiers[name] = replace(profiles[name], name=name)
    possibility = {}
    if 'possibility' in raw:
        if engine != 'random':
            raise RouterConfigError('possibility is only supported by RandomRouter')
        weights = raw['possibility']
        if not isinstance(weights, dict) or set(weights) != set(pool):
            raise RouterConfigError('possibility must specify exactly every profile in models')
        try:
            invalid = any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v < 0
                          for v in weights.values())
        except OverflowError:
            invalid = True
        if invalid or max(weights.values()) <= 0:
            raise RouterConfigError('possibility values must be finite nonnegative numbers with positive total')
        # Store relative weights unchanged for stable snapshot round trips.
        # Uniform settings retain the old RNG sequence and checkpoint schema.
        if len(set(weights.values())) > 1:
            possibility = {name: weights[name] for name in pool}
    def path(key):
        if not raw.get(key): return None
        p = Path(raw[key]).expanduser()
        return (p if p.is_absolute() else Path(config_dir)/p).resolve()
    checkpoint, metadata = path('checkpoint'), path('model_metadata')
    service = raw.get('service_url','').rstrip('/')
    timeout = raw.get('service_timeout',120.0)
    if isinstance(timeout,bool) or not isinstance(timeout,(float,int)) or not math.isfinite(timeout) or timeout <= 0:
        raise RouterConfigError('Router service_timeout must be finite and positive')
    if type(raw.get('seed',42)) is not int:
        raise RouterConfigError('Router seed must be an integer')
    names = raw.get('model_names', [p.model for p in tiers.values()])
    if not isinstance(names,(list,tuple)) or len(names)!=len(pool) or any(not isinstance(n,str) or not n for n in names):
        raise RouterConfigError('Router model_names must map one checkpoint name to each pool profile')
    if engine == 'mtrouter':
        if len(set(names)) != len(names):
            raise RouterConfigError('MTRouter model_names must be distinct')
        if bool(checkpoint) == bool(service):
            raise RouterConfigError('MTRouter requires exactly one of checkpoint or service_url')
        if checkpoint and not checkpoint.is_file():
            raise RouterConfigError(f'MTRouter checkpoint not found: {checkpoint}')
        if metadata and not metadata.is_file():
            raise RouterConfigError(f'MTRouter model metadata not found: {metadata}')
        if service:
            url = urlsplit(service)
            if url.scheme not in {'http','https'} or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise RouterConfigError('MTRouter service_url must be an HTTP(S) URL without credentials/query/fragment')
    elif checkpoint or service or metadata or raw.get('embedding_url'):
        raise RouterConfigError('Random router does not accept MTRouter runtime options')
    if raw.get('device','cpu') not in {'cpu','cuda','mps'}:
        raise RouterConfigError('Router device must be cpu, cuda or mps')
    return RouterConfig(path=Path(config_dir)/'harness.json', engine=engine, tiers=tiers,
        checkpoint=checkpoint, service_url=service, service_timeout=timeout,
        device=raw.get('device','cpu'), seed=raw.get('seed',42), model_names=tuple(names),
        embedding_url=raw.get('embedding_url',''), model_metadata=metadata, possibility=possibility)


def profile_config(profile, profiles, config_dir):
    if profile.router_config and profile.router:
        raise RouterConfigError('Use either router_config or inline router, not both')
    if profile.router_config:
        path = Path(profile.router_config).expanduser()
        return load_router_config(path if path.is_absolute() else Path(config_dir)/path, profiles)
    return inline_config(profile.router, profiles, config_dir)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def snapshot(config, profile_name):
    """Only routing settings and file hashes; credentials stay in providers."""
    return {'profile':profile_name, 'engine':config.engine,
        'models':[p.name for p in config.tiers.values()],
        'model_names':list(config.model_names or tuple(p.model for p in config.tiers.values())),
        'checkpoint':str(config.checkpoint) if config.checkpoint else '',
        'checkpoint_sha256':file_hash(config.checkpoint) if config.checkpoint else None,
        'model_metadata':str(config.model_metadata) if config.model_metadata else '',
        'model_metadata_sha256':file_hash(config.model_metadata) if config.model_metadata else None,
        'service_url':config.service_url, 'service_timeout':config.service_timeout,
        'device':config.device, 'seed':config.seed, 'embedding_url':config.embedding_url,
        **({'possibility':dict(config.possibility)} if config.possibility else {})}


def from_snapshot(value, profiles):
    return inline_config({k:v for k,v in value.items()
        if k not in {'profile','checkpoint_sha256','model_metadata_sha256'}}, profiles, Path.cwd())
