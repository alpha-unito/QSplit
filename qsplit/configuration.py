import argparse
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

import yaml

_settings = ContextVar("qsplit_configuration", default=None)


def current() -> dict:
    settings = _settings.get()
    if settings is None:
        settings = {}
        _settings.set(settings)
    return settings


def get(key, default=None):
    value = current().get(key)
    if value is None:
        value = default
    return str(value) if value is not None else None


def require(key):
    value = get(key)
    if value is None or value == "":
        raise ValueError(f"Missing QSplit configuration key: {key}")
    return value


def load(source) -> dict:
    if source is None:
        return {}
    if isinstance(source, Mapping):
        data = dict(source)
    elif isinstance(source, (str, Path)):
        try:
            data = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
        except yaml.YAMLError:
            raise ValueError("Invalid QSplit YAML configuration") from None
        if data is None:
            data = {}
    else:
        data = {}
        for item in source:
            data.update(load(item))
    if not isinstance(data, dict) or any(not isinstance(k, str) for k in data):
        raise ValueError("QSplit configuration must be a mapping with string keys")
    if any(v is not None and not isinstance(v, (str, int, float, bool)) for v in data.values()):
        raise ValueError("QSplit configuration values must be scalars")
    for key in ("CUT_DIM", "LOCAL_TRAINING_QUBITS"):
        if key not in data:
            continue
        try:
            cut = int(data[key])
            valid = not isinstance(data[key], bool) and cut > 0 and float(data[key]) == cut
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError(f"{key} must be a positive integer")
        data[key] = cut
    return data


@contextmanager
def use(source):
    token = _settings.set(load(source))
    try:
        yield current()
    finally:
        _settings.reset(token)


def configured(fn):
    @wraps(fn)
    def wrapped(*args, config=None, **kwargs):
        if config is None:
            return fn(*args, **kwargs)
        with use(config):
            return fn(*args, **kwargs)

    return wrapped


def cli(fn):
    @wraps(fn)
    def wrapped():
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        parser.add_argument("--config", action="append", default=[])
        args, remaining = parser.parse_known_args()
        argv = sys.argv
        try:
            sys.argv = [argv[0], *remaining]
            if args.config:
                with use(args.config):
                    return fn()
            return fn()
        finally:
            sys.argv = argv

    return wrapped
