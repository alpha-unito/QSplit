from __future__ import annotations

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
try:
    from .version import __version__
except Exception:
    __version__ = "0"
