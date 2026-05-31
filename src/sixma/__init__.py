from importlib.metadata import PackageNotFoundError, version

from .core import certify, require

try:
    __version__ = version("sixma")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"


__all__ = ["certify", "require"]
