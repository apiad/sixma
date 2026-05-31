from importlib.metadata import PackageNotFoundError, version

from .core import PickContext, certify, require

try:
    __version__ = version("sixma")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"


__all__ = ["PickContext", "certify", "require"]
