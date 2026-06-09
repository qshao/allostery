"""Allostery package."""

from importlib.metadata import PackageNotFoundError as _PNF, version as _pkg_version

try:
    __version__: str = _pkg_version("allostery")
except _PNF:
    __version__ = "unknown"

__all__: list[str] = ["__version__"]
