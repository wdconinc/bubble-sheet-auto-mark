"""Bubble Sheet Auto-Mark package."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("bubble-sheet-auto-mark")
except PackageNotFoundError:
    __version__ = "0.0.0"
