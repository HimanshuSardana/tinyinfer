from pathlib import Path
from typing import Any

__version__ = "0.1.0"

from .model import download_model, inference, load_model

__all__ = ["__version__", "download_model", "inference", "load_model", "main"]


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: CLI args excluding prog name. None uses sys.argv.
    """
    print("Hello from tinyinfer!")
