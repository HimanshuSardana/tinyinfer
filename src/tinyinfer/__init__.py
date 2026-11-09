__version__ = "0.1.0"

from .model import detokenize, download_model, forward, inference, load_model, tokenize

__all__ = ["__version__", "detokenize", "download_model", "forward", "inference", "load_model", "main", "tokenize"]


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: CLI args excluding prog name. None uses sys.argv.
    """
    print("Hello from tinyinfer!")
