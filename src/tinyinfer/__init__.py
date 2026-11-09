from pathlib import Path
from typing import Any

__version__ = "0.1.0"


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: CLI args excluding prog name. None uses sys.argv.
    """
    print("Hello from tinyinfer!")


def download_model(model_id: str, cache_dir: str | Path | None = None) -> Path:
    """Download model snapshot from Hugging Face.

    Args:
        model_id: Hub repo id, e.g. "gpt2".
        cache_dir: Optional cache dir. None uses HF default.

    Returns:
        Local directory containing the snapshot.
    """
    pass


def load_model(model_path: str | Path) -> dict[str, Any]:
    """Load locally downloaded model.

    Args:
        model_path: Directory from download_model.

    Returns:
        Dict of weights/config consumed by inference.
    """
    pass


def inference(model: dict[str, Any], input_ids: list[int], max_new_tokens: int = 32) -> list[int]:
    """Run generation.

    Args:
        model: Loaded model from load_model.
        input_ids: Input token ids.
        max_new_tokens: Tokens to generate.

    Returns:
        Input + generated token ids.
    """
    pass
