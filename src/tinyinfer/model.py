"""Model IO + generation for tinyinfer."""

from pathlib import Path
from typing import Any


def download_model(model_id: str, cache_dir: str | Path | None = None) -> Path:
    """Download model snapshot from Hugging Face.

    Args:
        model_id: Hub repo id, e.g. "gpt2".
        cache_dir: Optional cache dir. None uses HF default.

    Returns:
        Local directory containing the snapshot.
    """
    from huggingface_hub import snapshot_download

    allow = ["config.json", "*.safetensors", "*.json", "*.txt"]
    snap = snapshot_download(
        repo_id=model_id,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        allow_patterns=allow,
    )
    return Path(snap)


def load_model(model_path: str | Path) -> dict[str, Any]:
    """Load locally downloaded model.

    Args:
        model_path: Directory from download_model.

    Returns:
        Dict of weights/config consumed by inference.
    """
    import json

    from safetensors.numpy import load_file

    path = Path(model_path)
    if not path.is_dir():
        raise FileNotFoundError(f"model dir not found: {path}")
    cfg_file = path / "config.json"
    if not cfg_file.is_file():
        raise FileNotFoundError(f"config.json not found in {path}")
    config = json.loads(cfg_file.read_text())
    st_file = path / "model.safetensors"
    if not st_file.is_file():
        cands = sorted(path.glob("*.safetensors"))
        if not cands:
            raise FileNotFoundError(f"no .safetensors weights in {path}")
        # TODO: merge all shards (respect model.safetensors.index.json); cands[0] drops the rest.
        st_file = cands[0]
    weights = load_file(str(st_file))
    return {"config": config, "weights": weights, "path": path}


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
