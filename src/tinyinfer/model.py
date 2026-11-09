"""Model IO + generation for tinyinfer."""

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

_PRESPLIT = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
)

_tokenizer_cache: dict[str, Any] | None = None


def _bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(0xA1, 0xAC + 1))
        + list(range(0xAE, 0xFF + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


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


def forward(model: dict[str, Any], input_ids: list[int]) -> "np.ndarray":
    """Single forward pass, no cache.

    Args:
        model: Loaded model from load_model.
        input_ids: Token ids, length T.

    Returns:
        Logits of shape (T, vocab_size), float32.
    """
    cfg = model["config"]
    w = model["weights"]
    n_layer = int(cfg["n_layer"])
    n_head = int(cfg["n_head"])
    n_embd = int(cfg["n_embd"])
    n_positions = int(cfg.get("n_positions", cfg.get("n_ctx", 1024)))
    eps = float(cfg.get("layer_norm_epsilon", 1e-5))
    hd = n_embd // n_head
    t = len(input_ids)
    if t > n_positions:
        raise ValueError(f"sequence of {t} tokens exceeds n_positions={n_positions}")
    x = np.ascontiguousarray(w["wte.weight"][np.asarray(input_ids, dtype=np.int64)], dtype=np.float32)
    x = x + np.ascontiguousarray(w["wpe.weight"][:t], dtype=np.float32)
    pos = np.arange(t)
    mask = np.where(pos[None, :] <= pos[:, None], 0.0, -np.inf).astype(np.float32)
    for l in range(n_layer):
        p = f"h.{l}."
        h = _ln(x, w[p + "ln_1.weight"], w[p + "ln_1.bias"], eps)
        qkv = h @ w[p + "attn.c_attn.weight"] + w[p + "attn.c_attn.bias"]
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(t, n_head, hd).transpose(1, 0, 2)
        k = k.reshape(t, n_head, hd).transpose(1, 0, 2)
        v = v.reshape(t, n_head, hd).transpose(1, 0, 2)
        s = (q @ k.transpose(0, 2, 1)) / np.sqrt(float(hd)) + mask
        s = s - np.where(np.isfinite(s.max(axis=-1, keepdims=True)), s.max(axis=-1, keepdims=True), 0.0)
        e = np.exp(s)
        probs = e / e.sum(axis=-1, keepdims=True)
        attn = (probs @ v).transpose(1, 0, 2).reshape(t, n_embd) @ w[p + "attn.c_proj.weight"] + w[p + "attn.c_proj.bias"]
        x = x + attn
        h = _ln(x, w[p + "ln_2.weight"], w[p + "ln_2.bias"], eps)
        # tanh approximation from the GPT-2 spec, not exact erf gelu
        fc_in = h @ w[p + "mlp.c_fc.weight"] + w[p + "mlp.c_fc.bias"]
        fc_out = 0.5 * fc_in * (1.0 + np.tanh(0.7978845608028654 * (fc_in + 0.044715 * fc_in**3)))
        x = x + fc_out @ w[p + "mlp.c_proj.weight"] + w[p + "mlp.c_proj.bias"]
    x = _ln(x, w["ln_f.weight"], w["ln_f.bias"], eps)
    return (x @ w["wte.weight"].T).astype(np.float32)


def _ln(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    c = x - x.mean(axis=-1, keepdims=True)
    return c / np.sqrt(np.mean(c * c, axis=-1, keepdims=True) + eps) * weight + bias


def sample(logits: "np.ndarray", temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None, rng: "np.random.Generator | None" = None) -> int:
    """Draw one token id from logits."""
    rng = rng if rng is not None else np.random.default_rng()
    scores = np.array(logits, dtype=np.float64, copy=True).ravel()
    if temperature <= 0:
        return int(np.argmax(scores))
    scores /= temperature
    if top_k is not None and 0 < top_k < scores.size:
        cutoff = np.partition(scores, -top_k)[-top_k]
        scores = np.where(scores >= cutoff, scores, -np.inf)
    if top_p is not None and 0.0 < top_p < 1.0:
        order = np.argsort(-scores, kind="stable")
        cumulative = np.cumsum(_softmax(scores[order]))
        keep = int(np.searchsorted(cumulative, top_p)) + 1
        filtered = np.full_like(scores, -np.inf)
        filtered[order[:keep]] = scores[order[:keep]]
        scores = filtered
    probs = _softmax(scores)
    if not np.isfinite(probs).all() or probs.sum() <= 0:
        return int(np.argmax(logits))
    return int(rng.choice(probs.size, p=probs / probs.sum()))


def _softmax(x: np.ndarray) -> np.ndarray:
    m = np.where(np.isfinite(x.max()), x.max(), 0.0)
    e = np.exp(x - m)
    return e / np.where(e.sum() > 0, e.sum(), 1.0)


def inference(model: dict[str, Any], input_ids: list[int], max_new_tokens: int = 32, temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None, seed: int | None = None, eos_token_id: int | None = None, stop_on_eos: bool = True) -> list[int]:
    """Run generation.

    Args:
        model: Loaded model from load_model.
        input_ids: Input token ids.
        max_new_tokens: Tokens to generate.

    Returns:
        Input + generated token ids.
    """
    cfg = model["config"]
    n_positions = int(cfg.get("n_positions", cfg.get("n_ctx", 1024)))
    if eos_token_id is None:
        eos_token_id = cfg.get("eos_token_id")
    ids = list(input_ids)
    if not ids:
        if eos_token_id is None:
            raise ValueError("empty prompt needs an eos token to start from")
        ids = [eos_token_id]
    if len(ids) >= n_positions:
        keep = max(1, n_positions - max_new_tokens)
        ids = ids[-keep:]
    rng = np.random.default_rng(seed)
    for _ in range(max_new_tokens):
        row = forward(model, ids)[-1].astype(np.float64)
        token = sample(row, temperature=temperature, top_k=top_k, top_p=top_p, rng=rng)
        ids.append(token)
        if stop_on_eos and eos_token_id is not None and token == eos_token_id:
            break
        if len(ids) >= n_positions:
            break
    return ids

def tokenize(input: str) -> list[int]:
    """Encode text to GPT-2 token ids (byte-level BPE).

    Uses a module-cached tokenizer built from ``vocab.json``/``merges.txt``.
    Source dir, in order: ``$TINYINFER_MODEL_DIR`` if set, else a
    ``download_model("gpt2")`` snapshot. The cache means repeated calls
    don't re-read vocab.

    Args:
        input: Text to encode (empty string gives ``[]``).

    Returns:
        Token ids for ``input``.

    Raises:
        TypeError: If ``input`` is not a string.
        FileNotFoundError: If vocab/merges files are missing.
    """
    if not isinstance(input, str):
        raise TypeError(f"tokenize() expects str, got {type(input).__name__}")
    tok = _get_tokenizer()
    encoder: dict[str, int] = tok["encoder"]
    ranks: dict[str, int] = tok["ranks"]
    byte_encoder: dict[int, str] = tok["byte_encoder"]
    cache: dict[str, list[int]] = tok["cache"]
    ids: list[int] = []
    for piece in _PRESPLIT.findall(input):
        mapped = "".join(byte_encoder[b] for b in piece.encode("utf-8"))
        cached = cache.get(mapped)
        if cached is not None:
            ids.extend(cached)
            continue
        word = list(mapped)
        while len(word) > 1:
            best_rank = None
            best_pair = None
            for a, b in zip(word, word[1:]):
                r = ranks.get(f"{a} {b}")
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, (a, b)
            if best_pair is None:
                break
            merged: list[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    merged.append(word[i] + word[i + 1])
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged
        piece_ids = [encoder[t] for t in word]
        cache[mapped] = piece_ids
        ids.extend(piece_ids)
    return ids


def detokenize(input: list[int]) -> str:
    """Decode GPT-2 token ids back to text.

    Uses the same cached tokenizer as :func:`tokenize`. Unknown ids are
    skipped; byte sequences that form invalid UTF-8 decode with U+FFFD.

    Args:
        input: Token ids to decode.

    Returns:
        Decoded string.
    """
    tok = _get_tokenizer()
    decoder: dict[int, str] = tok["decoder"]
    byte_decoder: dict[str, int] = tok["byte_decoder"]
    raw = bytearray()
    for i in input:
        text = decoder.get(i)
        if text is None:
            continue
        for ch in text:
            b = byte_decoder.get(ch)
            if b is None:
                raw.extend(ch.encode("utf-8"))
            else:
                raw.append(b)
    return bytes(raw).decode("utf-8", errors="replace")


def _get_tokenizer() -> dict[str, Any]:
    """Return cached BPE tables, loading vocab/merges on first use."""
    global _tokenizer_cache
    if _tokenizer_cache is not None:
        return _tokenizer_cache
    env_dir = os.environ.get("TINYINFER_MODEL_DIR")
    base = Path(env_dir) if env_dir else download_model("gpt2")
    vocab_file = base / "vocab.json"
    merges_file = base / "merges.txt"
    if not vocab_file.is_file():
        raise FileNotFoundError(f"vocab.json not found in {base}")
    if not merges_file.is_file():
        raise FileNotFoundError(f"merges.txt not found in {base}")
    encoder = json.loads(vocab_file.read_text())
    decoder = {i: s for s, i in encoder.items()}
    if len(decoder) != len(encoder):
        raise ValueError("vocab.json contains duplicate ids")
    lines = merges_file.read_text().splitlines()
    if lines and lines[0].startswith("#version"):
        lines = lines[1:]
    ranks = {m: r for r, m in enumerate(l for l in lines if l.strip() and len(l.split()) == 2)}
    byte_encoder = _bytes_to_unicode()
    _tokenizer_cache = {
        "encoder": encoder,
        "decoder": decoder,
        "ranks": ranks,
        "byte_encoder": byte_encoder,
        "byte_decoder": {ch: b for b, ch in byte_encoder.items()},
        "cache": {},
        "dir": base,
    }
    return _tokenizer_cache

