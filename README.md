# tinyinfer

A tiny GPT-2 inference engine in pure NumPy. Loads real Hugging Face checkpoints, runs a hand-written transformer forward pass, keeps a KV cache, and streams tokens to a CLI, a Python API, or an HTTP endpoint with a bundled web UI.

No PyTorch, no ONNX, no C extensions. Just `numpy` and `safetensors`.

```bash
uv sync
uv run tinyinfer "The capital of France is"
```

## Features

- GPT-2 forward pass written from scratch in NumPy (layer norm, causal self-attention, GELU MLP, tied output matrix)
- KV cache so each generated token costs one step instead of a full re-run
- Byte-level BPE tokenizer parsed from the real `vocab.json` and `merges.txt`
- Sampling with temperature, top-k, top-p, greedy decoding, and seeded reproducibility
- Streaming generation: `generate()` yields token ids, an incremental decoder turns them into text without splitting UTF-8 sequences
- Stdlib-only HTTP server with Server-Sent Events, plus a single-file web UI in `web/index.html`
- Weights fetched from the Hugging Face Hub on first use and cached locally

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for the environment and lockfile
- A few hundred MB of disk and RAM for GPT-2 weights

## Install

```bash
git clone https://github.com/<you>/tinyinfer
cd tinyinfer
uv sync
```

The first run downloads the `gpt2` snapshot from the Hub. Set `TINYINFER_MODEL_DIR` to a local snapshot directory to skip the download, for example when offline or on a slow connection.

## Usage

### CLI

```bash
uv run tinyinfer "Once upon a time"
uv run tinyinfer "def fibonacci(n):" -n 60 --temperature 0.7 --top-k 40
uv run tinyinfer "The sky is" --greedy
uv run tinyinfer "hello" --model gpt2 --seed 42
```

Output streams to stdout as tokens arrive, so you see text before generation finishes.

### Python API

```python
from tinyinfer import download_model, load_model, tokenize, generate, detokenize

snapshot = download_model("gpt2")
model = load_model(snapshot)

ids = tokenize("The capital of France is")
for token_id in generate(model, ids, max_new_tokens=20, temperature=0.8, top_k=40, seed=0):
    print(detokenize([token_id]), end="", flush=True)
```

For streaming text without broken multi-byte characters, use the incremental decoder:

```python
from tinyinfer.model import _StreamDecoder

decoder = _StreamDecoder()
for token_id in generate(model, ids, max_new_tokens=20):
    print(decoder.push([token_id]), end="", flush=True)
print(decoder.flush())
```

### HTTP server

```bash
uv run server.py          # http://127.0.0.1:8765
uv run server.py 9000     # custom port
```

| Route | Description |
| --- | --- |
| `GET /` | Web UI |
| `GET /api/health` | Status, model name, whether weights are loaded, uptime |
| `POST /api/generate` | Generate text, streaming or as one JSON response |

Streaming request:

```bash
curl -sN http://127.0.0.1:8765/api/generate \
  -H 'content-type: application/json' \
  -d '{"prompt": "The capital of France is", "max_new_tokens": 32}'
```

Each SSE frame carries `{"token_id", "text", "n"}`, a final `{"done": true, "tokens", "elapsed_s"}`, then `data: [DONE]`. Send `"stream": false` to get `{"text", "tokens", "token_ids"}` instead.

Model weights load once at startup and a generation lock keeps concurrent requests from interleaving KV cache state.

## API

| Function | Purpose |
| --- | --- |
| `download_model(model_id, cache_dir=None)` | Snapshot a Hub repo, returns the local `Path` |
| `load_model(path)` | Read `config.json` and `*.safetensors` into config and weights |
| `forward(model, input_ids)` | Logits of shape `(T, vocab_size)` |
| `generate(model, ...)` | Generator yielding new token ids, reuses a KV cache |
| `inference(model, ...)` | Same, but returns the full id list at once |
| `sample(logits, ...)` | Draw one id with temperature, top-k, top-p |
| `tokenize(text)` / `detokenize(ids)` | GPT-2 byte-level BPE encode and decode |

`generate` and `inference` accept `max_new_tokens`, `temperature`, `top_k`, `top_p`, `seed`, `eos_token_id`, and `stop_on_eos`.

## Project layout

```
src/tinyinfer/
  __init__.py    CLI entry point (tinyinfer command)
  model.py       tokenizer, model IO, forward pass, sampling, generation
server.py        stdlib HTTP server with SSE streaming
web/index.html   single-file web UI
tests/           model IO and smoke tests
```

## Tests

```bash
uv run pytest
```

Tests cover the Hub download passthrough, safetensors loading with the shard glob fallback, and missing-file errors. They do not download weights.

## Notes and limitations

- Only GPT-2 architecture checkpoints are supported. `config.json` must describe a GPT-2 style model.
- Sharded safetensors are not merged yet: `load_model` reads the first shard and ignores the rest. GPT-2 small and medium ship as a single file.
- Attention uses the GPT-2 tanh approximation of GELU, which matches the reference implementation rather than the exact `erf` form.
- Generation is CPU-bound and single-sequence. Expect interactive but not fast throughput.

> [!NOTE]
> Long prompts are truncated from the left to fit `n_positions`, keeping room for `max_new_tokens`. Generation stops when the context window fills or an EOS token is produced.
