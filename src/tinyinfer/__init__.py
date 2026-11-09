__version__ = "0.1.0"

from .model import detokenize, download_model, forward, inference, load_model, sample, tokenize

__all__ = ["__version__", "detokenize", "download_model", "forward", "inference", "load_model", "main", "sample", "tokenize"]


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: CLI args excluding prog name. None uses sys.argv.
    """
    import argparse
    import os
    import sys

    from . import model as _m

    parser = argparse.ArgumentParser(prog="tinyinfer", description="Generate text with a GPT-2 checkpoint.")
    parser.add_argument("prompt", nargs="?", default="The capital of France is", help="prompt text")
    parser.add_argument("--model", default="gpt2", help="HF Hub model id (default: gpt2)")
    parser.add_argument("-n", "--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--greedy", action="store_true", help="argmax decoding")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if args.greedy:
        args.temperature = 0.0
    snapshot = download_model(args.model)
    if os.environ.get("TINYINFER_MODEL_DIR") != str(snapshot):
        os.environ["TINYINFER_MODEL_DIR"] = str(snapshot)
        _m._tokenizer_cache = None
    loaded = load_model(snapshot)
    ids = tokenize(args.prompt)
    out = inference(loaded, ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k or None, top_p=args.top_p, seed=args.seed)
    print(detokenize(out))
