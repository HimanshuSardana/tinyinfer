from pathlib import Path


def _make_snapshot(d: Path, config=None, weights=None):
    import json

    import numpy as np
    from safetensors.numpy import save_file

    (d / "config.json").write_text(json.dumps(config or {"n_layer": 2}))
    save_file(weights or {"w": np.zeros((2, 2))}, str(d / "model.safetensors"))
    return d


def test_download_model_passes_through(tmp_path, monkeypatch):
    import huggingface_hub

    from tinyinfer.model import download_model

    seen = {}

    def fake_snapshot_download(repo_id=None, cache_dir=None, allow_patterns=None):
        seen.update(repo_id=repo_id, cache_dir=cache_dir, allow_patterns=allow_patterns)
        return str(tmp_path / "snap")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    out = download_model("gpt2", cache_dir=tmp_path)
    assert out == tmp_path / "snap"
    assert seen["repo_id"] == "gpt2"
    assert seen["cache_dir"] == str(tmp_path)
    assert "config.json" in seen["allow_patterns"]


def test_download_model_default_cache_dir(tmp_path, monkeypatch):
    import huggingface_hub

    from tinyinfer.model import download_model

    seen = {}

    def fake_snapshot_download(repo_id=None, cache_dir=None, allow_patterns=None):
        seen["cache_dir"] = cache_dir
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    download_model("gpt2")
    assert seen["cache_dir"] is None


def test_load_model_ok(tmp_path):
    from tinyinfer.model import load_model

    _make_snapshot(tmp_path)
    m = load_model(tmp_path)
    assert m["config"] == {"n_layer": 2}
    assert list(m["weights"]) == ["w"]
    assert m["path"] == tmp_path


def test_load_model_fallback_glob(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file

    from tinyinfer.model import load_model

    (tmp_path / "config.json").write_text("{}")
    save_file({"a": np.ones(3)}, str(tmp_path / "model-00001-of-00002.safetensors"))
    m = load_model(tmp_path)
    assert list(m["weights"]) == ["a"]


def test_load_model_missing(tmp_path):
    import pytest

    from tinyinfer.model import load_model

    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope")

    d = tmp_path / "no-config"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        load_model(d)

    e = tmp_path / "no-weights"
    e.mkdir()
    (e / "config.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        load_model(e)
