from pathlib import Path

from evoresearcher.config import load_config


def test_load_config_uses_reasoning_model_default(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=test\n")

    config = load_config()

    assert config.deepseek_model == "deepseek-chat"
    assert config.deepseek_reasoning_model == "deepseek-v4-pro"


def test_load_config_reads_reasoning_model_override(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "DEEPSEEK_API_KEY=test\n"
        "DEEPSEEK_MODEL=deepseek-chat\n"
        "DEEPSEEK_REASONING_MODEL=custom-reasoner\n"
    )

    config = load_config()

    assert config.deepseek_reasoning_model == "custom-reasoner"


def test_load_config_accepts_explicit_model_overrides(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "DEEPSEEK_API_KEY=test\n"
        "DEEPSEEK_MODEL=deepseek-chat\n"
        "DEEPSEEK_REASONING_MODEL=deepseek-v4-pro\n"
    )

    config = load_config(
        deepseek_model="deepseek-v4-flash",
        deepseek_reasoning_model="deepseek-v4-flash",
    )

    assert config.deepseek_model == "deepseek-v4-flash"
    assert config.deepseek_reasoning_model == "deepseek-v4-flash"
