from pathlib import Path

from evoresearcher.config import AppConfig
from evoresearcher.llm import LLMClient


class FakeResponse:
    def __init__(self, content: str = "ok"):
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


class RecordingHTTPClient:
    def __init__(self):
        self.payloads = []

    def post(self, url, *, headers, json):
        self.payloads.append(json)
        return FakeResponse()


def build_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        workspace_dir=tmp_path,
        outputs_dir=tmp_path / "outputs",
        memory_dir=tmp_path / "memory",
        author_line="Test",
        deepseek_api_key="test-key",
        deepseek_model="deepseek-chat",
        deepseek_base_url="https://example.com/chat",
        deepseek_reasoning_model="deepseek-v4-pro",
    )


def test_extract_json_repairs_latex_backslashes_without_touching_valid_escapes():
    client = LLMClient.__new__(LLMClient)

    data = client._extract_json(
        '{"title": "Sparse gate", "body": "Use \\alpha in $x \\in X$ and keep\\nnewlines."}'
    )

    assert data["title"] == "Sparse gate"
    assert data["body"] == "Use \\alpha in $x \\in X$ and keep\nnewlines."


def test_escape_invalid_backslashes_preserves_valid_json_escapes():
    client = LLMClient.__new__(LLMClient)

    repaired = client._escape_invalid_backslashes('"line\\n latex \\alpha slash\\/"')

    assert repaired == '"line\\n latex \\\\alpha slash\\/"'


def test_text_uses_default_model_without_override(tmp_path):
    client = LLMClient(build_config(tmp_path))
    http = RecordingHTTPClient()
    client._client = http

    client.text(label="test", system_prompt="system", user_prompt="user")

    assert http.payloads[0]["model"] == "deepseek-chat"


def test_text_uses_model_override_when_provided(tmp_path):
    client = LLMClient(build_config(tmp_path))
    http = RecordingHTTPClient()
    client._client = http

    client.text(
        label="test",
        system_prompt="system",
        user_prompt="user",
        model_override="deepseek-v4-pro",
    )

    assert http.payloads[0]["model"] == "deepseek-v4-pro"
