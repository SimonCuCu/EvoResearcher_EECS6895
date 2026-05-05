from evoresearcher.llm import LLMClient


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
