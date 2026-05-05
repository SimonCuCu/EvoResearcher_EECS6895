"""DeepSeek client with structured-output helpers."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import httpx
from pydantic import BaseModel

from evoresearcher.config import AppConfig

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(self, config: AppConfig):
        self.config = config
        self._client = httpx.Client(timeout=90)

    def text(
        self,
        *,
        label: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> str:
        response = self._client.post(
            self.config.deepseek_base_url,
            headers={
                "Authorization": f"Bearer {self.config.deepseek_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.config.deepseek_model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"].strip()

    def structured(
        self,
        model: type[T],
        *,
        label: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> T:
        schema = json.dumps(model.model_json_schema(), indent=2)
        prompt = (
            f"{user_prompt}\n\n"
            "Return valid JSON only. Do not use markdown fences. "
            "Inside JSON strings, escape every literal backslash as \\\\; this is required for LaTeX too.\n"
            f"JSON schema:\n{schema}"
        )
        raw = self.text(
            label=label,
            system_prompt=system_prompt,
            user_prompt=prompt,
            temperature=temperature,
        )
        try:
            data = self._extract_json(raw)
        except json.JSONDecodeError:
            self._write_invalid_json(label=label, raw=raw)
            repaired = self._repair_json_with_model(
                label=label,
                schema=schema,
                raw=raw,
            )
            try:
                data = self._extract_json(repaired)
            except json.JSONDecodeError:
                self._write_invalid_json(label=f"{label}_repair", raw=repaired)
                raise
        return model.model_validate(data)

    def _extract_json(self, raw: str) -> dict:
        json_text = raw
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return json.loads(self._escape_invalid_backslashes(json_text))
            json_text = match.group(0)
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                return json.loads(self._escape_invalid_backslashes(json_text))

    def _escape_invalid_backslashes(self, json_text: str) -> str:
        return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", json_text)

    def _repair_json_with_model(self, *, label: str, schema: str, raw: str) -> str:
        return self.text(
            label=f"{label}_json_repair",
            system_prompt=(
                "You repair malformed JSON. Return valid JSON only, with no markdown fences and no commentary. "
                "Preserve the original content as much as possible. Escape quotes and backslashes correctly."
            ),
            user_prompt=(
                "The following model output was intended to match this JSON schema, but it is invalid JSON.\n\n"
                f"JSON schema:\n{schema}\n\n"
                "Malformed JSON:\n"
                f"{raw}\n\n"
                "Return the repaired JSON object only."
            ),
            temperature=0.0,
        )

    def _write_invalid_json(self, *, label: str, raw: str) -> None:
        try:
            debug_dir = Path(self.config.outputs_dir) / "_llm_failures"
            debug_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            safe_label = re.sub(r"[^a-zA-Z0-9_.-]+", "-", label)[:80]
            (debug_dir / f"{stamp}-{safe_label}.json.txt").write_text(raw)
        except Exception:
            pass
