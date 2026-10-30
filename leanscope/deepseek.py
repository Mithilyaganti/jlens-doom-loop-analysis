"""DeepSeek V4.1 Flash client.

The API model name is ``deepseek-flash`` (served by DeepSeek-V4.1-Flash).
The key is read from ``DEEPSEEK_API_KEY`` and is never logged.

Tests use ``FakeScenarioDrafter`` and ``FakeVoicedJudge``. They must not call
``DeepSeekClient.complete`` without an injected transport, and they must not
call ``_http_post``.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Callable

# Current Flash API name. DeepSeek-V4.1-Flash, including the 1017 refresh,
# is served under this name (https://api-docs.deepseek.com/).
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com/chat/completions"

Transport = Callable[[str, str, dict[str, Any]], dict[str, Any]]


class DeepSeekClient:
    """OpenAI-compatible chat client. Network only happens in ``complete``."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        transport: Transport | None = None,
        timeout: float = 120.0,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.api_key = api_key
        self.model = model or os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL)
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
        self._transport = transport
        self.timeout = float(timeout)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        if not self.api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Refusing to call DeepSeek."
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if self._transport is not None:
            data = self._transport(self.base_url, self.api_key, payload)
        else:
            data = _http_post(
                self.base_url, self.api_key, payload, timeout=self.timeout
            )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("DeepSeek response did not contain a message") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("DeepSeek returned an empty message")
        return content


def _http_post(
    url: str,
    api_key: str,
    payload: dict[str, Any],
    *,
    timeout: float,
) -> dict[str, Any]:
    """POST one chat completion. Not used when a transport is injected."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("DeepSeek response was not a JSON object")
    return parsed
