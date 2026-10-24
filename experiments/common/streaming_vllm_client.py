from __future__ import annotations

import json
import logging
import random
import time
from typing import Any

import requests
from terrarium.llm.clients.vllm_client import VLLMClient, _convert_tools

logger = logging.getLogger(__name__)
# terrarium.compaction.compactor sets the root logger to DEBUG on import, which
# makes urllib3 log every connection; Terrarium quiets httpx the same way.
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Statuses worth another attempt; any other non-2xx fails the run as before.
_RETRY_STATUSES = {408, 429, 500, 502, 503, 504}


class _TransientError(Exception):
    pass


class StreamingVLLMClient(VLLMClient):
    """Terrarium's VLLMClient, but streaming and retrying.

    A non-streaming request sends nothing back for minutes while a long prompt is
    served, and the Modal tunnel (or WSL2's NAT) can drop such a silent connection
    without the client noticing; the thread then waits out the whole timeout.
    Streaming sends a chunk per token, so `request_timeout` becomes the longest
    allowed gap between chunks rather than the length of the whole request, and a
    dropped connection surfaces as a timeout that is retried.

    The chunks are folded back into the non-streaming response shape, so
    `process_tool_calls` and `get_usage` work unchanged.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "EMPTY",
        request_timeout: int = 60,
        max_retries: int = 5,
        connect_timeout: float = 30.0,
    ):
        super().__init__(base_url, model_name, api_key, request_timeout)
        self.max_retries = max_retries
        self.connect_timeout = connect_timeout

    def generate_response(
        self,
        input: list[dict[str, Any]],
        params: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        payload: dict[str, Any] = {
            "model": params.get("model", self.model_name),
            "messages": input,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        max_tokens = params.get("max_completion_tokens") or params.get(
            "max_output_tokens"
        )
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if params.get("temperature") is not None:
            payload["temperature"] = params["temperature"]
        converted_tools = _convert_tools(params.get("tools"))
        if converted_tools:
            payload["tools"] = converted_tools
        body = json.dumps(payload)

        for attempt in range(self.max_retries + 1):
            try:
                data = self._stream_once(body)
                break
            except (requests.ConnectionError, requests.Timeout, _TransientError) as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"vLLM chat request failed after {attempt + 1} attempts: {exc}"
                    ) from exc
                delay = min(60.0, 2.0**attempt) + random.uniform(0, 1)
                logger.warning(
                    "vLLM request attempt %s/%s failed (%s); retrying in %.1fs",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)

        choices = data.get("choices") or []
        first_message = choices[0]["message"] if choices else {"content": ""}
        return data, self._extract_message_content(first_message)

    def _stream_once(self, body: str) -> dict[str, Any]:
        with self.session.post(
            f"{self.base_url}/chat/completions",
            headers=self._build_headers(),
            data=body,
            stream=True,
            timeout=(self.connect_timeout, self.request_timeout),
        ) as response:
            if response.status_code in _RETRY_STATUSES:
                raise _TransientError(
                    f"HTTP {response.status_code}: {response.text[:500]}"
                )
            if not response.ok:
                raise RuntimeError(
                    f"vLLM chat request failed ({response.status_code}): {response.text}"
                )
            return self._assemble(response)

    @staticmethod
    def _assemble(response: requests.Response) -> dict[str, Any]:
        """Fold SSE chunks into one chat.completion response."""
        meta: dict[str, Any] = {}
        text_fields: dict[str, list[str]] = {}
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason = None
        usage = None
        done = False

        for line in response.iter_lines():
            if not line.startswith(b"data:"):
                continue
            chunk_raw = line[len(b"data:") :].strip()
            if chunk_raw == b"[DONE]":
                done = True
                break
            chunk = json.loads(chunk_raw)
            if "error" in chunk:
                raise _TransientError(f"stream error: {chunk['error']}")
            for key in ("id", "created", "model"):
                if key in chunk:
                    meta.setdefault(key, chunk[key])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                for key, value in delta.items():
                    # content, plus reasoning / reasoning_content when a parser is on.
                    if key not in ("role", "tool_calls") and isinstance(value, str):
                        text_fields.setdefault(key, []).append(value)
                for call_delta in delta.get("tool_calls") or []:
                    call = tool_calls.setdefault(
                        call_delta.get("index", 0),
                        {
                            "id": None,
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if call_delta.get("id"):
                        call["id"] = call_delta["id"]
                    if call_delta.get("type"):
                        call["type"] = call_delta["type"]
                    function_delta = call_delta.get("function") or {}
                    call["function"]["name"] += function_delta.get("name") or ""
                    call["function"]["arguments"] += (
                        function_delta.get("arguments") or ""
                    )

        if not done:
            raise _TransientError("stream ended before [DONE]")

        message: dict[str, Any] = {"role": "assistant", "content": None}
        for key, parts in text_fields.items():
            message[key] = "".join(parts)
        if tool_calls:
            message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
        data: dict[str, Any] = {
            **meta,
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": message, "finish_reason": finish_reason}
            ],
        }
        if usage:
            data["usage"] = usage
        return data
