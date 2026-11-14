"""Anthropic (Claude API key) provider.

Set ``ANTHROPIC_API_KEY`` (and optionally ``LLM_MODEL``; the default is
``claude-opus-5-5``). Uses the official ``anthropic`` SDK.

Notes on the current Claude models this targets:
* ``temperature`` is rejected on current models, so it is accepted and ignored.
* Thinking is adaptive and always on for Opus 5.5; depth is set with
  ``output_config.effort`` (``ANTHROPIC_EFFORT``, default ``medium``).
* Server-side refusal fallback is enabled (``fallbacks: "default"``), so a
  rare safety decline is retried on a fallback model inside the same call
  instead of failing the job. Set ``ANTHROPIC_FALLBACKS=0`` to turn it off.
"""

from __future__ import annotations

import json
import logging
import os

import anthropic

from applypilot.providers.base import AgentResult, EventCallback, ToolExecutor, ToolSpec, split_system, truncate

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.model = model or DEFAULT_MODEL
        self.effort = os.environ.get("ANTHROPIC_EFFORT", "medium")
        self.use_fallbacks = os.environ.get("ANTHROPIC_FALLBACKS", "1") != "0"
        # SDK retries 408/409/429/5xx and connection errors with backoff.
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=5)

    # -- low-level ---------------------------------------------------------

    def _create(self, **kwargs):
        """Send one request, streamed (the SDK refuses large max_tokens without streaming)."""
        kwargs.setdefault("model", self.model)
        kwargs.setdefault("output_config", {"effort": self.effort})
        if self.use_fallbacks:
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
            api = self.client.beta.messages
        else:
            api = self.client.messages
        with api.stream(**kwargs) as stream:
            return stream.get_final_message()

    @staticmethod
    def _text(response) -> str:
        return "".join(b.text for b in response.content if b.type == "text")

    # -- public API ----------------------------------------------------------

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 4096) -> str:
        system, rest = split_system(messages)
        kwargs: dict = {
            # Thinking tokens count against max_tokens; give headroom so short
            # outputs (scores, letters) are not cut off after thinking.
            "max_tokens": max(max_tokens, 16000),
            "messages": [{"role": m["role"], "content": m["content"]} for m in rest],
        }
        if system:
            kwargs["system"] = system
        response = self._create(**kwargs)
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Claude declined the request: {getattr(response, 'stop_details', None)}")
        return self._text(response)

    def ask(self, prompt: str, **kwargs) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def run_agent(
        self,
        system: str,
        prompt: str,
        tools: list[ToolSpec],
        execute: ToolExecutor,
        max_turns: int = 80,
        on_event: EventCallback | None = None,
    ) -> AgentResult:
        tool_defs = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]
        messages: list[dict] = [{"role": "user", "content": prompt}]
        texts: list[str] = []
        in_tok = out_tok = 0

        for turn in range(1, max_turns + 1):
            response = self._create(
                max_tokens=32000,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                tools=tool_defs,
                messages=messages,
                cache_control={"type": "ephemeral"},
            )
            in_tok += response.usage.input_tokens or 0
            out_tok += response.usage.output_tokens or 0

            text = self._text(response)
            if text:
                texts.append(text)
                if on_event:
                    on_event("text", text)

            if response.stop_reason == "refusal":
                texts.append("RESULT:FAILED:model_refused")
                return AgentResult("\n".join(texts), turn, in_tok, out_tok, "error")

            # Append the full content (thinking + tool_use blocks) unchanged.
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "pause_turn":
                continue

            calls = [b for b in response.content if b.type == "tool_use"]
            if not calls:
                return AgentResult("\n".join(texts), turn, in_tok, out_tok, "done")

            results = []
            for call in calls:
                args = call.input if isinstance(call.input, dict) else json.loads(call.input or "{}")
                if on_event:
                    on_event("tool_call", f"{call.name} {json.dumps(args)[:200]}")
                out = execute(call.name, args)
                content: list[dict] = [{"type": "text", "text": truncate(out.text) or "(no output)"}]
                for media_type, data in out.images:
                    content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
                results.append({"type": "tool_result", "tool_use_id": call.id, "content": content, "is_error": out.is_error})
            # All results for one turn go back in a single user message.
            messages.append({"role": "user", "content": results})

        return AgentResult("\n".join(texts), max_turns, in_tok, out_tok, "max_turns")

    def close(self) -> None:
        self.client.close()
