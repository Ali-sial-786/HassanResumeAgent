"""ChatGPT-subscription provider: the Codex backend Responses endpoint.

Endpoint: https://chatgpt.com/backend-api/codex/responses (streaming only).
Auth: OAuth tokens from ``codex_auth`` (``applypilot login`` or Codex CLI).

Request shape follows the open-source Codex CLI and ChatMock (RayBytes/ChatMock):
Responses API, ``stream: true``, ``store: false``, no ``temperature`` /
``max_output_tokens``, reasoning items round-tripped via
``include: ["reasoning.encrypted_content"]``.

Model: ``LLM_MODEL`` if set, otherwise the top "list"-visible model in the
account's catalog (GET /backend-api/codex/models), falling back to
``FALLBACK_MODEL``. Reasoning effort: ``CODEX_EFFORT`` (default ``medium``).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import time
import uuid
from typing import Any, Iterator

import httpx

from applypilot import __version__
from applypilot.providers import codex_auth
from applypilot.providers.base import AgentResult, EventCallback, ToolExecutor, ToolSpec, split_system, truncate

log = logging.getLogger(__name__)

BASE_URL = os.environ.get("CODEX_BASE_URL", "https://chatgpt.com/backend-api/codex")
FALLBACK_MODEL = "gpt-5.5"
MODELS_CLIENT_VERSION = os.environ.get("CODEX_CLIENT_VERSION", "0.160.0")
_MAX_RETRIES = 6
_MAX_RATE_WAIT_S = 300  # longer than this (e.g. a usage window reset hours away) -> fail with a clear message


class CodexUsageLimit(RuntimeError):
    """The ChatGPT plan's usage window is exhausted."""


class CodexProvider:
    name = "codex"

    def __init__(self, model: str | None = None) -> None:
        self.effort = os.environ.get("CODEX_EFFORT", "medium")
        self.verbosity = os.environ.get("CODEX_VERBOSITY", "")
        self.installation_id = self._installation_id()
        self._client = httpx.Client(timeout=httpx.Timeout(600, connect=30))
        self.model = model or self._default_model()

    # -- plumbing ------------------------------------------------------------

    @staticmethod
    def _installation_id() -> str:
        path = codex_auth.own_auth_path().parent / "codex_installation_id"
        try:
            return str(uuid.UUID(path.read_text().strip()))
        except Exception:
            value = str(uuid.uuid4())
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value)
            except OSError:
                pass
            return value

    def _headers(self, access: str, account: str, session_id: str, accept: str = "text/event-stream") -> dict:
        return {
            "Authorization": f"Bearer {access}",
            "ChatGPT-Account-ID": account,
            "Content-Type": "application/json",
            "Accept": accept,
            "OpenAI-Beta": "responses=experimental",
            "originator": codex_auth.ORIGINATOR,
            "User-Agent": f"applypilot/{__version__} ({platform.system()} {platform.release()}; {platform.machine()})",
            "session-id": session_id,
            "x-codex-installation-id": self.installation_id,
        }

    def _default_model(self) -> str:
        try:
            access, account = codex_auth.get_access()
            resp = self._client.get(
                f"{BASE_URL}/models",
                params={"client_version": MODELS_CLIENT_VERSION},
                headers=self._headers(access, account, str(uuid.uuid4()), accept="application/json"),
                timeout=15,
            )
            resp.raise_for_status()
            models = [m for m in resp.json().get("models", []) if isinstance(m, dict) and m.get("visibility") == "list"]
            models.sort(key=lambda m: m.get("priority", 1_000_000) if isinstance(m.get("priority"), int) else 1_000_000)
            if models and models[0].get("slug"):
                return models[0]["slug"]
        except Exception as exc:
            log.warning("Could not read the Codex model catalog (%s); using %s", exc, FALLBACK_MODEL)
        return FALLBACK_MODEL

    def list_models(self) -> list[str]:
        access, account = codex_auth.get_access()
        resp = self._client.get(
            f"{BASE_URL}/models",
            params={"client_version": MODELS_CLIENT_VERSION},
            headers=self._headers(access, account, str(uuid.uuid4()), accept="application/json"),
            timeout=15,
        )
        resp.raise_for_status()
        return [m["slug"] for m in resp.json().get("models", []) if m.get("visibility") == "list" and m.get("slug")]

    @staticmethod
    def _iter_sse(resp: httpx.Response) -> Iterator[dict]:
        data_lines: list[str] = []
        for line in resp.iter_lines():
            if line == "":
                if data_lines:
                    raw = "\n".join(data_lines)
                    data_lines = []
                    if raw.strip() == "[DONE]":
                        return
                    try:
                        yield json.loads(raw)
                    except json.JSONDecodeError:
                        log.debug("Bad SSE payload: %s", raw[:200])
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if data_lines:
            try:
                yield json.loads("\n".join(data_lines))
            except json.JSONDecodeError:
                pass

    @staticmethod
    def _rate_wait(resp: httpx.Response, attempt: int) -> float:
        """Seconds to wait on a 429, or raise CodexUsageLimit if the window is far away."""
        wait = None
        try:
            err = resp.json().get("error", {})
            if err.get("resets_in_seconds") is not None:
                wait = float(err["resets_in_seconds"])
            if err.get("type") == "usage_limit_reached" or err.get("code") == "usage_limit_reached":
                if wait is None or wait > _MAX_RATE_WAIT_S:
                    raise CodexUsageLimit(
                        f"ChatGPT usage limit reached ({err.get('plan_type', 'plan')}); resets in "
                        f"{int(wait or 0) // 60} min. Wait, or set LLM_PROVIDER=anthropic to use a Claude API key."
                    )
        except (ValueError, AttributeError):
            pass
        if wait is None and resp.headers.get("retry-after"):
            try:
                wait = float(resp.headers["retry-after"])
            except ValueError:
                pass
        return min(wait if wait is not None else 5 * 2**attempt, _MAX_RATE_WAIT_S)

    def _respond(self, payload: dict, on_text: EventCallback | None = None) -> dict:
        """POST one streamed Responses request. Returns {"items", "text", "usage"}."""
        session_id = payload.get("prompt_cache_key") or str(uuid.uuid4())
        force_refresh = False
        for attempt in range(_MAX_RETRIES):
            access, account = codex_auth.get_access(force_refresh=force_refresh)
            force_refresh = False
            try:
                with self._client.stream(
                    "POST", f"{BASE_URL}/responses", json=payload, headers=self._headers(access, account, session_id)
                ) as resp:
                    if resp.status_code == 401 and attempt == 0:
                        force_refresh = True
                        continue
                    if resp.status_code == 429 or resp.status_code >= 500:
                        resp.read()
                        if attempt == _MAX_RETRIES - 1:
                            raise RuntimeError(f"Codex backend error {resp.status_code}: {resp.text[:300]}")
                        wait = self._rate_wait(resp, attempt) if resp.status_code == 429 else 3 * 2**attempt
                        log.warning("Codex backend %s; retrying in %.0fs", resp.status_code, wait)
                        time.sleep(wait)
                        continue
                    if resp.status_code >= 400:
                        resp.read()
                        raise RuntimeError(f"Codex backend rejected the request ({resp.status_code}): {resp.text[:500]}")

                    items: list[dict] = []
                    text_parts: list[str] = []
                    usage: dict = {}
                    for event in self._iter_sse(resp):
                        etype = event.get("type", "")
                        if etype == "response.output_text.delta":
                            text_parts.append(event.get("delta", ""))
                        elif etype == "response.output_item.done":
                            items.append(event["item"])
                        elif etype in ("response.completed", "response.incomplete"):
                            usage = (event.get("response") or {}).get("usage") or {}
                            break
                        elif etype in ("response.failed", "error"):
                            err = (event.get("response") or {}).get("error") or event.get("error") or event
                            raise RuntimeError(f"Codex response failed: {json.dumps(err)[:500]}")
                    text = "".join(text_parts)
                    if not text:  # some streams only carry the final message item
                        text = "".join(
                            c.get("text", "")
                            for it in items if it.get("type") == "message"
                            for c in it.get("content", []) if c.get("type") == "output_text"
                        )
                    if text and on_text:
                        on_text("text", text)
                    return {"items": items, "text": text, "usage": usage}
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                if attempt == _MAX_RETRIES - 1:
                    raise
                log.warning("Codex transport error (%s); retrying", exc)
                time.sleep(3 * 2**attempt)
        raise RuntimeError("Codex request failed after retries")

    def _base_payload(self, instructions: str, input_items: list[dict]) -> dict:
        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions or "You are a helpful assistant.",
            "input": input_items,
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"effort": self.effort},
            "prompt_cache_key": str(uuid.uuid4()),
        }
        if self.verbosity:
            payload["text"] = {"verbosity": self.verbosity}
        return payload

    @staticmethod
    def _message_item(role: str, text: str) -> dict:
        kind = "output_text" if role == "assistant" else "input_text"
        return {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}

    # -- public API ----------------------------------------------------------

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 4096) -> str:
        # temperature / max_tokens are not accepted by this endpoint; length is steered by the prompt.
        system, rest = split_system(messages)
        items = [self._message_item(m["role"], m["content"]) for m in rest]
        return self._respond(self._base_payload(system, items))["text"]

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
        tool_defs = [
            {"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters, "strict": False}
            for t in tools
        ]
        history: list[dict] = [self._message_item("user", prompt)]
        texts: list[str] = []
        in_tok = out_tok = 0
        cache_key = str(uuid.uuid4())  # one key per application keeps the prefix cache warm

        for turn in range(1, max_turns + 1):
            payload = self._base_payload(system, history)
            payload.update({"tools": tool_defs, "tool_choice": "auto", "parallel_tool_calls": False, "prompt_cache_key": cache_key})
            out = self._respond(payload, on_text=on_event)
            in_tok += out["usage"].get("input_tokens", 0)
            out_tok += out["usage"].get("output_tokens", 0)
            if out["text"]:
                texts.append(out["text"])

            # store=false: the full output (reasoning incl. encrypted_content, messages,
            # function calls) must be replayed on the next turn, without server item
            # ids (nothing is persisted, so ids would point at missing items).
            history.extend({k: v for k, v in item.items() if k != "id"} for item in out["items"])
            calls = [it for it in out["items"] if it.get("type") == "function_call"]
            if not calls:
                return AgentResult("\n".join(texts), turn, in_tok, out_tok, "done")

            for call in calls:
                try:
                    args = json.loads(call.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if on_event:
                    on_event("tool_call", f"{call['name']} {call.get('arguments', '')[:200]}")
                result = execute(call["name"], args)
                text = truncate(result.text) or "(no output)"
                if result.is_error:
                    text = "ERROR: " + text
                if result.images:
                    output: Any = [{"type": "input_text", "text": text}] + [
                        {"type": "input_image", "image_url": f"data:{mt};base64,{b64}"} for mt, b64 in result.images
                    ]
                else:
                    output = text
                history.append({"type": "function_call_output", "call_id": call["call_id"], "output": output})

        return AgentResult("\n".join(texts), max_turns, in_tok, out_tok, "max_turns")

    def close(self) -> None:
        self._client.close()
