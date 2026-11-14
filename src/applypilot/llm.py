"""LLM provider selection.

Pick a provider with ``LLM_PROVIDER`` in ``~/.applypilot/.env``:

  codex      ChatGPT sign-in (Plus/Pro/Team subscription) via the Codex backend.
             Run ``applypilot login`` once, or reuse an existing Codex CLI login
             (~/.codex/auth.json). Default model: see providers/codex.py.
  anthropic  Claude API key (ANTHROPIC_API_KEY). Default model: claude-opus-5-5.
  gemini     GEMINI_API_KEY (upstream default).
  openai     OPENAI_API_KEY (platform API, pay-per-token).
  local      LLM_URL pointing at an OpenAI-compatible server (llama.cpp, Ollama).

If ``LLM_PROVIDER`` is unset, the first available one wins, in this order:
codex login -> ANTHROPIC_API_KEY -> GEMINI_API_KEY -> OPENAI_API_KEY -> LLM_URL.

``LLM_MODEL`` overrides the model for whichever provider is chosen.
With ChatGPT sign-in active and ANTHROPIC_API_KEY also set, a ChatGPT usage-limit
error switches to Claude for the rest of the run (``ANTHROPIC_MODEL`` picks its
model; ``LLM_FALLBACK=0`` disables this).
``APPLY_LLM_PROVIDER`` / ``APPLY_LLM_MODEL`` optionally use a different
provider or model for the browser apply agent only.

Callers use ``get_client().chat(...)`` / ``.ask(...)`` for text and
``get_client("apply").run_agent(...)`` for tool-calling.
"""

from __future__ import annotations

import logging
import os
import threading

from applypilot.providers.base import LLMProvider

log = logging.getLogger(__name__)

PROVIDERS = ("codex", "anthropic", "gemini", "openai", "local")


def _codex_available() -> bool:
    from applypilot.providers import codex_auth

    return codex_auth.has_credentials()


def detect_provider_name(role: str = "text") -> str:
    """Return the provider name for a role ("text" or "apply")."""
    explicit = ""
    if role == "apply":
        explicit = os.environ.get("APPLY_LLM_PROVIDER", "")
    explicit = (explicit or os.environ.get("LLM_PROVIDER", "")).strip().lower()
    if explicit:
        if explicit not in PROVIDERS:
            raise RuntimeError(f"Unknown LLM_PROVIDER '{explicit}'. Choose one of: {', '.join(PROVIDERS)}")
        return explicit

    if _codex_available():
        return "codex"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("LLM_URL"):
        return "local"
    raise RuntimeError(
        "No LLM provider configured. Run `applypilot login` to sign in with ChatGPT, "
        "or set ANTHROPIC_API_KEY (or GEMINI_API_KEY / OPENAI_API_KEY / LLM_URL) in ~/.applypilot/.env."
    )


def build_provider(name: str, model: str | None = None) -> LLMProvider:
    model = model or os.environ.get("LLM_MODEL") or None

    if name == "codex":
        from applypilot.providers.codex import CodexProvider

        return CodexProvider(model=model)

    if name == "anthropic":
        from applypilot.providers.anthropic_provider import AnthropicProvider

        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set.")
        return AnthropicProvider(api_key=key, model=model)

    from applypilot.providers.compat import LLMClient

    if name == "gemini":
        return LLMClient(
            "https://generativelanguage.googleapis.com/v1beta/openai",
            model or "gemini-2.0-flash",
            os.environ.get("GEMINI_API_KEY", ""),
            name="gemini",
        )
    if name == "openai":
        return LLMClient("https://api.openai.com/v1", model or "gpt-4o-mini", os.environ.get("OPENAI_API_KEY", ""), name="openai")
    if name == "local":
        url = os.environ.get("LLM_URL", "")
        if not url:
            raise RuntimeError("LLM_PROVIDER=local but LLM_URL is not set.")
        return LLMClient(url.rstrip("/"), model or "local-model", os.environ.get("LLM_API_KEY", ""), name="local")

    raise RuntimeError(f"Unknown provider '{name}'")


class FallbackProvider:
    """Use ``primary`` until it hits a usage limit, then ``backup`` for the rest of the run.

    Wraps ChatGPT sign-in with a Claude API key when both are configured
    (disable with LLM_FALLBACK=0).
    """

    def __init__(self, primary: LLMProvider, backup_factory) -> None:
        self._primary = primary
        self._backup_factory = backup_factory
        self._active = primary
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._active.name

    @property
    def model(self) -> str:
        return self._active.model

    def _switch(self, exc: Exception) -> None:
        with self._lock:
            if self._active is self._primary:
                log.warning("%s; switching to the Claude API key for the rest of this run.", exc)
                self._active = self._backup_factory()

    def _call(self, method: str, *args, **kwargs):
        from applypilot.providers.codex import CodexUsageLimit

        try:
            return getattr(self._active, method)(*args, **kwargs)
        except CodexUsageLimit as exc:
            self._switch(exc)
            return getattr(self._active, method)(*args, **kwargs)

    def chat(self, *args, **kwargs):
        return self._call("chat", *args, **kwargs)

    def ask(self, *args, **kwargs):
        return self._call("ask", *args, **kwargs)

    def run_agent(self, *args, **kwargs):
        return self._call("run_agent", *args, **kwargs)

    def list_models(self):
        return self._primary.list_models()

    def close(self) -> None:
        self._primary.close()
        if self._active is not self._primary:
            self._active.close()


_instances: dict[str, LLMProvider] = {}
_lock = threading.Lock()


def get_client(role: str = "text") -> LLMProvider:
    """Return (or create) the provider singleton for a role.

    role="text"  : scoring, tailoring, cover letters, stories, extraction.
    role="apply" : the browser agent (honours APPLY_LLM_PROVIDER / APPLY_LLM_MODEL).
    """
    with _lock:
        if role not in _instances:
            name = detect_provider_name(role)
            model = os.environ.get("APPLY_LLM_MODEL") if role == "apply" else None
            provider = build_provider(name, model)
            if name == "codex" and os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("LLM_FALLBACK", "1") != "0":
                provider = FallbackProvider(provider, lambda: build_provider("anthropic", os.environ.get("ANTHROPIC_MODEL")))
            log.info("LLM provider (%s): %s  model: %s", role, provider.name, provider.model)
            _instances[role] = provider
        return _instances[role]


def reset_clients() -> None:
    """Drop cached providers (used after login/logout and in tests)."""
    with _lock:
        for p in _instances.values():
            try:
                p.close()
            except Exception:
                pass
        _instances.clear()
