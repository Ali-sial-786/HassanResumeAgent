"""Provider-neutral LLM interface.

Every provider implements two calls:

* ``chat(messages, temperature, max_tokens) -> str``: one completion. ``messages``
  uses the OpenAI-style ``[{"role": "system"|"user"|"assistant", "content": str}]``
  shape that the rest of ApplyPilot already builds.
* ``run_agent(system, prompt, tools, execute, ...) -> AgentResult``: a tool-calling
  loop. The provider keeps its own wire-format history; the caller only supplies
  tool specs and a function that executes one call and returns text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

log = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    """A tool the agent may call. ``parameters`` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolOutput:
    """What a tool call produced. ``images`` holds (media_type, base64) pairs."""

    text: str
    images: list[tuple[str, str]] = field(default_factory=list)
    is_error: bool = False


# execute(name, arguments) -> ToolOutput
ToolExecutor = Callable[[str, dict[str, Any]], ToolOutput]

# on_event(kind, detail): kind is "text", "tool_call" or "tool_result"
EventCallback = Callable[[str, str], None]


@dataclass
class AgentResult:
    text: str  # all assistant text, joined; the apply launcher scans it for RESULT: codes
    turns: int
    input_tokens: int = 0
    output_tokens: int = 0
    stopped_reason: str = "done"  # done | max_turns | error


class LLMProvider(Protocol):
    name: str
    model: str

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 4096) -> str: ...

    def ask(self, prompt: str, **kwargs) -> str: ...

    def run_agent(
        self,
        system: str,
        prompt: str,
        tools: list[ToolSpec],
        execute: ToolExecutor,
        max_turns: int = 80,
        on_event: EventCallback | None = None,
    ) -> AgentResult: ...

    def close(self) -> None: ...


def split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Pull system messages out of an OpenAI-style list (for APIs that take them separately)."""
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    return "\n\n".join(system_parts), rest


def truncate(text: str, limit: int = 20000) -> str:
    """Keep tool output from flooding the context window."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"
