import json
import sys
import threading
from pathlib import Path

import httpx
import pytest

from applypilot import llm
from applypilot.providers.anthropic_provider import AnthropicProvider
from applypilot.providers.base import ToolOutput, ToolSpec
from applypilot.providers.compat import LLMClient

from helpers import chatgpt_tokens

TOOLS = [ToolSpec("browser_snapshot", "snap", {"type": "object", "properties": {}})]


# -- selection --------------------------------------------------------------------


def test_detection_order_prefers_chatgpt_login(monkeypatch):
    from applypilot.providers import codex_auth

    with pytest.raises(RuntimeError, match="applypilot login"):
        llm.detect_provider_name()
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    assert llm.detect_provider_name() == "gemini"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert llm.detect_provider_name() == "anthropic"
    path = codex_auth.own_auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"tokens": chatgpt_tokens()}))
    assert llm.detect_provider_name() == "codex"
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert llm.detect_provider_name() == "anthropic"
    monkeypatch.setenv("APPLY_LLM_PROVIDER", "gemini")
    assert llm.detect_provider_name("apply") == "gemini"


def test_unknown_provider_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bogus")
    with pytest.raises(RuntimeError, match="Unknown LLM_PROVIDER"):
        llm.detect_provider_name()


# -- anthropic (through the real SDK, streaming, against a mock HTTP transport) --


def _anthropic_sse(blocks, stop_reason):
    """Build a Messages streaming response body for the given content blocks."""
    events = [("message_start", {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 0}}})]
    for i, block in enumerate(blocks):
        if block["type"] == "text":
            events.append(("content_block_start", {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}))
            events.append(("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": block["text"]}}))
        elif block["type"] == "tool_use":
            events.append(("content_block_start", {"type": "content_block_start", "index": i, "content_block": {
                "type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}}))
            events.append(("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {
                "type": "input_json_delta", "partial_json": json.dumps(block["input"])}}))
        events.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
    events.append(("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                                     "usage": {"output_tokens": 5}}))
    events.append(("message_stop", {"type": "message_stop"}))
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


def _anthropic_with(responses):
    import anthropic
    import httpx2

    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        blocks, stop = responses.pop(0)
        return httpx2.Response(200, content=_anthropic_sse(blocks, stop), headers={"content-type": "text/event-stream"})

    p = AnthropicProvider(api_key="sk-test")
    p.client = anthropic.Anthropic(api_key="sk-test", max_retries=0,
                                   http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    return p, bodies


def test_anthropic_agent_loop_and_request_shape():
    p, bodies = _anthropic_with([
        ([{"type": "tool_use", "id": "tu_1", "name": "browser_snapshot", "input": {}}], "tool_use"),
        ([{"type": "text", "text": "RESULT:APPLIED"}], "end_turn"),
    ])
    result = p.run_agent("sys", "go", TOOLS, lambda n, a: ToolOutput("page", images=[("image/png", "AAAA")]))
    assert result.text == "RESULT:APPLIED" and result.turns == 2
    first, second = bodies
    assert first["stream"] is True  # large max_tokens requires streaming in the SDK
    assert first["model"] == "claude-opus-5-5" and first["max_tokens"] == 32000
    assert first["fallbacks"] == "default"
    assert "temperature" not in first
    assert first["tools"][0]["input_schema"] == {"type": "object", "properties": {}}
    # assistant tool_use appended unchanged; all tool results in one user turn
    assert second["messages"][1]["content"][0]["type"] == "tool_use"
    tr = second["messages"][2]["content"][0]
    assert tr["type"] == "tool_result" and tr["tool_use_id"] == "tu_1"
    assert tr["content"][1]["type"] == "image"


def test_anthropic_chat_ignores_temperature():
    p, bodies = _anthropic_with([([{"type": "text", "text": "hi"}], "end_turn")])
    assert p.chat([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}], temperature=0.9) == "hi"
    call = bodies[0]
    assert call["system"] == "S" and "temperature" not in call and call["max_tokens"] >= 16000


# -- OpenAI-compatible ----------------------------------------------------------


def test_compat_agent_loop():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "browser_snapshot", "arguments": "{}"}}]}
        else:
            msg = {"role": "assistant", "content": "RESULT:EXPIRED"}
        return httpx.Response(200, json={"choices": [{"message": msg}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}})

    c = LLMClient("http://local.test/v1", "m", "k", name="local")
    c._client = httpx.Client(transport=httpx.MockTransport(handler))
    result = c.run_agent("sys", "go", TOOLS, lambda n, a: ToolOutput("snap"))
    assert result.text == "RESULT:EXPIRED"
    assert bodies[1]["messages"][-1] == {"role": "tool", "tool_call_id": "c1", "content": "snap"}


# -- MCP hub against a real (fake) stdio MCP server ------------------------------


def test_mcp_hub_lists_and_calls_tools():
    from applypilot.apply.agent import _McpHub

    hub = _McpHub()
    try:
        server = str(Path(__file__).with_name("fake_mcp_server.py"))
        hub.start({"fake": {"command": sys.executable, "args": [server]}})
        names = sorted(s.name for s in hub.specs)
        assert names == ["browser_navigate", "browser_snapshot"]
        assert all("$schema" not in s.parameters for s in hub.specs)
        assert hub.call("browser_navigate", {"url": "https://a.test"}).text == "navigated to https://a.test"
        assert hub.call("missing", {}).is_error
    finally:
        hub.close()


def test_run_apply_agent_end_to_end_with_fake_browser(monkeypatch):
    """Native agent: provider loop -> MCP server -> RESULT line, no Claude CLI involved."""
    from applypilot.apply import agent

    server = str(Path(__file__).with_name("fake_mcp_server.py"))
    monkeypatch.setattr(agent, "mcp_servers", lambda port: {"fake": {"command": sys.executable, "args": [server]}})

    class ScriptedProvider:
        name, model = "scripted", "x"

        def run_agent(self, system, prompt, tools, execute, max_turns=80, on_event=None):
            out = execute("browser_snapshot", {})
            assert "Submit" in out.text
            from applypilot.providers.base import AgentResult
            return AgentResult("Submitted.\nRESULT:APPLIED", 2)

    monkeypatch.setattr(agent, "get_client", lambda role="text": ScriptedProvider())
    result = agent.run_apply_agent("prompt", 9222, threading.Event())
    assert "RESULT:APPLIED" in result.text


def test_skip_event_aborts_agent(monkeypatch):
    from applypilot.apply import agent

    server = str(Path(__file__).with_name("fake_mcp_server.py"))
    monkeypatch.setattr(agent, "mcp_servers", lambda port: {"fake": {"command": sys.executable, "args": [server]}})

    class Provider:
        name, model = "p", "x"

        def run_agent(self, system, prompt, tools, execute, max_turns=80, on_event=None):
            return execute("browser_snapshot", {})

    monkeypatch.setattr(agent, "get_client", lambda role="text": Provider())
    ev = threading.Event()
    ev.set()
    with pytest.raises(agent.AgentSkipped):
        agent.run_apply_agent("prompt", 9222, ev)
