import json
import urllib.parse

import httpx
import pytest

from applypilot import llm
from applypilot.providers import codex, codex_auth
from applypilot.providers.base import ToolOutput, ToolSpec

from helpers import chatgpt_tokens, sse


def _write_auth(path, tokens):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"auth_mode": "chatgpt", "tokens": tokens, "extra": "keep-me"}))


# -- auth ---------------------------------------------------------------------


def test_authorize_url_matches_codex_cli_flow():
    url = codex_auth.authorize_url("challenge123", "state456")
    parsed = urllib.parse.urlparse(url)
    q = dict(urllib.parse.parse_qsl(parsed.query))
    assert parsed.netloc == "auth.openai.com" and parsed.path == "/oauth/authorize"
    assert q["client_id"] == "app_EMoamEEZ73f0CkXaXp7hrann"
    assert q["redirect_uri"] == "http://localhost:1455/auth/callback"
    assert q["code_challenge_method"] == "S256"
    assert q["scope"] == "openid profile email offline_access"
    assert q["codex_cli_simplified_flow"] == "true" and q["id_token_add_organizations"] == "true"


def test_account_id_from_jwt_claim():
    tokens = chatgpt_tokens(account="acct-xyz")
    assert codex_auth.account_id_from_tokens(tokens) == "acct-xyz"


def test_reuses_codex_cli_login_and_writes_refresh_back(monkeypatch):
    cli_path = codex_auth.codex_cli_auth_path()
    _write_auth(cli_path, chatgpt_tokens(exp_in=10))  # about to expire -> refresh
    fresh = chatgpt_tokens(exp_in=3600)
    sent = {}

    def fake_post(url, data=None, headers=None, timeout=None):
        sent.update(url=url, data=data)
        return httpx.Response(200, json={**fresh, "refresh_token": "rt-new"}, request=httpx.Request("POST", url))

    monkeypatch.setattr(codex_auth.httpx, "post", fake_post)
    assert codex_auth.has_credentials()
    access, account = codex_auth.get_access()
    assert access == fresh["access_token"] and account == "acct-123"
    assert sent["data"] == {"grant_type": "refresh_token", "client_id": codex_auth.CLIENT_ID, "refresh_token": "rt-old"}
    saved = json.loads(cli_path.read_text())
    assert saved["tokens"]["refresh_token"] == "rt-new"  # rotated token persisted where it came from
    assert saved["extra"] == "keep-me"


# -- provider -------------------------------------------------------------------


def _provider_with(handler, monkeypatch):
    _write_auth(codex_auth.own_auth_path(), chatgpt_tokens())
    p = codex.CodexProvider(model="gpt-test")
    p._client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(codex.time, "sleep", lambda s: None)
    return p


def test_chat_request_shape_and_text(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        body = sse(
            {"type": "response.output_text.delta", "delta": "Hel"},
            {"type": "response.output_text.delta", "delta": "lo"},
            {"type": "response.completed", "response": {"usage": {"input_tokens": 5, "output_tokens": 2}}},
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    p = _provider_with(handler, monkeypatch)
    out = p.chat([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}], temperature=0.7, max_tokens=50)
    assert out == "Hello"
    req = seen[0]
    assert str(req.url) == "https://chatgpt.com/backend-api/codex/responses"
    assert req.headers["chatgpt-account-id"] == "acct-123"
    assert req.headers["openai-beta"] == "responses=experimental"
    body = json.loads(req.content)
    assert body["stream"] is True and body["store"] is False
    assert body["instructions"] == "Be brief."
    assert "temperature" not in body and "max_output_tokens" not in body
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["input"] == [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi"}]}]


def test_run_agent_replays_items_and_tool_outputs(monkeypatch):
    bodies = []
    reasoning = {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc", "summary": []}
    call = {"type": "function_call", "call_id": "call_1", "name": "browser_navigate", "arguments": '{"url": "https://jobs.test"}'}

    def handler(request):
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            events = [{"type": "response.output_item.done", "item": reasoning},
                      {"type": "response.output_item.done", "item": call},
                      {"type": "response.completed", "response": {"usage": {}}}]
        else:
            msg = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Done. RESULT:APPLIED"}]}
            events = [{"type": "response.output_text.delta", "delta": "Done. RESULT:APPLIED"},
                      {"type": "response.output_item.done", "item": msg},
                      {"type": "response.completed", "response": {"usage": {"input_tokens": 9}}}]
        return httpx.Response(200, content=sse(*events))

    p = _provider_with(handler, monkeypatch)
    calls = []

    def execute(name, args):
        calls.append((name, args))
        return ToolOutput("navigated", images=[("image/png", "AAAA")])

    tools = [ToolSpec("browser_navigate", "nav", {"type": "object", "properties": {"url": {"type": "string"}}})]
    result = p.run_agent("sys", "apply please", tools, execute)

    assert calls == [("browser_navigate", {"url": "https://jobs.test"})]
    assert "RESULT:APPLIED" in result.text and result.turns == 2
    first, second = bodies
    assert first["tools"][0]["name"] == "browser_navigate" and first["tools"][0]["type"] == "function"
    assert first["prompt_cache_key"] == second["prompt_cache_key"]
    # store=false: reasoning + call replayed, followed by the function output (with screenshot)
    replayed_reasoning = {k: v for k, v in reasoning.items() if k != "id"}
    assert second["input"][1] == replayed_reasoning and second["input"][2] == call
    assert all("id" not in item for item in second["input"])  # stateless: no server item ids
    out = second["input"][3]
    assert out["type"] == "function_call_output" and out["call_id"] == "call_1"
    assert out["output"][0] == {"type": "input_text", "text": "navigated"}
    assert out["output"][1]["image_url"].startswith("data:image/png;base64,")


def test_401_triggers_refresh_then_succeeds(monkeypatch):
    statuses = [401, 200]

    def handler(request):
        code = statuses.pop(0)
        if code == 401:
            return httpx.Response(401, json={"detail": "expired"})
        return httpx.Response(200, content=sse({"type": "response.output_text.delta", "delta": "ok"},
                                               {"type": "response.completed", "response": {}}))

    p = _provider_with(handler, monkeypatch)
    forced = []
    real = codex_auth.get_access
    monkeypatch.setattr(codex_auth, "get_access", lambda force_refresh=False: (forced.append(force_refresh), real())[1])
    assert p.ask("x") == "ok"
    assert forced == [False, True]


def test_usage_limit_raises_and_fallback_switches_to_claude(monkeypatch):
    def handler(request):
        return httpx.Response(429, json={"error": {"type": "usage_limit_reached", "resets_in_seconds": 7200, "plan_type": "plus"}})

    p = _provider_with(handler, monkeypatch)
    with pytest.raises(codex.CodexUsageLimit):
        p.ask("x")

    class FakeClaude:
        name, model = "anthropic", "claude-opus-5-5"

        def ask(self, prompt, **kw):
            return "from claude"

        def close(self):
            pass

    fb = llm.FallbackProvider(p, lambda: FakeClaude())
    assert fb.ask("x") == "from claude"
    assert fb.name == "anthropic"
