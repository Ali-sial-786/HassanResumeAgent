"""ChatGPT sign-in ("Sign in with ChatGPT") for the Codex backend.

This uses the same public OAuth client as the open-source Codex CLI
(github.com/openai/codex, codex-rs/login), so a ChatGPT Plus/Pro/Team
subscription can power ApplyPilot without a pay-per-token API key.

Three ways to get credentials, tried in this order by ``load_tokens``:

1. ``~/.applypilot/codex_auth.json``, written by ``applypilot login``.
2. An existing Codex CLI login at ``$CODEX_HOME/auth.json`` (default
   ``~/.codex/auth.json``). Refreshed tokens are written back to that same
   file, because refresh tokens rotate and the CLI's copy would otherwise go
   stale.

Login flows:
* Browser (default): PKCE authorization-code flow with a callback on
  http://localhost:1455/auth/callback (the port the OAuth client allows).
* Device code (``applypilot login --device``): for headless machines; open
  https://auth.openai.com/codex/device on any device and type the code.

This endpoint is not an official public API. OpenAI may change or restrict it,
and your subscription's usage limits apply.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import logging
import os
import secrets
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

ISSUER = os.environ.get("CODEX_OAUTH_ISSUER", "https://auth.openai.com")
CLIENT_ID = os.environ.get("CODEX_OAUTH_CLIENT_ID", "app_EMoamEEZ73f0CkXaXp7hrann")
TOKEN_URL = f"{ISSUER}/oauth/token"
CALLBACK_PORT = 1455
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}/auth/callback"
SCOPE = "openid profile email offline_access"
ORIGINATOR = os.environ.get("CODEX_ORIGINATOR", "applypilot")
JWT_AUTH_CLAIM = "https://api.openai.com/auth"
REFRESH_MARGIN_S = 300  # refresh when the access token expires within 5 minutes

_lock = threading.RLock()


def _app_dir() -> Path:
    from applypilot.config import APP_DIR

    return APP_DIR


def own_auth_path() -> Path:
    return _app_dir() / "codex_auth.json"


def codex_cli_auth_path() -> Path:
    return Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "auth.json"


# ---------------------------------------------------------------------------
# JWT helpers (claims only; signatures are verified by the server, not us)
# ---------------------------------------------------------------------------


def jwt_claims(token: str) -> dict:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return {}


def account_id_from_tokens(tokens: dict) -> str | None:
    if tokens.get("account_id"):
        return tokens["account_id"]
    for key in ("id_token", "access_token"):
        claims = jwt_claims(tokens.get(key) or "")
        acct = (claims.get(JWT_AUTH_CLAIM) or {}).get("chatgpt_account_id")
        if acct:
            return acct
    return None


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


@dataclass
class StoredAuth:
    path: Path
    data: dict  # whole auth.json document (Codex CLI shape)

    @property
    def tokens(self) -> dict:
        return self.data.get("tokens") or {}


def _read(path: Path) -> StoredAuth | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    tokens = data.get("tokens") or {}
    if not tokens.get("access_token") or not tokens.get("refresh_token"):
        return None
    return StoredAuth(path, data)


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def load_tokens() -> StoredAuth | None:
    """Return stored ChatGPT credentials, preferring ApplyPilot's own login."""
    if os.environ.get("CODEX_USE_CLI_AUTH", "1") == "0":
        return _read(own_auth_path())
    return _read(own_auth_path()) or _read(codex_cli_auth_path())


def has_credentials() -> bool:
    return load_tokens() is not None


def save_new_login(id_token: str, access_token: str, refresh_token: str) -> Path:
    tokens = {"id_token": id_token, "access_token": access_token, "refresh_token": refresh_token}
    tokens["account_id"] = account_id_from_tokens(tokens)
    data = {
        "auth_mode": "chatgpt",
        "OPENAI_API_KEY": None,
        "tokens": tokens,
        "last_refresh": datetime.now(timezone.utc).isoformat(),
    }
    path = own_auth_path()
    _write(path, data)
    return path


def logout() -> bool:
    path = own_auth_path()
    if path.exists():
        path.unlink()
        return True
    return False


# ---------------------------------------------------------------------------
# Token refresh
# ---------------------------------------------------------------------------


def _token_request(form: dict) -> dict:
    resp = httpx.post(TOKEN_URL, data=form, headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=30)
    if resp.status_code >= 400:
        raise RuntimeError(f"ChatGPT token request failed ({resp.status_code}): {resp.text[:300]}")
    return resp.json()


def _refresh(stored: StoredAuth) -> StoredAuth:
    body = _token_request(
        {"grant_type": "refresh_token", "client_id": CLIENT_ID, "refresh_token": stored.tokens["refresh_token"]}
    )
    data = dict(stored.data)
    tokens = dict(stored.tokens)
    for key in ("id_token", "access_token", "refresh_token"):
        if body.get(key):
            tokens[key] = body[key]
    tokens["account_id"] = tokens.get("account_id") or account_id_from_tokens(tokens)
    data["tokens"] = tokens
    data["last_refresh"] = datetime.now(timezone.utc).isoformat()
    _write(stored.path, data)  # write back where it came from (refresh tokens rotate)
    return StoredAuth(stored.path, data)


def get_access(force_refresh: bool = False) -> tuple[str, str]:
    """Return (access_token, chatgpt_account_id), refreshing when needed."""
    with _lock:
        stored = load_tokens()
        if stored is None:
            raise RuntimeError("Not signed in to ChatGPT. Run `applypilot login` (or `codex login`).")
        exp = jwt_claims(stored.tokens["access_token"]).get("exp", 0)
        if force_refresh or exp - time.time() < REFRESH_MARGIN_S:
            log.info("Refreshing ChatGPT access token (%s)", stored.path)
            stored = _refresh(stored)
        account_id = account_id_from_tokens(stored.tokens)
        if not account_id:
            raise RuntimeError("ChatGPT login has no account id; sign in again with `applypilot login`.")
        return stored.tokens["access_token"], account_id


def describe() -> dict:
    """Non-secret summary of the current login, for `applypilot doctor`."""
    stored = load_tokens()
    if stored is None:
        return {}
    claims = jwt_claims(stored.tokens.get("id_token") or "")
    auth = claims.get(JWT_AUTH_CLAIM) or {}
    return {
        "source": str(stored.path),
        "email": claims.get("email"),
        "plan": auth.get("chatgpt_plan_type"),
        "access_expires": datetime.fromtimestamp(jwt_claims(stored.tokens["access_token"]).get("exp", 0), timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Browser (PKCE) login
# ---------------------------------------------------------------------------


def _pkce() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def authorize_url(challenge: str, state: str) -> str:
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": ORIGINATOR,
    }
    return f"{ISSUER}/oauth/authorize?{urllib.parse.urlencode(params, quote_via=urllib.parse.quote)}"


def login_browser(open_browser: bool = True, timeout_s: int = 600, echo=print) -> Path:
    verifier, challenge = _pkce()
    state = secrets.token_urlsafe(32)
    result: dict = {}
    done = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/auth/callback":
                self.send_response(404)
                self.end_headers()
                return
            q = urllib.parse.parse_qs(parsed.query)
            if q.get("state", [""])[0] != state:
                result["error"] = "state mismatch"
            elif "error" in q:
                result["error"] = q.get("error_description", q["error"])[0]
            else:
                result["code"] = q.get("code", [""])[0]
            ok = "code" in result
            self.send_response(200 if ok else 400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            msg = "Signed in. You can close this tab and return to the terminal." if ok else f"Sign-in failed: {result.get('error')}"
            self.wfile.write(f"<html><body style='font-family:sans-serif;padding:3em'><h2>ApplyPilot</h2><p>{msg}</p></body></html>".encode())
            done.set()

    try:
        server = http.server.HTTPServer(("127.0.0.1", CALLBACK_PORT), Handler)
    except OSError as exc:
        raise RuntimeError(
            f"Port {CALLBACK_PORT} is busy (is `codex login` running?). Close it, or use `applypilot login --device`."
        ) from exc
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = authorize_url(challenge, state)
    echo(f"Open this URL to sign in with ChatGPT:\n\n  {url}\n")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        if not done.wait(timeout_s):
            raise RuntimeError("Timed out waiting for the ChatGPT sign-in callback.")
    finally:
        server.shutdown()
    if "code" not in result:
        raise RuntimeError(f"ChatGPT sign-in failed: {result.get('error')}")

    body = _token_request(
        {
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": result["code"],
            "redirect_uri": REDIRECT_URI,
            "code_verifier": verifier,
        }
    )
    return save_new_login(body["id_token"], body["access_token"], body["refresh_token"])


# ---------------------------------------------------------------------------
# Device-code login (headless)
# ---------------------------------------------------------------------------


def login_device(echo=print, timeout_s: int = 900) -> Path:
    api = f"{ISSUER}/api/accounts"
    resp = httpx.post(f"{api}/deviceauth/usercode", json={"client_id": CLIENT_ID}, timeout=30)
    if resp.status_code == 404:
        raise RuntimeError("Device-code sign-in is not enabled for this account/server. Use `applypilot login` instead.")
    resp.raise_for_status()
    uc = resp.json()
    user_code = uc.get("user_code") or uc.get("usercode")
    interval = int(uc.get("interval") or 5)
    echo(f"1. Open {ISSUER}/codex/device on any device and sign in to ChatGPT\n2. Enter this code: {user_code}\n")

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        poll = httpx.post(
            f"{api}/deviceauth/token", json={"device_auth_id": uc["device_auth_id"], "user_code": user_code}, timeout=30
        )
        if poll.status_code == 200:
            code = poll.json()
            body = _token_request(
                {
                    "grant_type": "authorization_code",
                    "client_id": CLIENT_ID,
                    "code": code["authorization_code"],
                    "redirect_uri": f"{ISSUER}/deviceauth/callback",
                    "code_verifier": code["code_verifier"],
                }
            )
            return save_new_login(body["id_token"], body["access_token"], body["refresh_token"])
        if poll.status_code not in (403, 404):
            raise RuntimeError(f"Device sign-in failed ({poll.status_code}): {poll.text[:200]}")
        time.sleep(interval)
    raise RuntimeError("Device sign-in timed out after 15 minutes.")
