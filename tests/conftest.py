import os
import sys
import tempfile
from pathlib import Path

# Isolate all user data before applypilot is imported (APP_DIR is read at import).
_TMP = tempfile.mkdtemp(prefix="applypilot-test-")
os.environ["APPLYPILOT_DIR"] = _TMP
os.environ["CODEX_HOME"] = str(Path(_TMP) / "codex-home")
for key in ("LLM_PROVIDER", "APPLY_LLM_PROVIDER", "LLM_MODEL", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
            "OPENAI_API_KEY", "LLM_URL"):
    os.environ.pop(key, None)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ("LLM_PROVIDER", "APPLY_LLM_PROVIDER", "LLM_MODEL", "APPLY_LLM_MODEL", "ANTHROPIC_API_KEY",
                "GEMINI_API_KEY", "OPENAI_API_KEY", "LLM_URL", "LLM_FALLBACK"):
        monkeypatch.delenv(key, raising=False)
    from applypilot import llm
    from applypilot.providers import codex_auth

    llm.reset_clients()
    for p in (codex_auth.own_auth_path(), codex_auth.codex_cli_auth_path()):
        if p.exists():
            p.unlink()
    yield
    llm.reset_clients()
