import base64
import json
import time


def make_jwt(claims: dict) -> str:
    def enc(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    return f"{enc({'alg': 'none'})}.{enc(claims)}.sig"


def chatgpt_tokens(exp_in: int = 3600, account: str = "acct-123", email: str = "me@example.com") -> dict:
    auth_claim = {"https://api.openai.com/auth": {"chatgpt_account_id": account, "chatgpt_plan_type": "plus"}}
    return {
        "id_token": make_jwt({"email": email, **auth_claim}),
        "access_token": make_jwt({"exp": int(time.time()) + exp_in, **auth_claim}),
        "refresh_token": "rt-old",
    }


def sse(*events: dict) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
