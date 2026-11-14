"""LinkedIn support: session handling, safety limits and the apply-page playbook.

How LinkedIn works with ApplyPilot:

* Discovery needs no account: JobSpy reads LinkedIn's public job search and
  job pages (full descriptions included).
* Applying needs you to be signed in. Logged out, LinkedIn hides the employer's
  application link behind a sign-in popup, and Easy Apply only exists for
  signed-in members. Run ``applypilot linkedin-login`` once: it opens the
  agent's own Chrome profile on LinkedIn's sign-in page and you sign in by hand
  (2FA included). The session cookie stays in that profile.
* The agent never types credentials into LinkedIn and never creates accounts.
  If it finds itself logged out or facing a security check, it stops and
  LinkedIn jobs are paused for the rest of the run.
* LinkedIn restricts accounts that behave like bots, so LinkedIn applications
  are capped per 24 hours (``LINKEDIN_DAILY_LIMIT``, default 25).
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from applypilot import config

log = logging.getLogger(__name__)

LOGIN_URL = "https://www.linkedin.com/login"
SESSION_COOKIE = "li_at"
DEFAULT_DAILY_LIMIT = 25

# Result reasons that mean "LinkedIn as a whole is unavailable right now", not "this job failed".
SESSION_FAILURES = {"linkedin_login_required", "linkedin_checkpoint"}


def is_linkedin_url(url: str | None) -> bool:
    return bool(url) and "linkedin.com" in url.lower()


def daily_limit() -> int:
    try:
        return int(os.environ.get("LINKEDIN_DAILY_LIMIT", DEFAULT_DAILY_LIMIT))
    except ValueError:
        return DEFAULT_DAILY_LIMIT


# ---------------------------------------------------------------------------
# Session detection (reads the worker profile's cookie DB; values stay encrypted)
# ---------------------------------------------------------------------------


def _cookie_dbs(profile_dir: Path) -> list[Path]:
    return [p for p in (profile_dir / "Default" / "Network" / "Cookies", profile_dir / "Default" / "Cookies") if p.exists()]


def _chrome_now() -> int:
    """Chrome cookie timestamps: microseconds since 1601-01-01."""
    return int((time.time() + 11644473600) * 1_000_000)


def has_session(worker_id: int = 0) -> bool:
    """True if the worker's Chrome profile holds an unexpired LinkedIn session cookie."""
    profile = config.CHROME_WORKER_DIR / f"worker-{worker_id}"
    for db in _cookie_dbs(profile):
        try:
            # Copy first: Chrome may hold a lock on the live DB.
            tmp = db.with_name(db.name + ".applypilot-read")
            shutil.copy2(db, tmp)
            conn = sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)
            try:
                row = conn.execute(
                    "SELECT expires_utc FROM cookies WHERE name = ? AND host_key LIKE '%linkedin.com' "
                    "ORDER BY expires_utc DESC LIMIT 1",
                    (SESSION_COOKIE,),
                ).fetchone()
            finally:
                conn.close()
                tmp.unlink(missing_ok=True)
            if row and (row[0] == 0 or row[0] > _chrome_now()):
                return True
        except (sqlite3.Error, OSError) as exc:
            log.debug("Could not read %s: %s", db, exc)
    return False


def worker_will_have_session(worker_id: int) -> bool:
    """Session check for a worker, counting profiles that will be cloned on first launch.

    A worker without a profile yet gets a copy of an existing worker's profile
    (apply/chrome.py), so it inherits that worker's LinkedIn session.
    """
    if (config.CHROME_WORKER_DIR / f"worker-{worker_id}" / "Default").exists():
        return has_session(worker_id)
    for other in sorted(config.CHROME_WORKER_DIR.glob("worker-*")):
        if (other / "Default").exists():
            try:
                return has_session(int(other.name.split("-", 1)[1]))
            except ValueError:
                continue
    return False


def applied_last_24h(conn) -> int:
    since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    return conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE apply_status = 'applied' AND applied_at >= ? "
        "AND (url LIKE '%linkedin.com%' OR COALESCE(application_url, '') LIKE '%linkedin.com%')",
        (since,),
    ).fetchone()[0]


# ---------------------------------------------------------------------------
# One-time interactive login
# ---------------------------------------------------------------------------


def login_interactive(worker_id: int = 0, timeout_s: int = 900, echo=print) -> bool:
    """Open the worker's Chrome on LinkedIn's sign-in page and wait for the user to sign in."""
    from playwright.sync_api import sync_playwright

    from applypilot.apply.chrome import cleanup_worker, launch_chrome

    port = 9400 + worker_id
    proc = launch_chrome(worker_id, port=port, headless=False)
    signed_in = False
    try:
        with sync_playwright() as pw:
            browser = None
            for _ in range(30):  # Chrome takes a moment to open its debugging port
                try:
                    browser = pw.chromium.connect_over_cdp(f"http://localhost:{port}")
                    break
                except Exception:
                    time.sleep(1)
            if browser is None:
                raise RuntimeError("Could not connect to the Chrome window.")
            context = browser.contexts[0] if browser.contexts else browser.new_context()
            page = context.new_page()
            page.goto(LOGIN_URL)
            echo("A Chrome window opened on LinkedIn. Sign in there (including any 2-step verification).\n"
                 "This window is the agent's own browser profile; your normal Chrome is not touched.\n"
                 "Waiting for you to finish signing in...")
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                if any(c["name"] == SESSION_COOKIE for c in context.cookies("https://www.linkedin.com")):
                    signed_in = True
                    break
                time.sleep(2)
            if signed_in:
                time.sleep(3)  # let LinkedIn finish setting its cookies
                # Close Chrome gracefully so the session is written to disk.
                try:
                    browser.new_browser_cdp_session().send("Browser.close")
                except Exception:
                    pass
                for _ in range(20):
                    if proc.poll() is not None:
                        break
                    time.sleep(0.5)
    finally:
        cleanup_worker(worker_id, proc)

    if signed_in:
        _share_profile_with_other_workers(worker_id)
    return signed_in and has_session(worker_id)


def _share_profile_with_other_workers(worker_id: int) -> None:
    """Other worker profiles are re-cloned from this one on their next launch."""
    for other in config.CHROME_WORKER_DIR.glob("worker-*"):
        if other.name != f"worker-{worker_id}" and other.is_dir():
            shutil.rmtree(other, ignore_errors=True)


# ---------------------------------------------------------------------------
# Apply-page playbook (added to the agent prompt for LinkedIn jobs)
# ---------------------------------------------------------------------------

PROMPT_SECTION = """== LINKEDIN JOB (read before acting) ==
This job is on linkedin.com. The browser profile should already be signed in to the candidate's LinkedIn account.
- NEVER type an email, password or code into any LinkedIn page. NEVER click "Join now", "Sign up" or create an account.
  If LinkedIn shows a sign-in page, a "Sign in to apply" popup, or a "Join LinkedIn" wall -> stop and output RESULT:FAILED:linkedin_login_required
- Security check, CAPTCHA, "verify it's you", "unusual activity" or restricted-account message on LinkedIn -> do NOT try to solve it. Output RESULT:FAILED:linkedin_checkpoint
- "No longer accepting applications" or "applications closed" -> RESULT:EXPIRED
- Already shows "Applied" (with a date) -> RESULT:FAILED:already_applied
- Do not message recruiters, connect, follow, save, or change anything on the candidate's LinkedIn profile.

A) "Easy Apply" button (LinkedIn's own form, opens a dialog):
   1. Click "Easy Apply". Take a snapshot of the dialog.
   2. Contact info is pre-filled from LinkedIn; fix email/phone only if they differ from the APPLICANT PROFILE.
   3. Resume step: upload the tailored resume PDF from FILES (use the upload button, then browser_file_upload). Make sure the uploaded file is the selected one.
   4. Each step: fill every required field from the profile/resume, answer screening questions using the rules above, click "Next" / "Continue to next step". Fix any red validation errors before moving on.
   5. On "Review your application": uncheck "Follow <company>" if checked. Verify everything. Then click "Submit application" (unless this is a dry run).
   6. Success = "Your application was sent" / "Application submitted". Close the dialog. Output RESULT:APPLIED.
B) "Apply" button with an external-link icon ("Apply on company website"):
   1. Click it. If a "Share your profile?" or similar dialog appears, choose to continue to the company site.
   2. The employer's site opens in a NEW TAB: browser_tabs action "list", then "select" the new tab. Continue the normal application steps there (the company's own site, not LinkedIn).
   3. Do not go back to LinkedIn afterwards."""
