import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import pytest

from applypilot import config, linkedin
from applypilot.apply import launcher
from applypilot.database import get_connection, init_db

LI_URL = "https://www.linkedin.com/jobs/view/111"
CO_URL = "https://jobs.example.test/222"


def _cookie_db(worker_id: int, expires_in_s: float) -> None:
    db = config.CHROME_WORKER_DIR / f"worker-{worker_id}" / "Default" / "Network" / "Cookies"
    db.parent.mkdir(parents=True, exist_ok=True)
    db.unlink(missing_ok=True)
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE cookies (host_key TEXT, name TEXT, expires_utc INTEGER)")
    expires = int((time.time() + expires_in_s + 11644473600) * 1_000_000)
    conn.execute("INSERT INTO cookies VALUES ('.www.linkedin.com', 'li_at', ?)", (expires,))
    conn.commit()
    conn.close()


@pytest.fixture
def jobs_db(tmp_path):
    init_db()
    conn = get_connection()
    conn.execute("DELETE FROM jobs")
    for url, score in ((LI_URL, 10), (CO_URL, 8)):
        conn.execute("INSERT INTO jobs (url, title, site, fit_score, tailored_resume_path) VALUES (?,?,?,?,?)",
                     (url, "Data Engineer", "x", score, str(tmp_path / "r.txt")))
    conn.commit()
    launcher._linkedin_paused = None
    yield conn
    launcher._linkedin_paused = None
    for wid in (0, 1):
        for p in (config.CHROME_WORKER_DIR / f"worker-{wid}").rglob("Cookies"):
            p.unlink()


def test_session_detection_honours_expiry():
    assert not linkedin.has_session(0)
    _cookie_db(0, 3600)
    assert linkedin.has_session(0)
    _cookie_db(0, -3600)
    assert not linkedin.has_session(0)


def test_linkedin_jobs_skipped_without_session(jobs_db):
    job = launcher.acquire_job(min_score=7, worker_id=0)
    assert job["url"] == CO_URL  # higher-scored LinkedIn job is held back
    assert launcher._linkedin_paused.startswith("not signed in")


def test_linkedin_jobs_used_with_session(jobs_db):
    _cookie_db(0, 3600)
    assert launcher.acquire_job(min_score=7, worker_id=0)["url"] == LI_URL


def test_daily_cap(jobs_db, monkeypatch):
    _cookie_db(0, 3600)
    monkeypatch.setenv("LINKEDIN_DAILY_LIMIT", "1")
    now = datetime.now(timezone.utc).isoformat()
    jobs_db.execute("INSERT INTO jobs (url, title, site, apply_status, applied_at) VALUES (?,?,?,?,?)",
                    ("https://www.linkedin.com/jobs/view/999", "x", "x", "applied", now))
    jobs_db.commit()
    job = launcher.acquire_job(min_score=7, worker_id=0)
    assert job["url"] == CO_URL and "daily limit" in launcher._linkedin_paused
    old = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()
    jobs_db.execute("UPDATE jobs SET applied_at = ? WHERE url LIKE '%999'", (old,))
    jobs_db.commit()
    assert linkedin.applied_last_24h(jobs_db) == 0


def test_login_wall_releases_job_and_pauses_linkedin(jobs_db, monkeypatch):
    _cookie_db(0, 3600)
    monkeypatch.setattr(launcher, "launch_chrome", lambda *a, **k: None)
    seen = []

    def fake_run_job(job, port, worker_id=0, model=None, dry_run=False):
        seen.append(job["url"])
        return ("failed:linkedin_login_required", 10) if job["url"] == LI_URL else ("applied", 10)

    monkeypatch.setattr(launcher, "run_job", fake_run_job)
    applied, failed = launcher.worker_loop(worker_id=0, limit=2)
    assert seen == [LI_URL, CO_URL]
    assert (applied, failed) == (1, 0)
    row = jobs_db.execute("SELECT apply_status, apply_attempts FROM jobs WHERE url = ?", (LI_URL,)).fetchone()
    assert row["apply_status"] is None and not row["apply_attempts"]  # kept for a later run, no attempt burned
    assert launcher._linkedin_paused == "linkedin_login_required"


def test_prompt_includes_linkedin_playbook_only_for_linkedin(tmp_path, monkeypatch):
    from applypilot.apply import prompt

    prof = json.loads(open("profile.example.json").read())
    monkeypatch.setattr(config, "load_profile", lambda: prof)
    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(b"%PDF")
    base = {"title": "Data Engineer", "site": "x", "tailored_resume_path": str(tmp_path / "r.txt")}
    li = prompt.build_prompt({**base, "url": LI_URL}, "resume")
    co = prompt.build_prompt({**base, "url": CO_URL}, "resume")
    assert "== LINKEDIN JOB" in li and "Easy Apply" in li and "RESULT:FAILED:linkedin_login_required" in li
    assert "== LINKEDIN JOB" not in co
    assert "NEVER linkedin.com" in co  # generic login step also forbids LinkedIn credentials


def test_easy_apply_only_runs_linkedin_separately(monkeypatch):
    from applypilot.discovery import jobspy as js

    calls = []
    monkeypatch.setattr(js, "_scrape_with_retry", lambda kw, max_retries=2: calls.append(kw) or __import__("pandas").DataFrame())
    js._run_one_search({"query": "q", "location": "Toronto, ON"}, ["indeed", "linkedin"], 5, 72, None,
                       {"linkedin_easy_apply_only": True}, 0, [], [], {})
    assert [c["site_name"] for c in calls] == [["indeed"], ["linkedin"]]
    assert calls[1]["easy_apply"] is True and calls[1]["fetch_description"] is True
    assert "easy_apply" not in calls[0]


def test_new_worker_inherits_session_check(jobs_db):
    import shutil

    shutil.rmtree(config.CHROME_WORKER_DIR / "worker-1", ignore_errors=True)
    _cookie_db(0, 3600)
    assert linkedin.worker_will_have_session(1)  # worker-1 will be cloned from worker-0
    assert launcher.acquire_job(min_score=7, worker_id=1)["url"] == LI_URL
