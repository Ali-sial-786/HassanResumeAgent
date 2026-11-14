"""Story bank: real, resume-grounded STAR stories for cover letters.

Cover letters land better when they tell one concrete story instead of listing
skills. The risk is an LLM inventing a story. This module keeps stories honest:

1. ``build_story_bank`` asks the LLM to turn each resume accomplishment into a
   STAR story (situation, task, action, result) and to quote the resume lines it
   came from. Stories whose quotes are not actually in the resume are dropped,
   and any metric not found in the resume is removed.
2. The bank is saved to ``~/.applypilot/stories.yaml`` for you to review and
   enrich. Add true details the resume leaves out (what made it hard, who you
   worked with, what you learned) in each story's ``my_notes`` field. Those
   notes are treated as fact, so only write things that happened.
3. For each job, ``select_stories`` picks the one or two stories that best match
   the posting, and the cover letter narrates them using only the bank's facts.
4. ``allowed_numbers`` feeds the validator: a number in a letter that is not in
   the resume, the story bank or the job posting fails validation.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

import yaml

from applypilot.config import APP_DIR
from applypilot.llm import get_client

log = logging.getLogger(__name__)

STORIES_PATH = APP_DIR / "stories.yaml"

_HEADER = """\
# ApplyPilot story bank: real stories from your experience, used in cover letters.
#
# Generated from your resume. Review every story:
#   - Delete anything that is wrong or that you would not want to talk about in an interview.
#   - Add TRUE details in `my_notes` (the obstacle, the stakes, what you learned). They are used as facts.
#   - Add new stories by copying the shape below. `source_quotes` can be empty for stories you add.
# Regenerate from scratch with `applypilot stories --rebuild` (this overwrites the file).
"""

_EXTRACT_PROMPT = """You turn a resume into a bank of interview-ready STAR stories.

For each distinct accomplishment in the resume (aim for 4 to 10), produce one story. Use ONLY facts in the resume.
Do not invent numbers, tools, team sizes, employers, dates or outcomes. Where the resume does not say something,
leave that field short and generic rather than making it up (for example "situation": "The team needed faster reports.").

Return ONLY JSON in this shape:
{"stories": [
  {"id": "short-kebab-id",
   "title": "one-line headline",
   "employer": "company or project name from the resume",
   "role": "job title from the resume",
   "situation": "1 sentence: the context or problem",
   "task": "1 sentence: what the candidate was responsible for",
   "action": "1-2 sentences: what the candidate did, with the tools named in the resume",
   "result": "1 sentence: the outcome, using the resume's own numbers if it has any",
   "metrics": ["numbers exactly as written in the resume, e.g. '40%', '$2M', '3x'"],
   "skills": ["skills/tools this story shows, as written in the resume"],
   "themes": ["2-4 of: leadership, ownership, collaboration, customer, speed, quality, cost, scale, learning, conflict, ambiguity, mentoring, data, automation"],
   "source_quotes": ["1-3 short phrases copied VERBATIM from the resume that this story is based on"]}
]}"""

_SELECT_PROMPT = """Pick the stories from the candidate's story bank that would best convince this employer.
Choose 1 or 2 stories (2 only if both are strongly relevant). Prefer stories whose skills and themes match
what the job description emphasizes.

Return ONLY JSON: {"picks": [{"id": "<story id>", "angle": "one sentence: why this story matters for THIS job"}]}"""


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9%$.]+", " ", text.lower()).strip()


def _extract_json(raw: str) -> dict:
    from applypilot.scoring.tailor import extract_json

    return extract_json(raw)


_NUM_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?\s?(?:%|x|k|m|b|\+)?", re.IGNORECASE)


def numbers_in(text: str, with_bare: bool = True) -> set[str]:
    """Numeric tokens in text, normalized ("$2,000" -> "2000", "40 %" -> "40%").

    With ``with_bare`` the unit-less value is added too ("40%" also yields "40"),
    which is what you want for an allow-list but not for reporting.
    """
    out: set[str] = set()
    for m in _NUM_RE.finditer(text or ""):
        tok = m.group(0).lower().replace(",", "").replace("$", "").replace(" ", "")
        tok = tok.rstrip(".")
        if tok:
            out.add(tok)
            if with_bare:
                out.add(re.sub(r"[%xkmb+]$", "", tok))
    return out


def _validate_story(story: dict, resume_norm: str, resume_numbers: set[str]) -> dict | None:
    if not isinstance(story, dict) or not story.get("title"):
        return None
    quotes = [q for q in story.get("source_quotes") or [] if isinstance(q, str) and q.strip()]
    grounded = [q for q in quotes if _norm(q) and _norm(q) in resume_norm]
    if quotes and not grounded:
        log.info("Dropping story %r: its quotes are not in the resume", story.get("title"))
        return None
    story["source_quotes"] = grounded
    kept, dropped = [], []
    for metric in story.get("metrics") or []:
        (kept if numbers_in(str(metric)) and numbers_in(str(metric)) <= resume_numbers else dropped).append(str(metric))
    story["metrics"] = kept
    if dropped:
        log.info("Story %r: removed metrics not in resume: %s", story.get("title"), dropped)
        # Scrub unsupported numbers out of the narrative fields too.
        for field in ("situation", "task", "action", "result"):
            val = str(story.get(field) or "")
            for d in dropped:
                val = val.replace(d, "").replace("  ", " ")
            story[field] = val.strip()
    story.setdefault("my_notes", "")
    return story


def build_story_bank(resume_text: str, profile: dict | None = None, force: bool = False) -> list[dict]:
    """Create (or load) the story bank. Writes ~/.applypilot/stories.yaml."""
    if STORIES_PATH.exists() and not force:
        return load_story_bank()

    client = get_client()
    raw = client.chat(
        [
            {"role": "system", "content": _EXTRACT_PROMPT},
            {"role": "user", "content": f"RESUME:\n{resume_text}"},
        ],
        max_tokens=8000,
        temperature=0.2,
    )
    data = _extract_json(raw)
    resume_norm = _norm(resume_text)
    resume_numbers = numbers_in(resume_text)
    stories: list[dict] = []
    seen: set[str] = set()
    for s in data.get("stories", []):
        s = _validate_story(s, resume_norm, resume_numbers)
        if not s:
            continue
        sid = re.sub(r"[^a-z0-9-]+", "-", str(s.get("id") or s["title"]).lower()).strip("-")[:40] or "story"
        while sid in seen:
            sid += "-2"
        seen.add(sid)
        s["id"] = sid
        stories.append(s)

    save_story_bank(stories)
    log.info("Story bank: %d stories written to %s", len(stories), STORIES_PATH)
    return stories


def save_story_bank(stories: list[dict]) -> None:
    STORIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "stories": stories}
    STORIES_PATH.write_text(_HEADER + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")


def load_story_bank() -> list[dict]:
    if not STORIES_PATH.exists():
        return []
    try:
        doc = yaml.safe_load(STORIES_PATH.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        log.warning("stories.yaml is not valid YAML (%s); ignoring it", exc)
        return []
    return [s for s in doc.get("stories") or [] if isinstance(s, dict) and s.get("id")]


def _keyword_rank(stories: list[dict], job_text: str, k: int) -> list[dict]:
    jt = job_text.lower()

    def score(s: dict) -> int:
        terms = [*(s.get("skills") or []), *(s.get("themes") or [])]
        return sum(1 for t in terms if isinstance(t, str) and t.lower() in jt)

    return sorted(stories, key=score, reverse=True)[:k]


def select_stories(stories: list[dict], job: dict, k: int = 2) -> list[dict]:
    """Return up to k stories for this job, each with an added 'angle'."""
    if not stories:
        return []
    job_text = f"{job.get('title', '')}\n{(job.get('full_description') or '')[:5000]}"
    index = [
        {"id": s["id"], "title": s.get("title"), "skills": s.get("skills"), "themes": s.get("themes"), "result": s.get("result")}
        for s in stories
    ]
    by_id = {s["id"]: s for s in stories}
    try:
        raw = get_client().chat(
            [
                {"role": "system", "content": _SELECT_PROMPT},
                {"role": "user", "content": f"JOB:\n{job_text}\n\nSTORY BANK:\n{json.dumps(index, indent=1)}"},
            ],
            max_tokens=600,
            temperature=0.0,
        )
        picks = []
        for p in _extract_json(raw).get("picks", [])[:k]:
            s = by_id.get(p.get("id"))
            if s:
                picks.append({**s, "angle": p.get("angle", "")})
        if picks:
            return picks
    except Exception as exc:
        log.debug("Story selection via LLM failed (%s); using keyword match", exc)
    return [{**s, "angle": ""} for s in _keyword_rank(stories, job_text, k)]


def format_for_prompt(picks: list[dict]) -> str:
    """Render chosen stories as fact sheets for the cover-letter prompt."""
    blocks = []
    for i, s in enumerate(picks, 1):
        lines = [f"STORY {i}: {s.get('title')} ({s.get('role', '')}, {s.get('employer', '')})"]
        for field in ("situation", "task", "action", "result"):
            if s.get(field):
                lines.append(f"  {field.upper()}: {s[field]}")
        if s.get("metrics"):
            lines.append(f"  NUMBERS YOU MAY USE: {', '.join(map(str, s['metrics']))}")
        if s.get("my_notes"):
            lines.append(f"  CANDIDATE'S OWN NOTES (true details): {s['my_notes']}")
        if s.get("angle"):
            lines.append(f"  WHY IT FITS THIS JOB: {s['angle']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def allowed_numbers(resume_text: str, stories: list[dict], job: dict) -> set[str]:
    """Numbers a cover letter may contain: from the resume, story bank or the posting itself."""
    texts = [resume_text, (job.get("full_description") or ""), job.get("title") or ""]
    for s in stories:
        texts.append(json.dumps(s))
    out: set[str] = set()
    for t in texts:
        out |= numbers_in(t)
    return out
