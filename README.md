# HassanResumeAgent

An AI job-application agent: it finds jobs that match your criteria, scores them against your resume, tailors the resume, writes a cover letter built around a **real story from your experience**, and fills in and submits the application in a browser.

It runs on your **ChatGPT subscription** (sign in with ChatGPT, through the Codex backend), or on a **Claude API key**. Gemini, OpenAI and local models still work too.

This is a fork of [ApplyPilot](https://github.com/Pickle-Pixel/ApplyPilot) by Pickle-Pixel (AGPL-3.0). See [What this fork changes](#what-this-fork-changes).

---

## Quick start

Needs Python 3.11+, Node.js 18+ (for `npx`) and Chrome or Chromium.

```bash
git clone <this repo> && cd HassanResumeAgent
python -m venv .venv && source .venv/bin/activate
pip install -e .

applypilot init        # resume, profile, job criteria, AI provider (choose "codex")
applypilot login       # sign in with ChatGPT (opens a browser)
applypilot doctor      # check that everything is ready

applypilot run         # discover > enrich > score > tailor > cover letters
applypilot stories     # review the story bank your cover letters draw on
applypilot apply --dry-run --limit 3   # fill 3 applications WITHOUT submitting; watch it work
applypilot apply --limit 10            # submit for real
```

On a remote or headless machine, use `applypilot login --device` and enter the code at https://auth.openai.com/codex/device from your phone or laptop.

If you already use the Codex CLI (`codex login`), ApplyPilot reuses that login automatically.

---

## Pick your AI

Set `LLM_PROVIDER` in `~/.applypilot/.env` (the wizard writes it for you):

| `LLM_PROVIDER` | What you need | Notes |
|---|---|---|
| `codex` (default) | A ChatGPT Plus/Pro/Team plan + `applypilot login` | No per-token bill. Your plan's usage limits apply. The model is chosen from your plan's catalog (`applypilot models` lists them). |
| `anthropic` | `ANTHROPIC_API_KEY` | Default model `claude-opus-5-5`. Pay per token. |
| `gemini` / `openai` / `local` | `GEMINI_API_KEY` / `OPENAI_API_KEY` / `LLM_URL` | Upstream's original providers. |

Useful settings:

```bash
LLM_PROVIDER=codex
ANTHROPIC_API_KEY=sk-ant-...      # optional with codex: used automatically if ChatGPT usage limits are hit
LLM_MODEL=                        # override the model for the chosen provider
APPLY_LLM_PROVIDER=anthropic      # optional: a different provider just for the browser agent
APPLY_LLM_MODEL=
CODEX_EFFORT=medium               # reasoning effort on the ChatGPT backend (low/medium/high)
ANTHROPIC_EFFORT=medium           # effort for Claude
```

> The ChatGPT/Codex endpoint is not an official public API. It is the same one the open-source Codex CLI uses, and third-party tools rely on it, but OpenAI can change or restrict it at any time. If it breaks, set `LLM_PROVIDER=anthropic`.

---

## Choosing which jobs to apply to

Jobs pass through four filters, each driven by your files in `~/.applypilot/` (created by `applypilot init`, editable any time):

1. **Search** (`searches.yaml`): which postings get pulled in.
   - `queries`: job titles to search for. `tier: 1` is searched first.
   - `locations`: where to search; add `{location: "Remote", remote: true}` to also search remote roles.
   - `defaults.distance`: radius in miles. `defaults.hours_old`: only postings this fresh.
   - `boards`: `indeed`, `linkedin`, `glassdoor`, `zip_recruiter`, `google`. `country`: for Indeed/Glassdoor.
   - It also checks 48 Workday employer portals and about 30 company career sites that ship with the tool.
2. **Hard filters** (`searches.yaml`): applied before any AI runs.
   - `exclude_titles`: skip titles containing these words (e.g. "intern", "director").
   - `location.accept_patterns` / `reject_patterns`: remote jobs are always kept; hybrid and onsite jobs are kept only if their location matches an accept pattern. If you list none, the cities you search in are used.
3. **AI fit score**: the AI reads each full posting against your resume and scores it 1-10. Only jobs at or above `--min-score` (default 7) get a tailored resume, cover letter and application. Run `applypilot status` or `applypilot dashboard` to see the scores and the AI's reasons.
4. **Checks on the application page** (from `profile.json`): before filling anything, the agent re-checks the work arrangement against your accepted locations. It answers salary questions from your `compensation` range (never below your floor) and work-authorization questions from your profile, and it stops on onsite-only roles elsewhere, freelance marketplaces, and SSO-only logins.

Example `searches.yaml`:

```yaml
queries:
  - {query: "Data Engineer", tier: 1}
  - {query: "Analytics Engineer", tier: 2}
locations:
  - {location: "Toronto, ON", remote: false}
  - {location: "Remote", remote: true}
defaults: {hours_old: 72, results_per_site: 50, distance: 25}
location:
  accept_patterns: ["Toronto", "Mississauga", "Ontario", "Canada"]
  reject_patterns: ["onsite only"]
boards: [indeed, linkedin, zip_recruiter]
country: canada
exclude_titles: [intern, director, "vice president", principal]
```

The full annotated example is [`searches.example.yaml`](src/applypilot/config/searches.example.yaml).

## Cover letters with real stories

Generic cover letters list skills. Good ones tell one short, specific story. That's also where AI is most tempted to make things up, so this fork adds a **story bank**:

1. On the first cover-letter run (or `applypilot stories`), the AI turns each accomplishment on your resume into a STAR story (situation, task, action, result) and quotes the resume lines it came from.
2. Stories whose quotes aren't actually in your resume are thrown out, and numbers that aren't in your resume are removed.
3. The bank is saved to `~/.applypilot/stories.yaml`. **Read it.** Delete anything wrong. In `my_notes`, add true details your resume leaves out: what made the problem hard, who was affected, what you learned. Those notes make the stories feel like you, and the AI uses them as facts.
4. For each job, the AI picks the one or two stories that fit the posting best and tells them as a short narrative.
5. Before a letter is saved, a check rejects it if it contains any number not found in your resume, story bank or the job posting. After several failed retries, the letter is not saved at all.

Regenerate the bank with `applypilot stories --rebuild` (this overwrites your edits).

---

## How applying works

For each job, ApplyPilot starts Chrome with a dedicated profile and launches the [Playwright MCP](https://github.com/microsoft/playwright-mcp) browser server. Your chosen model then drives the browser through it: navigate, read the page, fill fields, upload the tailored resume and cover letter PDFs, answer screening questions from your profile, and submit. Results (`applied`, `expired`, `captcha`, `login_issue`, `failed:<reason>`) go into the local database; `applypilot status` and `applypilot dashboard` show them.

- `--dry-run` fills everything but stops before the final Submit. Use it first.
- `--workers 3` runs three browsers in parallel.
- Ctrl+C once skips the current job; twice stops.
- `CAPSOLVER_API_KEY` (optional) lets the agent solve CAPTCHAs; without it those jobs are marked `captcha`.
- `APPLY_GMAIL_MCP=1` (optional) lets the agent read verification codes from Gmail (read/search/send only).

**Browser profile:** the agent uses its own Chrome profile under `~/.applypilot/chrome-workers/`. Sign in to job sites in it once and the sessions persist. Upstream copied your real Chrome profile, which would give the AI-driven browser all your logged-in accounts; that's now opt-in (`APPLY_CLONE_CHROME_PROFILE=1`) and not recommended.

---

## Be careful

- **Accuracy is on you.** The pipeline is built not to fabricate, but read the tailored resumes, `stories.yaml` and a sample of cover letters before running `apply` at volume.
- **Job-site terms.** Automated applying and scraping go against the terms of some sites (LinkedIn especially), and accounts can be restricted. Prefer company career pages and ATS forms (Greenhouse, Lever, Workday...).
- **Prompt injection.** Job pages are untrusted. The agent is told to ignore instructions found on web pages, the tool that runs arbitrary code is hidden from it, and it runs in its own browser profile, but don't give it more access than it needs.
- **Volume.** Ten well-matched applications beat a thousand random ones. Keep `--min-score` high.

---

## What this fork changes

Compared with ApplyPilot 0.3.0 (`Pickle-Pixel/ApplyPilot@4a8d521`):

**New**
- `codex` provider: ChatGPT sign-in via the Codex backend (`providers/codex.py`, `providers/codex_auth.py`). Browser PKCE login, device-code login, reuse of `~/.codex/auth.json`, automatic token refresh, model picked from your plan's catalog, usage-limit detection.
- `anthropic` provider using the official SDK (`providers/anthropic_provider.py`), with automatic fallback from ChatGPT to Claude when ChatGPT usage limits are hit.
- A native browser agent (`apply/agent.py`) that replaces the `claude -p` subprocess. Any provider can drive Playwright MCP, so Claude Code is no longer required.
- Story bank (`scoring/stories.py`) and invented-number rejection for cover letters.
- New commands: `applypilot login | logout | models | stories`.

**Upstream bugs fixed**
- Location accept/reject patterns in `searches.yaml` were ignored, so non-remote jobs were silently dropped (upstream issue #109).
- `applypilot apply --url` never matched a job that had not been attempted before (SQL `NULL != 'in_progress'`).
- Auto-apply crashed on machines without an existing Chrome profile.
- `boards`, `country` and `exclude_titles` in `searches.yaml` were documented but never read, and the wizard's search radius was never used.
- The wizard wrote no location accept list, so onsite jobs in the city you entered were dropped.

**Hardening**
- Playwright MCP pinned to a tested version (upstream used `@latest`). Its arbitrary-code tool is hidden from the model.
- Cloning your real Chrome profile is opt-in.

---

## Development

```bash
pip install -e '.[dev]'
pytest            # provider request formats, agent loops, MCP plumbing, story grounding, filters
```

Layout: `llm.py` picks the provider; `providers/` holds one module per backend, all implementing `chat()` and `run_agent()`; `apply/agent.py` connects a provider to the MCP browser tools; the rest of the pipeline is upstream ApplyPilot.

## License

AGPL-3.0, inherited from ApplyPilot. You may use and modify it freely. If you distribute it, or run it as a service for other people, you must publish your source under the same license.
