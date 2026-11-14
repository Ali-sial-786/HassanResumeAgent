"""ApplyPilot CLI — the main entry point."""

from __future__ import annotations

import logging
import os
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from applypilot import __version__

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

app = typer.Typer(
    name="applypilot",
    help="AI-powered end-to-end job application pipeline.",
    no_args_is_help=True,
)
console = Console()
log = logging.getLogger(__name__)

# Valid pipeline stages (in execution order)
VALID_STAGES = ("discover", "enrich", "score", "tailor", "cover", "pdf")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bootstrap() -> None:
    """Common setup: load env, create dirs, init DB."""
    from applypilot.config import load_env, ensure_dirs
    from applypilot.database import init_db

    load_env()
    ensure_dirs()
    init_db()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold]applypilot[/bold] {__version__}")
        raise typer.Exit()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """ApplyPilot — AI-powered end-to-end job application pipeline."""


@app.command()
def init() -> None:
    """Run the first-time setup wizard (profile, resume, search config)."""
    from applypilot.wizard.init import run_wizard

    run_wizard()


@app.command()
def login(
    device: bool = typer.Option(False, "--device", help="Use a device code (for headless/remote machines)."),
    no_browser: bool = typer.Option(False, "--no-browser", help="Print the sign-in URL instead of opening a browser."),
) -> None:
    """Sign in with your ChatGPT account (Plus/Pro/Team) to power ApplyPilot via Codex."""
    from applypilot.config import ensure_dirs, load_env
    from applypilot.providers import codex_auth

    load_env()
    ensure_dirs()
    try:
        path = codex_auth.login_device(echo=console.print) if device else codex_auth.login_browser(
            open_browser=not no_browser, echo=console.print
        )
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    info = codex_auth.describe()
    console.print(f"[green]Signed in[/green] as {info.get('email') or 'ChatGPT user'} "
                  f"({info.get('plan') or 'plan unknown'}). Tokens saved to {path}")


@app.command()
def logout() -> None:
    """Remove ApplyPilot's saved ChatGPT sign-in (does not touch ~/.codex)."""
    from applypilot.providers import codex_auth

    if codex_auth.logout():
        console.print("[green]Signed out.[/green]")
    else:
        console.print("No ApplyPilot ChatGPT login was stored.")
    if codex_auth.codex_cli_auth_path().exists():
        console.print(f"[dim]Note: a Codex CLI login at {codex_auth.codex_cli_auth_path()} is still used "
                      "unless you set CODEX_USE_CLI_AUTH=0.[/dim]")


@app.command("linkedin-login")
def linkedin_login() -> None:
    """Sign in to LinkedIn once in the agent's browser so it can apply to LinkedIn jobs."""
    _bootstrap()
    from applypilot import linkedin

    if linkedin.has_session(0):
        console.print("[green]Already signed in to LinkedIn[/green] in the agent's browser profile.")
        if not typer.confirm("Sign in again?", default=False):
            return
    console.print(
        "[bold]LinkedIn sign-in[/bold]\n"
        "Sign in with your own LinkedIn account in the window that opens. ApplyPilot never sees your password;\n"
        "it only keeps the browser session, in ~/.applypilot/chrome-workers/.\n"
        "[yellow]LinkedIn restricts accounts that look automated. ApplyPilot caps LinkedIn applications at "
        f"{linkedin.daily_limit()} per 24h (LINKEDIN_DAILY_LIMIT).[/yellow]\n"
    )
    try:
        ok = linkedin.login_interactive(0, echo=console.print)
    except Exception as exc:
        console.print(f"[red]LinkedIn sign-in failed: {exc}[/red]")
        raise typer.Exit(code=1)
    if ok:
        console.print("[green]Signed in.[/green] LinkedIn jobs will now be included in `applypilot apply`.")
    else:
        console.print("[red]No LinkedIn session was saved.[/red] Make sure you finished signing in, then try again.")
        raise typer.Exit(code=1)


@app.command()
def stories(
    rebuild: bool = typer.Option(False, "--rebuild", help="Regenerate the story bank from your resume (overwrites edits)."),
) -> None:
    """Build or show your story bank: real STAR stories from your resume, used in cover letters."""
    _bootstrap()
    from applypilot.config import RESUME_PATH, load_profile
    from applypilot.scoring import stories as story_mod

    if not RESUME_PATH.exists():
        console.print("[red]No resume.txt found.[/red] Run [bold]applypilot init[/bold] first.")
        raise typer.Exit(code=1)
    try:
        bank = story_mod.build_story_bank(RESUME_PATH.read_text(encoding="utf-8"), load_profile(), force=rebuild)
    except Exception as exc:
        console.print(f"[red]Could not build the story bank: {exc}[/red]")
        raise typer.Exit(code=1)
    table = Table(title=f"Story bank ({len(bank)} stories)")
    table.add_column("id", style="cyan")
    table.add_column("story")
    table.add_column("result")
    for s in bank:
        table.add_row(s["id"], f"{s.get('title', '')}\n[dim]{s.get('employer', '')}[/dim]", str(s.get("result", ""))[:90])
    console.print(table)
    console.print(f"\nEdit [bold]{story_mod.STORIES_PATH}[/bold] to fix stories or add true details in `my_notes`.")


@app.command()
def models() -> None:
    """Show the active LLM provider and, for ChatGPT sign-in, the models your plan offers."""
    _bootstrap()
    from applypilot.llm import get_client

    try:
        client = get_client()
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    console.print(f"Provider: [bold]{client.name}[/bold]   default model: [bold]{client.model}[/bold]")
    if hasattr(client, "list_models"):
        try:
            console.print("Available: " + ", ".join(client.list_models()))
        except Exception as exc:
            console.print(f"[yellow]Could not list models: {exc}[/yellow]")
    console.print("[dim]Set LLM_MODEL in ~/.applypilot/.env to choose one.[/dim]")


@app.command()
def run(
    stages: Optional[list[str]] = typer.Argument(
        None,
        help=(
            "Pipeline stages to run. "
            f"Valid: {', '.join(VALID_STAGES)}, all. "
            "Defaults to 'all' if omitted."
        ),
    ),
    min_score: int = typer.Option(7, "--min-score", help="Minimum fit score for tailor/cover stages."),
    workers: int = typer.Option(1, "--workers", "-w", help="Parallel threads for discovery/enrichment stages."),
    stream: bool = typer.Option(False, "--stream", help="Run stages concurrently (streaming mode)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview stages without executing."),
    validation: str = typer.Option(
        "normal",
        "--validation",
        help=(
            "Validation strictness for tailor/cover stages. "
            "strict: banned words = errors, judge must pass. "
            "normal: banned words = warnings only (default, recommended for Gemini free tier). "
            "lenient: banned words ignored, LLM judge skipped (fastest, fewest API calls)."
        ),
    ),
) -> None:
    """Run pipeline stages: discover, enrich, score, tailor, cover, pdf."""
    _bootstrap()

    from applypilot.pipeline import run_pipeline

    stage_list = stages if stages else ["all"]

    # Validate stage names
    for s in stage_list:
        if s != "all" and s not in VALID_STAGES:
            console.print(
                f"[red]Unknown stage:[/red] '{s}'. "
                f"Valid stages: {', '.join(VALID_STAGES)}, all"
            )
            raise typer.Exit(code=1)

    # Gate AI stages behind Tier 2
    llm_stages = {"score", "tailor", "cover"}
    if any(s in stage_list for s in llm_stages) or "all" in stage_list:
        from applypilot.config import check_tier
        check_tier(2, "AI scoring/tailoring")

    # Validate the --validation flag value
    valid_modes = ("strict", "normal", "lenient")
    if validation not in valid_modes:
        console.print(
            f"[red]Invalid --validation value:[/red] '{validation}'. "
            f"Choose from: {', '.join(valid_modes)}"
        )
        raise typer.Exit(code=1)

    result = run_pipeline(
        stages=stage_list,
        min_score=min_score,
        dry_run=dry_run,
        stream=stream,
        workers=workers,
        validation_mode=validation,
    )

    if result.get("errors"):
        raise typer.Exit(code=1)


@app.command()
def apply(
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Max applications to submit."),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of parallel browser workers."),
    min_score: int = typer.Option(7, "--min-score", help="Minimum fit score for job selection."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override the apply-agent model (else APPLY_LLM_MODEL / LLM_MODEL / provider default)."),
    continuous: bool = typer.Option(False, "--continuous", "-c", help="Run forever, polling for new jobs."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview actions without submitting."),
    headless: bool = typer.Option(False, "--headless", help="Run browsers in headless mode."),
    url: Optional[str] = typer.Option(None, "--url", help="Apply to a specific job URL."),
    gen: bool = typer.Option(False, "--gen", help="Generate prompt file for manual debugging instead of running."),
    mark_applied: Optional[str] = typer.Option(None, "--mark-applied", help="Manually mark a job URL as applied."),
    mark_failed: Optional[str] = typer.Option(None, "--mark-failed", help="Manually mark a job URL as failed (provide URL)."),
    fail_reason: Optional[str] = typer.Option(None, "--fail-reason", help="Reason for --mark-failed."),
    reset_failed: bool = typer.Option(False, "--reset-failed", help="Reset all failed jobs for retry."),
) -> None:
    """Launch auto-apply to submit job applications."""
    _bootstrap()

    from applypilot.config import check_tier, PROFILE_PATH as _profile_path
    from applypilot.database import get_connection

    # --- Utility modes (no Chrome/Claude needed) ---

    if mark_applied:
        from applypilot.apply.launcher import mark_job
        mark_job(mark_applied, "applied")
        console.print(f"[green]Marked as applied:[/green] {mark_applied}")
        return

    if mark_failed:
        from applypilot.apply.launcher import mark_job
        mark_job(mark_failed, "failed", reason=fail_reason)
        console.print(f"[yellow]Marked as failed:[/yellow] {mark_failed} ({fail_reason or 'manual'})")
        return

    if reset_failed:
        from applypilot.apply.launcher import reset_failed as do_reset
        count = do_reset()
        console.print(f"[green]Reset {count} failed job(s) for retry.[/green]")
        return

    # --- Full apply mode ---

    if model:
        os.environ["APPLY_LLM_MODEL"] = model

    from applypilot import linkedin as _linkedin
    if url and _linkedin.is_linkedin_url(url) and not _linkedin.has_session(0):
        console.print("[red]This is a LinkedIn job and the agent's browser is not signed in to LinkedIn.[/red]\n"
                      "Run [bold]applypilot linkedin-login[/bold] first.")
        raise typer.Exit(code=1)

    # Check 1: Tier 3 required (LLM provider + Chrome + npx)
    check_tier(3, "auto-apply")

    # Check 2: Profile exists
    if not _profile_path.exists():
        console.print(
            "[red]Profile not found.[/red]\n"
            "Run [bold]applypilot init[/bold] to create your profile first."
        )
        raise typer.Exit(code=1)

    # Check 3: Tailored resumes exist (skip for --gen with --url)
    if not (gen and url):
        conn = get_connection()
        ready = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL AND applied_at IS NULL"
        ).fetchone()[0]
        if ready == 0:
            console.print(
                "[red]No tailored resumes ready.[/red]\n"
                "Run [bold]applypilot run score tailor[/bold] first to prepare applications."
            )
            raise typer.Exit(code=1)

    if gen:
        from applypilot.apply.launcher import gen_prompt
        target = url or ""
        if not target:
            console.print("[red]--gen requires --url to specify which job.[/red]")
            raise typer.Exit(code=1)
        prompt_file = gen_prompt(target, min_score=min_score, model=model)
        if not prompt_file:
            console.print("[red]No matching job found for that URL.[/red]")
            raise typer.Exit(code=1)
        mcp_path = _profile_path.parent / ".mcp-apply-0.json"
        console.print(f"[green]Wrote prompt to:[/green] {prompt_file}")
        console.print(f"[green]Wrote MCP config to:[/green] {mcp_path}")
        console.print("Feed both to any MCP-capable agent to replay this application by hand.")
        return

    from applypilot.apply.launcher import main as apply_main

    effective_limit = limit if limit is not None else (0 if continuous else 1)

    console.print("\n[bold blue]Launching Auto-Apply[/bold blue]")
    console.print(f"  Limit:    {'unlimited' if continuous else effective_limit}")
    console.print(f"  Workers:  {workers}")
    try:
        from applypilot.llm import detect_provider_name
        _prov = detect_provider_name("apply")
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    console.print(f"  Provider: {_prov}{f' ({model})' if model else ''}")
    console.print(f"  Headless: {headless}")
    console.print(f"  Dry run:  {dry_run}")
    if url:
        console.print(f"  Target:   {url}")
    console.print()

    apply_main(
        limit=effective_limit,
        target_url=url,
        min_score=min_score,
        headless=headless,
        model=model,
        dry_run=dry_run,
        continuous=continuous,
        workers=workers,
    )


@app.command()
def status() -> None:
    """Show pipeline statistics from the database."""
    _bootstrap()

    from applypilot.database import get_stats

    stats = get_stats()

    console.print("\n[bold]ApplyPilot Pipeline Status[/bold]\n")

    # Summary table
    summary = Table(title="Pipeline Overview", show_header=True, header_style="bold cyan")
    summary.add_column("Metric", style="bold")
    summary.add_column("Count", justify="right")

    summary.add_row("Total jobs discovered", str(stats["total"]))
    summary.add_row("With full description", str(stats["with_description"]))
    summary.add_row("Pending enrichment", str(stats["pending_detail"]))
    summary.add_row("Enrichment errors", str(stats["detail_errors"]))
    summary.add_row("Scored by LLM", str(stats["scored"]))
    summary.add_row("Pending scoring", str(stats["unscored"]))
    summary.add_row("Tailored resumes", str(stats["tailored"]))
    summary.add_row("Pending tailoring (7+)", str(stats["untailored_eligible"]))
    summary.add_row("Cover letters", str(stats["with_cover_letter"]))
    summary.add_row("Ready to apply", str(stats["ready_to_apply"]))
    summary.add_row("Applied", str(stats["applied"]))
    summary.add_row("Apply errors", str(stats["apply_errors"]))

    console.print(summary)

    # Score distribution
    if stats["score_distribution"]:
        dist_table = Table(title="\nScore Distribution", show_header=True, header_style="bold yellow")
        dist_table.add_column("Score", justify="center")
        dist_table.add_column("Count", justify="right")
        dist_table.add_column("Bar")

        max_count = max(count for _, count in stats["score_distribution"]) or 1
        for score, count in stats["score_distribution"]:
            bar_len = int(count / max_count * 30)
            if score >= 7:
                color = "green"
            elif score >= 5:
                color = "yellow"
            else:
                color = "red"
            bar = f"[{color}]{'=' * bar_len}[/{color}]"
            dist_table.add_row(str(score), str(count), bar)

        console.print(dist_table)

    # By site
    if stats["by_site"]:
        site_table = Table(title="\nJobs by Source", show_header=True, header_style="bold magenta")
        site_table.add_column("Site")
        site_table.add_column("Count", justify="right")

        for site, count in stats["by_site"]:
            site_table.add_row(site or "Unknown", str(count))

        console.print(site_table)

    console.print()


@app.command()
def dashboard() -> None:
    """Generate and open the HTML dashboard in your browser."""
    _bootstrap()

    from applypilot.view import open_dashboard

    open_dashboard()


@app.command()
def doctor() -> None:
    """Check your setup and diagnose missing requirements."""
    import shutil
    from applypilot.config import (
        load_env, PROFILE_PATH, RESUME_PATH, RESUME_PDF_PATH,
        SEARCH_CONFIG_PATH, get_chrome_path,
    )

    load_env()

    ok_mark = "[green]OK[/green]"
    fail_mark = "[red]MISSING[/red]"
    warn_mark = "[yellow]WARN[/yellow]"

    results: list[tuple[str, str, str]] = []  # (check, status, note)

    # --- Tier 1 checks ---
    # Profile
    if PROFILE_PATH.exists():
        results.append(("profile.json", ok_mark, str(PROFILE_PATH)))
    else:
        results.append(("profile.json", fail_mark, "Run 'applypilot init' to create"))

    # Resume
    if RESUME_PATH.exists():
        results.append(("resume.txt", ok_mark, str(RESUME_PATH)))
    elif RESUME_PDF_PATH.exists():
        results.append(("resume.txt", warn_mark, "Only PDF found — plain-text needed for AI stages"))
    else:
        results.append(("resume.txt", fail_mark, "Run 'applypilot init' to add your resume"))

    # Search config
    if SEARCH_CONFIG_PATH.exists():
        results.append(("searches.yaml", ok_mark, str(SEARCH_CONFIG_PATH)))
    else:
        results.append(("searches.yaml", warn_mark, "Will use example config — run 'applypilot init'"))

    # jobspy (discovery dep installed separately)
    try:
        import jobspy  # noqa: F401
        results.append(("python-jobspy", ok_mark, "Job board scraping available"))
    except ImportError:
        results.append(("python-jobspy", warn_mark,
                        "pip install --no-deps python-jobspy && pip install pydantic tls-client requests markdownify regex"))

    # --- Tier 2 checks ---
    import os
    from applypilot.llm import detect_provider_name
    from applypilot.providers import codex_auth
    login = codex_auth.describe()
    if login:
        results.append(("ChatGPT sign-in", ok_mark,
                        f"{login.get('email') or 'signed in'} ({login.get('plan') or 'plan unknown'}) via {login['source']}"))
    else:
        results.append(("ChatGPT sign-in", warn_mark, "Run 'applypilot login' to use your ChatGPT subscription"))
    if os.environ.get("ANTHROPIC_API_KEY"):
        results.append(("Claude API key", ok_mark, f"ANTHROPIC_API_KEY set ({os.environ.get('LLM_MODEL') or 'claude-opus-5-5'})"))
    try:
        results.append(("Active LLM", ok_mark, f"text: {detect_provider_name('text')}, apply: {detect_provider_name('apply')}"))
    except RuntimeError as exc:
        results.append(("Active LLM", fail_mark, str(exc)))

    # --- Tier 3 checks ---
    from applypilot import linkedin as _linkedin
    if _linkedin.has_session(0):
        results.append(("LinkedIn session", ok_mark, f"signed in (applications capped at {_linkedin.daily_limit()}/24h)"))
    else:
        results.append(("LinkedIn session", warn_mark, "Run 'applypilot linkedin-login' to apply to LinkedIn jobs"))
    # Chrome
    try:
        chrome_path = get_chrome_path()
        results.append(("Chrome/Chromium", ok_mark, chrome_path))
    except FileNotFoundError:
        results.append(("Chrome/Chromium", fail_mark,
                        "Install Chrome or set CHROME_PATH env var (needed for auto-apply)"))

    # Node.js / npx (for Playwright MCP)
    npx_bin = shutil.which("npx")
    if npx_bin:
        results.append(("Node.js (npx)", ok_mark, npx_bin))
    else:
        results.append(("Node.js (npx)", fail_mark,
                        "Install Node.js 18+ from nodejs.org (needed for auto-apply)"))

    # CapSolver (optional)
    capsolver = os.environ.get("CAPSOLVER_API_KEY")
    if capsolver:
        results.append(("CapSolver API key", ok_mark, "CAPTCHA solving enabled"))
    else:
        results.append(("CapSolver API key", "[dim]optional[/dim]",
                        "Set CAPSOLVER_API_KEY in .env for CAPTCHA solving"))

    # --- Render results ---
    console.print()
    console.print("[bold]ApplyPilot Doctor[/bold]\n")

    col_w = max(len(r[0]) for r in results) + 2
    for check, status, note in results:
        pad = " " * (col_w - len(check))
        console.print(f"  {check}{pad}{status}  [dim]{note}[/dim]")

    console.print()

    # Tier summary
    from applypilot.config import get_tier, TIER_LABELS
    tier = get_tier()
    console.print(f"[bold]Current tier: Tier {tier} — {TIER_LABELS[tier]}[/bold]")

    if tier == 1:
        console.print("[dim]  → Tier 2 unlocks: scoring, tailoring, cover letters (needs `applypilot login` or an API key)[/dim]")
        console.print("[dim]  → Tier 3 unlocks: auto-apply (needs Chrome + Node.js)[/dim]")
    elif tier == 2:
        console.print("[dim]  → Tier 3 unlocks: auto-apply (needs Chrome + Node.js)[/dim]")

    console.print()


if __name__ == "__main__":
    app()
