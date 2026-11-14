import json

from applypilot import config
from applypilot.discovery import jobspy as jobspy_mod
from applypilot.scoring import cover_letter, stories
from applypilot.scoring.validator import unsupported_numbers, validate_cover_letter

RESUME = """JANE DOE
EXPERIENCE
Data Engineer, Acme Corp
- Rebuilt the nightly ETL pipeline in Airflow, cutting runtime by 40%
- Automated invoice reconciliation with Python, saving $120,000 a year
"""


def test_location_filter_reads_documented_yaml_keys():
    """Upstream issue #109: accept_patterns under `location:` were ignored."""
    cfg = {"location": {"accept_patterns": ["Toronto"], "reject_patterns": ["onsite only"]}}
    accept, reject = config.location_filter_lists(cfg)
    assert accept == ["Toronto"] and reject == ["onsite only"]
    assert jobspy_mod._location_ok("Toronto, ON", accept, reject)
    assert not jobspy_mod._location_ok("Austin, TX", accept, reject)
    # legacy flat keys still honoured
    assert config.location_filter_lists({"location_accept": ["Austin"]})[0] == ["Austin"]


def test_story_validation_drops_ungrounded_and_invented_metrics():
    norm, nums = stories._norm(RESUME), stories.numbers_in(RESUME)
    good = {"title": "ETL", "source_quotes": ["cutting runtime by 40%"], "metrics": ["40%", "3x"],
            "result": "Runtime fell 40%, 3x faster."}
    out = stories._validate_story(dict(good), norm, nums)
    assert out["metrics"] == ["40%"]  # "3x" is not in the resume
    assert "3x" not in out["result"]
    fake = {"title": "Led team of 12", "source_quotes": ["managed a team of 12 engineers"], "metrics": []}
    assert stories._validate_story(fake, norm, nums) is None


def test_unsupported_numbers_check():
    allowed = stories.allowed_numbers(RESUME, [], {"full_description": "Team of 15 engineers"})
    assert unsupported_numbers("Cut runtime by 40% and saved $120,000.", allowed) == []
    assert unsupported_numbers("Grew revenue 300% across a team of 15.", allowed) == ["300%"]
    assert unsupported_numbers("In 2024 I spent 3 years on it.", allowed) == []
    res = validate_cover_letter("Dear Hiring Manager,\nI grew sales 250%.\nJane", allowed_numbers=allowed)
    assert not res["passed"] and "250%" in res["errors"][0]


def test_build_bank_and_letter_uses_story(monkeypatch, tmp_path):
    monkeypatch.setattr(stories, "STORIES_PATH", tmp_path / "stories.yaml")
    bank_json = {"stories": [
        {"id": "etl", "title": "Rebuilt nightly ETL", "employer": "Acme Corp", "role": "Data Engineer",
         "situation": "Nightly ETL was slow.", "task": "Own the pipeline.", "action": "Rebuilt it in Airflow.",
         "result": "Runtime dropped 40%.", "metrics": ["40%"], "skills": ["Airflow"], "themes": ["speed"],
         "source_quotes": ["Rebuilt the nightly ETL pipeline in Airflow"]},
        {"id": "made-up", "title": "Keynote at PyCon", "source_quotes": ["gave a keynote at PyCon"]},
    ]}
    letter = ("Dear Hiring Manager,\n\nYour team runs Airflow at scale. At Acme Corp our nightly ETL was slow, "
              "so I rebuilt it in Airflow and runtime dropped 40%.\n\nHappy to walk through it.\n\nJane")
    prompts = []

    class FakeLLM:
        name, model = "fake", "fake"

        def chat(self, messages, temperature=0.0, max_tokens=4096):
            prompts.append(messages)
            sysmsg = messages[0]["content"]
            if sysmsg.startswith("You turn a resume"):
                return json.dumps(bank_json)
            if sysmsg.startswith("Pick the stories"):
                return json.dumps({"picks": [{"id": "etl", "angle": "They need faster pipelines."}]})
            return letter

    monkeypatch.setattr(stories, "get_client", lambda role="text": FakeLLM())
    monkeypatch.setattr(cover_letter, "get_client", lambda role="text": FakeLLM())

    bank = stories.build_story_bank(RESUME)
    assert [s["id"] for s in bank] == ["etl"]  # ungrounded story dropped
    assert (tmp_path / "stories.yaml").read_text().startswith("# ApplyPilot story bank")
    assert stories.load_story_bank()[0]["my_notes"] == ""

    job = {"title": "Data Engineer", "site": "Globex", "full_description": "We use Airflow."}
    profile = {"personal": {"full_name": "Jane Doe"}, "skills_boundary": {"tools": ["Airflow", "Python"]}}
    out = cover_letter.generate_cover_letter(RESUME, job, profile, story_bank=bank)
    assert out == letter
    user_msg = prompts[-1][1]["content"]
    assert "STORY 1: Rebuilt nightly ETL" in user_msg and "They need faster pipelines." in user_msg


def test_location_defaults_to_searched_cities():
    """Wizard-style config without accept_patterns must keep onsite jobs in the searched city."""
    cfg = {"locations": [{"location": "New York, NY", "remote": False}, {"location": "Remote", "remote": True}]}
    accept, _ = config.location_filter_lists(cfg)
    assert accept == ["New York"]
    assert jobspy_mod._location_ok("New York, NY (Hybrid)", accept, [])
    assert not jobspy_mod._location_ok("Chicago, IL", accept, [])


def test_documented_criteria_keys_are_honoured(monkeypatch):
    import pandas as pd
    from applypilot.database import init_db

    cfg = {"boards": ["indeed", "linkedin"], "country": "Canada", "exclude_titles": ["Intern", "Director"]}
    assert config.job_boards(cfg) == ["indeed", "linkedin"]
    assert config.job_boards({"sites": ["google"]}) == ["google"]
    assert config.indeed_country(cfg) == "canada"
    monkeypatch.setattr(config, "load_search_config", lambda: cfg)
    df = pd.DataFrame([
        {"job_url": "https://x.test/1", "title": "Data Engineer", "location": "Toronto"},
        {"job_url": "https://x.test/2", "title": "Data Engineering Intern", "location": "Toronto"},
        {"job_url": "https://x.test/3", "title": "Director of Data", "location": "Toronto"},
    ])
    conn = init_db()
    new, _ = jobspy_mod.store_jobspy_results(conn, df, "q")
    assert new == 1
    titles = [r[0] for r in conn.execute("SELECT title FROM jobs WHERE url LIKE 'https://x.test/%'")]
    assert titles == ["Data Engineer"]
