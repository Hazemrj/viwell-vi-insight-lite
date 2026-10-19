"""Unit tests for the safety rules.  Run:  pytest -q"""
import json

import pytest

from src.vi.guardrails import (check_grounding, check_no_medical, escalation_needed, escalation_response,
                               ungrounded_numbers, validate_output)
from src.vi.llm import MockClient
from src.vi.service import fallback_response, generate_insight

SNAP = {
    "member_id": "M0042", "as_of": "2026-09-28", "windows_days": [7, 30],
    "profile": {"age_band": "35-44", "job_type": "desk"},
    "metrics": {"sleep_hours": {"7d": 5.6, "30d": 6.9, "chg_pct": -18.8},
                "steps": {"7d": 4918, "30d": 5182, "chg_pct": -5.1},
                "active_minutes": {"7d": 17.2, "30d": 16.2, "chg_pct": 5.7},
                "resting_hr": {"7d": 71.3, "30d": 71.0, "chg_pct": 0.5}},
    "engagement": {"app_opens_per_day": {"7d": 1.4, "30d": 1.5, "chg_pct": -2.6}, "days_since_last_open": 0,
                   "last_active_date": "2026-09-28", "app_streak_days": 4, "content_views_7d": 5, "challenges_joined_30d": 1},
    "activity": {"active_streak_days": 0, "consistency_score": 68},
    "mood": {"avg_30d": 3.4, "trend": "stable", "recent_checkins": [3, 4, 3, 3, 4], "low_count_recent": 0},
    "data_quality": {"sync_rate_30d": 0.97}, "ml": {"persona": "Quiet Drifters", "disengagement_risk": 0.2, "risk_band": "low"},
}


def with_mood(values):
    s = json.loads(json.dumps(SNAP))
    s["mood"]["recent_checkins"] = values
    return s


# ---------- escalation rule ----------
@pytest.mark.parametrize("values,expected", [
    ([2, 1, 3, 2, 2], True),        # 4 low of 5
    ([3, 2, 2, 2, 3], True),        # exactly 3 low -> escalate
    ([2, 3, 2, 4, 3], False),       # only 2 low -> no
    ([1, 1, 5, 5, 5], False),       # 2 low
    ([2, 2, 2], True),              # only 3 check-ins, all low
    ([2, 2], False),                # not enough check-ins
    ([], False),
    ([3, 3, 3, 3, 3], False),       # 3 is NOT low
])
def test_escalation_rule(values, expected):
    assert escalation_needed(with_mood(values)) is expected


def test_escalation_response_is_safe_and_grounded():
    snap = with_mood([2, 1, 3, 2, 2])
    out = escalation_response(snap)
    assert out["safety_flag"] is True and out["tone"] == "supportive"
    assert "professional" in " ".join(out["actions"])
    assert check_grounding(out, snap) == []
    assert check_no_medical(out) == []


def test_escalation_skips_the_model(monkeypatch):
    class Boom:
        name = "boom"
        def chat(self, messages):
            raise AssertionError("the LLM must not be called for an escalation case")
    import src.vi.service as svc
    monkeypatch.setattr(svc, "load_snapshot", lambda mid: with_mood([1, 2, 2, 3, 4]))
    res = generate_insight("M0042", Boom())
    assert res["safety_flag"] is True and res["meta"]["source"] == "escalation_template"


# ---------- grounding ----------
def test_grounded_numbers_pass():
    assert ungrounded_numbers("Sleep was 5.6 hours vs 6.9 usual.", SNAP) == []
    assert ungrounded_numbers("steps_7d: 4918 vs baseline 5182", SNAP) == []
    assert ungrounded_numbers("Steps are down 5.1% and about 5% lower", SNAP) == []      # sign ignored, rounding ok
    assert ungrounded_numbers("You took 4,918 steps", SNAP) == []                         # thousands separator
    assert ungrounded_numbers("sleep_hours_7d: 5.6 vs baseline 6.9", SNAP) == []         # '_7d' label is not a number
    assert ungrounded_numbers("over the last 7 days", SNAP) == []                         # window length is in the snapshot


def test_invented_numbers_fail():
    assert ungrounded_numbers("Your sleep dropped 37%", SNAP) == ["37"]
    assert ungrounded_numbers("You slept 6.1 hours", SNAP) == ["6.1"]
    assert ungrounded_numbers("Member 42 is great", SNAP) == ["42"]     # digits of the member_id do not count
    assert ungrounded_numbers("since 2026-09-27", SNAP)                 # a date that is NOT in the snapshot


def test_dates_from_snapshot_are_ok():
    assert ungrounded_numbers("Last opened on 2026-09-28.", SNAP) == []


def test_action_goals_need_a_unit():
    out = {"insight": "x", "evidence": ["steps_7d: 4918 vs baseline 5182"],
           "actions": ["Walk 10 minutes after lunch.", "Sleep 8 hours."]}
    assert check_grounding(out, SNAP) == []                              # unit-attached goals are advice, allowed
    out["actions"] = ["Do 25 squats.", "Walk."]
    assert check_grounding(out, SNAP)                                    # '25' has no unit -> rejected


# ---------- medical ----------
@pytest.mark.parametrize("text,bad", [
    ("You may have insomnia.", True), ("Consider melatonin tonight.", True), ("This looks like depression.", True),
    ("Your blood pressure is probably high.", True), ("Ask for a diagnosis.", True),
    ("Try a wind-down routine before bed.", False), ("A 10 minute walk can lift your energy.", False),
])
def test_medical_blocklist(text, bad):
    out = {"insight": text, "evidence": [], "actions": []}
    assert bool(check_no_medical(out)) is bad


# ---------- output validation + retry/fallback ----------
GOOD = {"member_id": "M0042", "insight": "Your sleep was lower this week, so let's protect your evenings.",
        "evidence": ["sleep_hours_7d: 5.6 vs baseline 6.9"], "actions": ["Wind down 30 minutes before bed.", "Keep a regular bedtime."],
        "tone": "supportive", "safety_flag": False}


def test_valid_output_accepted_and_safety_flag_forced_false():
    bad_flag = dict(GOOD, safety_flag=True, member_id="M9999")
    out, problems = validate_output(json.dumps(bad_flag), SNAP)
    assert problems == [] and out["safety_flag"] is False and out["member_id"] == "M0042"


@pytest.mark.parametrize("raw", ["not json at all", "{broken", "[]", json.dumps({**GOOD, "tone": "angry"}),
                                 json.dumps({**GOOD, "actions": ["only one"]}),
                                 json.dumps({**GOOD, "insight": "One. Two. Three sentences here."})])
def test_invalid_outputs_rejected(raw):
    out, problems = validate_output(raw, SNAP)
    assert out is None and problems


def test_code_fences_are_tolerated():
    out, problems = validate_output("```json\n" + json.dumps(GOOD) + "\n```", SNAP)
    assert problems == [] and out


class ScriptedClient:
    name = "scripted"
    def __init__(self, replies): self.replies, self.calls = list(replies), 0
    def chat(self, messages):
        self.calls += 1
        return self.replies.pop(0)


def _patch_snapshot(monkeypatch):
    import src.vi.service as svc
    monkeypatch.setattr(svc, "load_snapshot", lambda mid: SNAP)


def test_retry_once_then_success(monkeypatch):
    _patch_snapshot(monkeypatch)
    c = ScriptedClient(["garbage", json.dumps(GOOD)])
    res = generate_insight("M0042", c)
    assert c.calls == 2 and res["meta"]["source"] == "llm" and res["meta"]["attempts"] == 2


def test_two_failures_fall_back_safely(monkeypatch):
    _patch_snapshot(monkeypatch)
    c = ScriptedClient(["garbage", json.dumps({**GOOD, "insight": "Sleep fell 99% this week."})])
    res = generate_insight("M0042", c)
    assert c.calls == 2 and res["meta"]["source"] == "fallback"
    assert check_grounding(res, SNAP) == [] and check_no_medical(res) == []


def test_model_crash_falls_back(monkeypatch):
    _patch_snapshot(monkeypatch)
    class Down:
        name = "down"
        def chat(self, m): raise ConnectionError("ollama is not running")
    assert generate_insight("M0042", Down())["meta"]["source"] == "fallback"


def test_fallback_is_grounded():
    fb = fallback_response(SNAP)
    assert check_grounding(fb, SNAP) == []
