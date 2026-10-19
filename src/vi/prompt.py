"""Prompt design for VI.

Design choices (explained in the README):
* The CODE picks the facts worth talking about ("signals") and pre-writes them in the exact
  evidence format. The model only has to write friendly language around them. This is much
  more reliable for small (3B) models than asking them to find and compute facts themselves.
* The model is told the output schema, the allowed tones and the forbidden topics.
* The member snapshot is passed as compact JSON between clear markers.
"""
from __future__ import annotations

import json

from ..snapshot import compact_json

SYSTEM_PROMPT = """You are VI, the friendly wellbeing coach inside the VIWELL app.
You write short, personal, safe coaching insights for ONE member, using ONLY the data given.

HARD RULES
1. Use only numbers that appear in the SNAPSHOT or FACTS. Never invent, estimate or calculate new numbers.
2. No medical content: no diagnoses, no named conditions, no medication or supplement advice, no words like "disorder" or "symptom".
3. Be kind and non-judgemental. Never shame the member. Do not mention risk scores or machine-learning labels.
4. Reply with ONE JSON object and nothing else."""

OUTPUT_SPEC = """Return exactly this JSON shape:
{
  "member_id": "<copy from snapshot>",
  "insight": "ONE or TWO short sentences about a trend in the member's data",
  "evidence": ["copy 1-2 lines from FACTS exactly as written"],
  "actions": ["small, specific action for this week", "second small, specific action"],
  "tone": "one of: encouraging | supportive | celebratory | neutral",
  "safety_flag": false
}
Actions: exactly 2, each under 20 words, doable this week. If you mention an amount, attach a unit (e.g. "10 minutes")."""


def _fmt(v):
    return "n/a" if v is None else str(v)


def key_signals(snap: dict, max_items: int = 3, min_abs_pct: float = 5.0) -> list[dict]:
    """The biggest week-vs-baseline changes, as ready-to-copy evidence lines."""
    items = []
    names = {"steps": "steps", "active_minutes": "active_minutes", "sleep_hours": "sleep_hours", "resting_hr": "resting_hr"}
    for key, label in names.items():
        t = snap["metrics"][key]
        if t["7d"] is not None and t["30d"] is not None and t["chg_pct"] is not None:
            items.append((label, t))
    ao = snap["engagement"]["app_opens_per_day"]
    if ao["7d"] is not None and ao["30d"] is not None and ao["chg_pct"] is not None:
        items.append(("app_opens_per_day", ao))
    items = [(l, t) for l, t in items if abs(t["chg_pct"]) >= min_abs_pct]
    items.sort(key=lambda it: abs(it[1]["chg_pct"]), reverse=True)
    out = []
    for label, t in items[:max_items]:
        # for sleep/steps/activity/app use: down is "worse"; for resting HR: up is "worse"
        worse = t["chg_pct"] > 0 if label == "resting_hr" else t["chg_pct"] < 0
        out.append({"metric": label, "direction": "worse" if worse else "better",
                    "line": f"{label}_7d: {_fmt(t['7d'])} vs baseline {_fmt(t['30d'])} ({t['chg_pct']:+.1f}%)"})
    return out


def suggest_tone(snap: dict, signals: list[dict]) -> str:
    worse = sum(1 for s in signals if s["direction"] == "worse")
    better = sum(1 for s in signals if s["direction"] == "better")
    if worse > better:
        return "supportive"
    if better > worse:
        return "celebratory" if better >= 2 else "encouraging"
    return "encouraging"


def build_messages(snap: dict, feedback: list[str] | None = None) -> list[dict]:
    signals = key_signals(snap)
    facts = "\n".join(f"- {s['line']}" for s in signals) or "- (no big changes this week: everything is close to the member's usual level)"
    ml = snap.get("ml", {})
    context = (f"Member persona (for tone only, never mention it): {ml.get('persona')}. "
               f"Job type: {snap['profile']['job_type']}.")
    user = (
        f"{context}\n\nFACTS (you may copy these exactly):\n{facts}\n\n"
        f"Suggested tone: {suggest_tone(snap, signals)}\n\n"
        f"SNAPSHOT:\n{compact_json(snap)}\n\n{OUTPUT_SPEC}"
    )
    if feedback:
        user += ("\n\nYour previous answer was rejected for these reasons - fix them and answer again:\n- "
                 + "\n- ".join(feedback))
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
