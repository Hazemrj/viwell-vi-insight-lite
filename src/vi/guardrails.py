"""All safety checks live HERE, in plain Python - not only in the prompt.

1. escalation_needed()   - repeated low mood -> supportive message, no coaching, no LLM
2. check_grounding()     - every number in the answer must exist in the snapshot
3. check_no_medical()    - no diagnoses, conditions or medication advice
4. validate_output()     - JSON shape, lengths, allowed tone (+ the checks above)
"""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, ValidationError, field_validator

ALLOWED_TONES = {"encouraging", "supportive", "celebratory", "neutral"}
LOW_MOOD = 2
LOW_MOOD_MIN_COUNT = 3

# --------------------------------------------------------------------------
# 1. Escalation
# --------------------------------------------------------------------------
def escalation_needed(snapshot: dict) -> bool:
    """True when 3 or more of the member's recent check-ins (the last 5 in the last
    14 days) are 2 or below. Computed from the raw list, not from a precomputed flag."""
    checkins = snapshot.get("mood", {}).get("recent_checkins") or []
    return sum(1 for v in checkins if isinstance(v, (int, float)) and v <= LOW_MOOD) >= LOW_MOOD_MIN_COUNT


def escalation_response(snapshot: dict) -> dict:
    """Fixed, human-reviewed wording. Contains no diagnosis and no coaching tips."""
    values = ", ".join(str(v) for v in snapshot["mood"]["recent_checkins"])
    return {
        "member_id": snapshot["member_id"],
        "insight": "Your recent mood check-ins have been low, and that matters more than any step or sleep goal this week. "
                   "You don't have to work through this alone.",
        "evidence": [f"recent_mood_checkins: {values}"],
        "actions": [
            "Please consider talking with a qualified professional, such as your doctor or a counsellor, this week.",
            "Reach out to someone you trust, and use your company's confidential support service if you have one.",
        ],
        "tone": "supportive",
        "safety_flag": True,
    }


# --------------------------------------------------------------------------
# 2. Grounding: every number must come from the snapshot
# --------------------------------------------------------------------------
_NUM = re.compile(r"(?<![A-Za-z_\d.])-?\d[\d,]*(?:\.\d+)?")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# in ACTIONS a number is a suggested goal ("walk 10 minutes"), not a claim about the member.
# It is allowed only when attached to one of these units.
_ACTION_UNITS = r"(?:min(?:ute)?s?|hours?|hrs?|days?|nights?|times?|glass(?:es)?|steps?|breaks?)"
_ACTION_NUM = re.compile(rf"(?<![A-Za-z_\d.])\d[\d,]*(?:\.\d+)?(?=\s*-?\s*{_ACTION_UNITS}\b)", re.I)


def flatten_numbers(obj) -> list[float]:
    """All numeric values in the snapshot (booleans excluded) plus numbers inside its date strings."""
    out: list[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        if _DATE.fullmatch(obj):                                   # "2026-09-28" -> 2026, 9, 28
            out += [float(x) for x in re.findall(r"\d+", obj)]    # (ids like "M0042" are NOT citable numbers)
    elif isinstance(obj, dict):
        for v in obj.values():
            out += flatten_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += flatten_numbers(v)
    return out


def _decimals(token: str) -> int:
    return len(token.split(".")[1]) if "." in token else 0


def number_is_grounded(token: str, snapshot_numbers: list[float]) -> bool:
    """A token is OK if it equals a snapshot number (sign ignored, so 'down 18.4%' matches -18.4),
    or equals a snapshot number rounded to the token's own precision ('18%' matches 18.4)."""
    try:
        x = abs(float(token.replace(",", "")))
    except ValueError:
        return False
    nd = _decimals(token)
    for v in snapshot_numbers:
        if abs(v) == x or round(abs(v), nd) == round(x, nd):
            return True
    return False


def ungrounded_numbers(text: str, snapshot: dict, *, is_action: bool = False) -> list[str]:
    allowed = flatten_numbers(snapshot)
    # a date written exactly as in the snapshot (e.g. last_active_date) is fine: remove it first
    snapshot_strings = [s for s in _all_strings(snapshot) if _DATE.fullmatch(s)]
    for d in snapshot_strings:
        text = text.replace(d, " ")
    bad = []
    skip_spans: list[tuple[int, int]] = []
    if is_action:                                    # suggested goals with a unit are allowed
        skip_spans = [m.span() for m in _ACTION_NUM.finditer(text)]
    for m in _NUM.finditer(text):
        if any(s <= m.start() < e for s, e in skip_spans):
            continue
        if not number_is_grounded(m.group(0), allowed):
            bad.append(m.group(0))
    return bad


def _all_strings(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _all_strings(v)]
    if isinstance(obj, (list, tuple)):
        return [s for v in obj for s in _all_strings(v)]
    return []


def check_grounding(out: dict, snapshot: dict) -> list[str]:
    """Return a list of problems (empty list = grounded)."""
    problems = []
    for field in ("insight",):
        bad = ungrounded_numbers(out.get(field, ""), snapshot)
        if bad:
            problems.append(f"{field} contains numbers not in the snapshot: {bad}")
    for i, ev in enumerate(out.get("evidence", [])):
        bad = ungrounded_numbers(ev, snapshot)
        if bad:
            problems.append(f"evidence[{i}] contains numbers not in the snapshot: {bad}")
    for i, act in enumerate(out.get("actions", [])):
        bad = ungrounded_numbers(act, snapshot, is_action=True)
        if bad:
            problems.append(f"actions[{i}] contains numbers not in the snapshot: {bad}")
    return problems


# --------------------------------------------------------------------------
# 3. No medical claims
# --------------------------------------------------------------------------
_MEDICAL_PATTERNS = [
    r"diagnos\w*", r"\bdepress\w*", r"\banxi\w*", r"\binsomnia\b", r"\bsleep apn\w*", r"\bhypertension\b",
    r"\bblood pressure\b", r"\bdiabet\w*", r"\bcardiac\b", r"\barrhythmia\b", r"\bheart (?:disease|condition|problem|attack)\b",
    r"\bdisorders?\b", r"\bsyndrome\b", r"\bcondition\b", r"\bdisease\b", r"\bclinical\w*", r"\bmedicat\w*",
    r"\bmedicine\b", r"\bprescri\w*", r"\bdos(?:e|age)\b", r"\bmg\b", r"\bsupplements?\b", r"\bmelatonin\b",
    r"\bantidepress\w*", r"\bcure\b", r"\btreat(?:ment|ing)?\b", r"\bsymptoms?\b", r"\bpainkillers?\b", r"\bpills?\b",
    r"\byou (?:probably |likely |may |might )?have (?:a |an )?[a-z]+ (?:problem|issue)\b",
]
_MEDICAL_RE = re.compile("|".join(_MEDICAL_PATTERNS), re.I)


def check_no_medical(out: dict) -> list[str]:
    text = " ".join([out.get("insight", ""), *out.get("evidence", []), *out.get("actions", [])])
    hits = sorted({m.group(0).lower() for m in _MEDICAL_RE.finditer(text)})
    return [f"medical wording not allowed: {hits}"] if hits else []


# --------------------------------------------------------------------------
# 4. Output validation
# --------------------------------------------------------------------------
class ViOutput(BaseModel):
    member_id: str
    insight: str = Field(min_length=10, max_length=300)
    evidence: list[str] = Field(min_length=1, max_length=3)
    actions: list[str] = Field(min_length=2, max_length=2)
    tone: str
    safety_flag: bool = False

    @field_validator("tone")
    @classmethod
    def _tone(cls, v):
        if v not in ALLOWED_TONES:
            raise ValueError(f"tone must be one of {sorted(ALLOWED_TONES)}")
        return v

    @field_validator("insight")
    @classmethod
    def _two_sentences(cls, v):
        n = len([s for s in re.split(r"(?<=[.!?])\s+", v.strip()) if s])
        if n > 2:
            raise ValueError("insight must be one or two sentences")
        return v

    @field_validator("evidence", "actions")
    @classmethod
    def _items(cls, v):
        for s in v:
            if not isinstance(s, str) or not s.strip() or len(s) > 160:
                raise ValueError("each list item must be a non-empty string of at most 160 characters")
        return v


def parse_json_loosely(raw: str) -> dict:
    """Models sometimes wrap JSON in ```json fences or add chatter. Take the first {...} block."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(text[start:end + 1])


def validate_output(raw: str, snapshot: dict) -> tuple[dict | None, list[str]]:
    """Returns (clean_output, problems). clean_output is None if anything failed."""
    try:
        data = parse_json_loosely(raw)
    except (ValueError, json.JSONDecodeError) as e:
        return None, [f"invalid JSON: {e}"]
    if not isinstance(data, dict):
        return None, ["JSON is not an object"]
    # the service, not the model, decides identity and the safety flag
    data["member_id"] = snapshot["member_id"]
    data["safety_flag"] = False
    try:
        out = ViOutput(**{k: data.get(k) for k in ViOutput.model_fields}).model_dump()
    except ValidationError as e:
        return None, [f"schema: {err['loc'][0]}: {err['msg']}" for err in e.errors()]
    problems = check_grounding(out, snapshot) + check_no_medical(out)
    return (None, problems) if problems else (out, [])
