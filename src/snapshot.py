"""Turn one row of features into the compact, privacy-safe JSON snapshot."""
from __future__ import annotations

import json
import math

import jsonschema
import pandas as pd

from .config import SCHEMA_DIR, SNAPSHOT_DIR

SCHEMA = json.loads((SCHEMA_DIR / "snapshot.schema.json").read_text())
MAX_TOKENS = 1500


def _num(x, nd=1):
    """Round to nd decimals; NaN -> None (JSON null). nd=0 gives an int."""
    if x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x):
        return None
    return int(round(float(x))) if nd == 0 else round(float(x), nd)


def _trend(row, prefix, nd=1, suffix=""):
    return {"7d": _num(row[f"{prefix}_7d"], nd), "30d": _num(row[f"{prefix}_30d"], nd),
            "chg_pct": _num(row[f"{prefix}_chg_pct"], 1)}


def build_snapshot(row: pd.Series, member: pd.Series, persona: str | None = None,
                   risk: float | None = None, risk_band: str | None = None) -> dict:
    """`member` supplies ONLY age_band and job_type. Nothing else leaves the database."""
    return {
        "member_id": row["member_id"],
        "as_of": row["as_of"],
        "windows_days": [7, 30],
        "profile": {"age_band": member["age_band"], "job_type": member["job_type"]},
        "metrics": {
            "steps": _trend(row, "steps", 0),
            "active_minutes": _trend(row, "active_minutes", 1),
            "sleep_hours": _trend(row, "sleep_hours", 1),
            "resting_hr": _trend(row, "resting_hr", 1),
        },
        "engagement": {
            "app_opens_per_day": {"7d": _num(row["app_opens_7d"], 1), "30d": _num(row["app_opens_30d"], 1),
                                  "chg_pct": _num(row["app_opens_chg_pct"], 1)},
            "days_since_last_open": int(row["days_since_last_open"]),
            "last_active_date": row["last_active_date"],
            "app_streak_days": int(row["app_streak_days"]),
            "content_views_7d": int(row["content_views_7d"]),
            "challenges_joined_30d": int(row["challenges_30d"]),
        },
        "activity": {"active_streak_days": int(row["active_streak_days"]),
                     "consistency_score": _num(row["consistency_score"], 0)},
        "mood": {"avg_30d": _num(row["mood_avg_30d"], 1), "trend": row["mood_trend"],
                 "recent_checkins": [int(x) for x in row["mood_recent_checkins"]],
                 "low_count_recent": int(row["mood_low_recent_count"])},
        "data_quality": {"sync_rate_30d": _num(row["sync_rate_30d"], 2)},
        "ml": {"persona": persona, "disengagement_risk": _num(risk, 2), "risk_band": risk_band},
    }


def compact_json(snap: dict) -> str:
    """The exact string that goes into the LLM prompt."""
    return json.dumps(snap, separators=(",", ":"))


def estimate_tokens(text: str) -> int:
    """Rough, deliberately pessimistic estimate: 1 token ~ 3 characters for JSON."""
    return math.ceil(len(text) / 3)


def validate_snapshot(snap: dict) -> None:
    jsonschema.validate(snap, SCHEMA)
    if estimate_tokens(compact_json(snap)) >= MAX_TOKENS:
        raise ValueError(f"{snap['member_id']}: snapshot too large for the LLM prompt")


def write_snapshot(snap: dict) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    (SNAPSHOT_DIR / f"{snap['member_id']}.json").write_text(json.dumps(snap, indent=2))
