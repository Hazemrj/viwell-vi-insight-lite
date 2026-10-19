"""The VI brain: load snapshot -> escalate OR (prompt -> model -> validate -> retry once -> fallback)."""
from __future__ import annotations

import json
import logging
import re
import time

from ..config import SNAPSHOT_DIR
from ..snapshot import estimate_tokens, compact_json
from .guardrails import escalation_needed, escalation_response, validate_output
from .prompt import build_messages, key_signals

log = logging.getLogger("vi")
MEMBER_ID_RE = re.compile(r"^M\d{4}$")
MAX_ATTEMPTS = 2        # first try + ONE retry


class MemberNotFound(Exception):
    pass


def load_snapshot(member_id: str) -> dict:
    if not MEMBER_ID_RE.match(member_id):               # also blocks path tricks like ../../etc
        raise MemberNotFound(member_id)
    path = SNAPSHOT_DIR / f"{member_id}.json"
    if not path.exists():
        raise MemberNotFound(member_id)
    return json.loads(path.read_text())


def fallback_response(snap: dict) -> dict:
    """Safe default when the model cannot produce a valid answer. Built from the snapshot by code, so it is always grounded."""
    signals = key_signals(snap, max_items=1)
    evidence = [signals[0]["line"]] if signals else [
        f"app_opens_per_day_30d: {snap['engagement']['app_opens_per_day']['30d']}"]
    return {
        "member_id": snap["member_id"],
        "insight": "Here is your weekly check-in: small, steady habits add up, so keep going at a pace that feels good.",
        "evidence": evidence,
        "actions": ["Take a short walk after one meal today.", "Pick a regular bedtime and keep it for the next few nights."],
        "tone": "encouraging",
        "safety_flag": False,
    }


def generate_insight(member_id: str, client) -> dict:
    t0 = time.perf_counter()
    snap = load_snapshot(member_id)
    meta = {"source": None, "attempts": 0, "backend": getattr(client, "name", "?"),
            "model": getattr(client, "model", None), "problems": []}

    # --- guardrail 1: escalation happens in code, BEFORE and instead of the model ---
    if escalation_needed(snap):
        out = escalation_response(snap)
        meta["source"] = "escalation_template"
    else:
        feedback: list[str] | None = None
        out = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            meta["attempts"] = attempt
            try:
                raw = client.chat(build_messages(snap, feedback))
                out, problems = validate_output(raw, snap)
            except Exception as e:                       # network error, timeout, model not found...
                out, problems = None, [f"model call failed: {type(e).__name__}"]
            if out is not None:
                meta["source"] = "llm"
                break
            meta["problems"] += problems
            feedback = problems
            log.warning("member=%s attempt=%d rejected: %s", member_id, attempt, problems)
        if out is None:
            out = fallback_response(snap)
            meta["source"] = "fallback"

    meta["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    log.info("member=%s source=%s attempts=%d latency_ms=%d", member_id, meta["source"], meta["attempts"], meta["latency_ms"])
    return {**out, "meta": meta}
