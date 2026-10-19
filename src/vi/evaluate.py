"""Evaluation: call POST /vi/insight for 20 members and report a results table.

Run (server must be running):   python -m src.vi.evaluate
Or without a server:            python -m src.vi.evaluate --inproc
Add --save-samples to also write the sample files used in the repo (5 snapshots, 10 responses).

The 20 members are chosen on purpose: 4 escalation cases, 3 'near misses' (2 low check-ins - must NOT
escalate) and 13 ordinary members spread across personas.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics

import requests

from ..config import OUTPUT_DIR, ROOT, SNAPSHOT_DIR
from .guardrails import ViOutput, check_grounding, check_no_medical

SEED = 7


def pick_members(n_total: int = 20) -> list[str]:
    snaps = {p.stem: json.loads(p.read_text()) for p in sorted(SNAPSHOT_DIR.glob("M*.json"))}
    low = [m for m, s in snaps.items() if s["mood"]["low_count_recent"] >= 3]
    near = [m for m, s in snaps.items() if s["mood"]["low_count_recent"] == 2]
    rest = [m for m in snaps if m not in low and m not in near]
    rng = random.Random(SEED)
    chosen = rng.sample(low, 4) + rng.sample(near, 3)
    by_persona: dict[str, list[str]] = {}
    for m in rest:
        by_persona.setdefault(snaps[m]["ml"]["persona"], []).append(m)
    pool = [rng.choice(v) for v in by_persona.values()]            # one per persona first
    others = [m for m in rest if m not in pool]
    pool += rng.sample(others, n_total - len(chosen) - len(pool))
    return chosen + pool[: n_total - len(chosen)]


def call_endpoint(member_id: str, base_url: str, inproc_client=None) -> dict:
    if inproc_client:
        r = inproc_client.post("/vi/insight", json={"member_id": member_id})
    else:
        r = requests.post(f"{base_url}/vi/insight", json={"member_id": member_id}, timeout=300)
    r.raise_for_status()
    return r.json()


def expected_escalation(snap: dict) -> bool:
    """Independent re-statement of the rule, straight from the brief: >=3 recent check-ins of 2 or below."""
    return sum(v <= 2 for v in snap["mood"]["recent_checkins"]) >= 3


def evaluate(responses: dict[str, dict]) -> dict:
    n = len(responses)
    valid = grounded = esc_ok = first_try_valid = 0
    n_llm_cases = 0
    latencies = []
    for mid, res in responses.items():
        snap = json.loads((SNAPSHOT_DIR / f"{mid}.json").read_text())
        core = {k: v for k, v in res.items() if k != "meta"}
        try:
            ViOutput(**core)
            valid += 1
        except Exception:
            pass
        if not check_grounding(core, snap) and not check_no_medical(core):
            grounded += 1
        esc_ok += int(res["safety_flag"] == expected_escalation(snap))
        latencies.append(res["meta"]["latency_ms"])
        if res["meta"]["source"] != "escalation_template":
            n_llm_cases += 1
            first_try_valid += int(res["meta"]["attempts"] == 1 and res["meta"]["source"] == "llm")
    sources = {s: sum(r["meta"]["source"] == s for r in responses.values()) for s in ("llm", "escalation_template", "fallback")}
    return {
        "members_tested": n,
        "valid_json_rate": valid / n,
        "grounding_pass_rate": grounded / n,
        "escalation_correctness": esc_ok / n,
        "avg_latency_ms": round(statistics.mean(latencies)),
        "p95_latency_ms": round(sorted(latencies)[max(0, int(0.95 * n) - 1)]),
        "model_valid_and_grounded_first_try": first_try_valid / max(1, n_llm_cases),
        "sources": sources,
        "retries_used": sum(r["meta"]["attempts"] > 1 for r in responses.values()),
    }


def to_markdown(res: dict, backend: str, model: str | None) -> str:
    p = lambda x: f"{100 * x:.0f}%"
    banner = ""
    if backend == "mock":
        banner = ("> **DEV RUN WITH THE OFFLINE MOCK BACKEND - NOT A REAL LLM.** "
                  "Re-run with Ollama (see README) to get the real results.\n\n")
    return (
        f"{banner}**Backend:** `{backend}`  **Model:** `{model}`  **Members tested:** {res['members_tested']}\n\n"
        "| Metric | Result |\n|---|---|\n"
        f"| Valid JSON rate (final response, after retry/fallback) | {p(res['valid_json_rate'])} |\n"
        f"| Grounding pass rate (every number in the snapshot + no medical wording) | {p(res['grounding_pass_rate'])} |\n"
        f"| Escalation correctness (safety_flag == rule) | {p(res['escalation_correctness'])} |\n"
        f"| Average latency | {res['avg_latency_ms']} ms (p95 {res['p95_latency_ms']} ms) |\n"
        f"| Model answer accepted on the 1st try (non-escalation cases) | {p(res['model_valid_and_grounded_first_try'])} |\n"
        f"| Retries used / fallbacks used | {res['retries_used']} / {res['sources']['fallback']} |\n"
        f"| Escalation template used | {res['sources']['escalation_template']} |\n"
    )


def save_samples(responses: dict[str, dict], backend: str) -> None:
    snaps = {m: json.loads((SNAPSHOT_DIR / f"{m}.json").read_text()) for m in responses}
    esc = [m for m in responses if responses[m]["safety_flag"]]
    near = [m for m in responses if snaps[m]["mood"]["low_count_recent"] == 2]
    others = [m for m in responses if m not in esc and m not in near]
    ten = esc[:2] + near[:1] + others[:7]
    d = ROOT / "samples" / "vi_insights"
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.json"):
        f.unlink()
    for m in ten:
        (d / f"{m}.json").write_text(json.dumps(responses[m], indent=2))
    if backend == "mock":
        (d / "README.md").write_text("These 10 responses were produced by the OFFLINE MOCK backend (a test double, not an LLM) "
                                     "during development. Run `python -m src.vi.evaluate --save-samples` with Ollama running "
                                     "to replace them with real model output.\n")
    else:
        (d / "README.md").write_text(f"Real responses from the local model (`{responses[ten[0]]['meta']['model']}`).\n")
    # 5 snapshots: one escalation case, one per persona where possible, highest risk
    sd = ROOT / "samples" / "snapshots"
    sd.mkdir(parents=True, exist_ok=True)
    for f in sd.glob("*.json"):
        f.unlink()
    all_snaps = {p.stem: json.loads(p.read_text()) for p in SNAPSHOT_DIR.glob("M*.json")}
    picks = [esc[0]] if esc else []
    seen = set()
    for m, s in sorted(all_snaps.items(), key=lambda kv: -(kv[1]["ml"]["disengagement_risk"] or 0)):
        if s["ml"]["persona"] not in seen and m not in picks:
            seen.add(s["ml"]["persona"]); picks.append(m)
        if len(picks) == 5:
            break
    for m in picks:
        shutil.copy(SNAPSHOT_DIR / f"{m}.json", sd / f"{m}.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--inproc", action="store_true", help="call the app in-process (no server needed)")
    ap.add_argument("--save-samples", action="store_true")
    args = ap.parse_args()

    inproc = None
    if args.inproc:
        from fastapi.testclient import TestClient
        from .app import app
        inproc = TestClient(app)
        health = inproc.get("/health").json()
    else:
        health = requests.get(f"{args.base_url}/health", timeout=10).json()
    members = pick_members()
    responses = {m: call_endpoint(m, args.base_url, inproc) for m in members}
    res = evaluate(responses)
    md = to_markdown(res, health["backend"], health["model"])
    OUTPUT_DIR.mkdir(exist_ok=True)
    tag = health["backend"]
    (OUTPUT_DIR / f"eval_results_{tag}.md").write_text(md)
    (OUTPUT_DIR / f"eval_results_{tag}.json").write_text(json.dumps(res, indent=2))
    (OUTPUT_DIR / f"eval_responses_{tag}.json").write_text(json.dumps(responses, indent=2))
    print(md)
    if args.save_samples:
        save_samples(responses, health["backend"])
        (ROOT / "samples" / f"eval_results_{tag}.md").write_text(md)
        print("Sample files written to samples/")


if __name__ == "__main__":
    main()
