"""PART 2 - Cleaning, feature engineering and member snapshots.

Run:  python -m src.part2_pipeline
Reads the 3 MySQL tables -> cleans -> builds features as of day 90 ->
writes table `member_features` and one JSON file per member in data/snapshots/.
"""
from __future__ import annotations

import json

import pandas as pd

from .config import N_DAYS, OUTPUT_DIR, SNAPSHOT_DIR
from .db import get_engine
from .features import clean_metrics, compute_features
from .snapshot import build_snapshot, estimate_tokens, compact_json, validate_snapshot, write_snapshot


def load_raw(engine):
    members = pd.read_sql("SELECT * FROM members", engine)
    metrics = pd.read_sql("SELECT * FROM daily_metrics", engine)
    engagement = pd.read_sql("SELECT * FROM engagement", engine)
    metrics["date"] = pd.to_datetime(metrics["date"]).dt.date
    engagement["date"] = pd.to_datetime(engagement["date"]).dt.date
    for c in ["steps", "active_minutes", "sleep_hours", "resting_hr"]:
        metrics[c] = metrics[c].astype(float)
    engagement["mood_checkin"] = engagement["mood_checkin"].astype(float)
    return members, metrics, engagement


def main() -> None:
    engine = get_engine()
    members, metrics, engagement = load_raw(engine)

    metrics_clean, report = clean_metrics(metrics)
    feats = compute_features(members, metrics_clean, engagement, as_of_day=N_DAYS)

    # store the feature table in MySQL (a mini "feature store")
    to_db = feats.drop(columns=["mood_recent_checkins"]).copy()
    to_db["mood_recent_checkins"] = feats["mood_recent_checkins"].apply(json.dumps)
    to_db.to_sql("member_features", engine, if_exists="replace", index=False)

    # one JSON snapshot per member (ML fields are filled in by Part 3)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    m_idx = members.set_index("member_id")
    sizes = []
    for _, row in feats.iterrows():
        snap = build_snapshot(row, m_idx.loc[row["member_id"]])
        validate_snapshot(snap)
        write_snapshot(snap)
        sizes.append(estimate_tokens(compact_json(snap)))

    report.update({"members": len(feats), "snapshot_tokens_max": max(sizes),
                   "snapshot_tokens_mean": round(sum(sizes) / len(sizes))})
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "cleaning_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {len(feats)} snapshots to {SNAPSHOT_DIR}. Next: python -m src.part3_ml")


if __name__ == "__main__":
    main()
