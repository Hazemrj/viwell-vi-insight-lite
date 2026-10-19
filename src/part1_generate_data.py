"""PART 1 - Synthetic data generator.

Creates 500 fake members x 90 days of wearable + app data and loads it into MySQL.
Run:  python -m src.part1_generate_data

Every "real-life" pattern is created on purpose in this file so that it can be
documented and later detected in the notebooks. The patterns are listed in the
PATTERNS string at the bottom (also copied into the README).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import N_DAYS, N_MEMBERS, SEED, START_DATE
from .db import get_engine, run_sql_file

# ---------------------------------------------------------------------------
# 1. "Archetypes": five broad kinds of people. Each has typical daily habits.
#    These are the hidden groups the clustering in Part 3 should rediscover.
# ---------------------------------------------------------------------------
ARCHETYPES = {
    #                 steps  active_min sleep  app/day content/day challenge/day mood
    "achiever":      dict(steps=10500, act=55, sleep=7.3, app=3.5, content=2.0, chal=0.030, mood=3.9, p=0.25),
    "steady":        dict(steps=7500,  act=35, sleep=7.0, app=1.8, content=0.8, chal=0.012, mood=3.5, p=0.30),
    "desk_curious":  dict(steps=4500,  act=15, sleep=6.8, app=2.5, content=1.6, chal=0.015, mood=3.3, p=0.20),
    "silent_mover":  dict(steps=9500,  act=50, sleep=7.1, app=0.4, content=0.1, chal=0.002, mood=3.6, p=0.13),
    "dabbler":       dict(steps=6000,  act=22, sleep=6.9, app=0.25, content=0.1, chal=0.002, mood=3.2, p=0.12),
}
COMPANIES = [f"C{i:02d}" for i in range(1, 9)]          # 8 companies
COMPANY_P = [0.22, 0.18, 0.15, 0.13, 0.11, 0.09, 0.07, 0.05]
AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55+"]
AGE_P = [0.08, 0.32, 0.30, 0.20, 0.10]
GENDERS = ["female", "male", "undisclosed"]
GENDER_P = [0.42, 0.55, 0.03]
JOB_TYPES = ["desk", "field", "shift"]
JOB_P = [0.55, 0.25, 0.20]

DISENGAGED_SHARE = 0.15      # 15% of members fade out
N_LOW_MOOD = 14              # members whose latest check-ins are low  -> must trigger escalation
N_NEAR_MISS = 8              # members with exactly 2 low of last 5    -> must NOT trigger escalation
N_OUTLIERS = 30              # obviously wrong sensor readings to clean


def make_members(rng: np.random.Generator) -> pd.DataFrame:
    n = N_MEMBERS
    df = pd.DataFrame({
        "member_id": [f"M{i:04d}" for i in range(1, n + 1)],
        "company_id": rng.choice(COMPANIES, n, p=COMPANY_P),
        "age_band": rng.choice(AGE_BANDS, n, p=AGE_P),
        "gender": rng.choice(GENDERS, n, p=GENDER_P),
        "job_type": rng.choice(JOB_TYPES, n, p=JOB_P),
        "join_date": [(pd.Timestamp(START_DATE) - pd.Timedelta(days=int(d))).date()
                      for d in rng.integers(30, 720, n)],
    })
    names = list(ARCHETYPES)
    df["_archetype"] = rng.choice(names, n, p=[ARCHETYPES[k]["p"] for k in names])
    return df


def pick_special_groups(members: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Mark who disengages / who has low mood. Kept in hidden '_' columns (not stored in DB)."""
    n = len(members)
    members["_disengaged"] = False
    members["_low_mood"] = False
    members["_near_miss"] = False

    # disengaging members: never the 'dabbler' type (they are low-engagement already)
    eligible = members.index[members["_archetype"] != "dabbler"].to_numpy()
    dis = rng.choice(eligible, int(round(DISENGAGED_SHARE * n)), replace=False)
    members.loc[dis, "_disengaged"] = True

    # low-mood / near-miss members: must be engaged enough to check in, and not fading
    pool = members.index[(~members["_disengaged"]) & (members["_archetype"].isin(["achiever", "steady", "desk_curious"]))].to_numpy()
    chosen = rng.choice(pool, N_LOW_MOOD + N_NEAR_MISS, replace=False)
    members.loc[chosen[:N_LOW_MOOD], "_low_mood"] = True
    members.loc[chosen[N_LOW_MOOD:], "_near_miss"] = True
    return members


# Hand-written mood endings for the last 6 days (None = no check-in that day).
# Only the last 5 *check-ins* matter for the escalation rule.
LOW_MOOD_PATTERNS = [
    [2, 1, 3, 2, 2, 1], [3, 2, 2, 2, 3, 2], [1, 2, 1, 3, 2, 3],
    [2, 2, 2, 4, 3, 3], [3, 1, 2, 2, 4, 2], [2, 3, 1, 1, 2, 2], [4, 2, 2, 3, 1, 3],
]
NEAR_MISS_PATTERNS = [   # exactly two values <= 2 among the last 5 check-ins
    [3, 2, 3, 2, 4, 3], [4, 3, 2, 3, 2, 4], [3, 3, 2, 4, 2, 3], [2, 4, 3, 3, 2, 4],
]


def simulate_member(m: pd.Series, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate 90 days of metrics + engagement for one member."""
    a = ARCHETYPES[m["_archetype"]]
    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")
    dow = dates.dayofweek.to_numpy()               # Mon=0 ... Sun=6
    weekend = dow >= 5                             # Sat/Sun  (assumption: UAE federal weekend)
    day = np.arange(1, N_DAYS + 1)
    job = m["job_type"]

    # personal quirks: everyone is a bit different from their archetype
    steps_mean = a["steps"] * rng.lognormal(0, 0.15) * (1.25 if job == "field" else 1.0)
    act_mean = a["act"] * rng.lognormal(0, 0.15) * (1.2 if job == "field" else 1.0)
    sleep_mean = a["sleep"] + rng.normal(0, 0.3)
    hr_base = float(np.clip(rng.normal(66, 6) - 0.0006 * (steps_mean - 7000), 48, 85))
    app_rate = a["app"] * rng.lognormal(0, 0.25)
    content_rate = a["content"] * rng.lognormal(0, 0.25)
    mood_mean = a["mood"] + rng.normal(0, 0.35)

    # --- disengagement curve m(d): 1 = fully engaged -> 0 = gone -------------
    mult = np.ones(N_DAYS)
    if m["_disengaged"]:
        fade_start = int(rng.integers(10, 56))
        gone_day = fade_start + int(rng.integers(15, 51))     # may fall after day 90 -> never fully gone
        mult = np.clip(1 - (day - fade_start) / (gone_day - fade_start), 0, 1)

    # --- steps: weekday vs weekend ------------------------------------------
    wk_steps = {"desk": 1.12, "field": 0.65, "shift": 0.90}[job]
    steps = steps_mean * np.where(weekend, wk_steps, 1.0) * rng.lognormal(0, 0.22, N_DAYS)
    steps = steps * (0.6 + 0.4 * mult)                          # disengaging people also move less
    active = act_mean * np.where(weekend, wk_steps, 1.0) * rng.lognormal(0, 0.30, N_DAYS) * (0.6 + 0.4 * mult)

    # --- sleep: weekends longer; shift workers irregular ---------------------
    if job == "shift":
        sleep = np.empty(N_DAYS)
        d = 0
        while d < N_DAYS:                                       # rotating blocks of day / night shifts
            block = int(rng.integers(4, 9))
            block_mean = rng.choice([7.0, 5.5])
            sleep[d:d + block] = block_mean + rng.normal(0, 1.3, min(block, N_DAYS - d))
            d += block
    else:
        bump = np.where(weekend, 0.8 if job == "desk" else 0.5, 0.0)
        sleep = sleep_mean + bump + rng.normal(0, 0.6, N_DAYS)
    sleep = np.clip(sleep, 3.0, 11.0)

    hr = hr_base + 1.5 * np.maximum(0, 7 - sleep) + rng.normal(0, 1.8, N_DAYS)

    # --- wearable sync gaps: ~7% of days, often 1-2 days in a row -----------
    miss_rate = float(np.clip(rng.beta(2, 26), 0.02, 0.14))
    miss_rate = miss_rate * (1 + 1.5 * (1 - mult))              # fading members sync less
    start_miss = rng.random(N_DAYS) < miss_rate * 0.8
    synced = np.ones(N_DAYS, bool)
    for i in np.flatnonzero(start_miss):
        synced[i] = False
        if rng.random() < 0.25 and i + 1 < N_DAYS:               # sometimes a 2-day gap
            synced[i + 1] = False

    metrics = pd.DataFrame({
        "member_id": m["member_id"], "date": dates.date,
        "steps": np.round(steps).astype(float),
        "active_minutes": np.round(active).astype(float),
        "sleep_hours": np.round(sleep, 1),
        "resting_hr": np.round(hr).astype(float),
        "wearable_synced": np.where(synced, "yes", "no"),
    })
    metrics.loc[~synced, ["steps", "active_minutes", "sleep_hours", "resting_hr"]] = np.nan

    # --- engagement ----------------------------------------------------------
    app_lambda = app_rate * mult * np.where(weekend, 0.85, 1.0)
    opens = rng.poisson(app_lambda)
    views = np.where(opens > 0, rng.poisson(content_rate * mult), 0)
    chal = rng.random(N_DAYS) < a["chal"] * mult * 1.0
    p_checkin = np.minimum(0.5, 0.12 + 0.08 * app_rate) * mult
    checks = rng.random(N_DAYS) < p_checkin
    mood_val = np.clip(np.round(mood_mean - 1.0 * (1 - mult) + rng.normal(0, 0.8, N_DAYS)), 1, 5)
    mood = np.where(checks & (opens > 0), mood_val, np.nan)

    # planted mood patterns for the escalation tests
    if m["_low_mood"] or m["_near_miss"]:
        pats = LOW_MOOD_PATTERNS if m["_low_mood"] else NEAR_MISS_PATTERNS
        mood[-6:] = pats[int(rng.integers(len(pats)))]           # last 6 days all have check-ins
        mood[:-6] = np.where(rng.random(N_DAYS - 6) < 0.3, np.clip(np.round(rng.normal(3.4, 0.7, N_DAYS - 6)), 2, 5), np.nan)
        opens[-6:] = np.maximum(opens[-6:], 1)

    engagement = pd.DataFrame({
        "member_id": m["member_id"], "date": dates.date,
        "app_opens": opens, "content_views": views,
        "challenge_joined": np.where(chal, "yes", "no"),
        "mood_checkin": mood,
    })
    return metrics, engagement


def inject_outliers(metrics: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Put obviously impossible sensor readings into a few synced rows."""
    rows = rng.choice(metrics.index[metrics["wearable_synced"] == "yes"], N_OUTLIERS, replace=False)
    bad = [("steps", 150000), ("steps", 99999), ("sleep_hours", 23.5), ("sleep_hours", 0.0),
           ("resting_hr", 0.0), ("resting_hr", 220.0), ("active_minutes", 1500.0)]
    for i, r in enumerate(rows):
        col, val = bad[i % len(bad)]
        metrics.loc[r, col] = val
    return metrics


def generate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    members = pick_special_groups(make_members(rng), rng)
    ms, es = [], []
    for _, m in members.iterrows():
        a, b = simulate_member(m, rng)
        ms.append(a)
        es.append(b)
    metrics = inject_outliers(pd.concat(ms, ignore_index=True), rng)
    engagement = pd.concat(es, ignore_index=True)
    truth = members[["member_id", "_archetype", "_disengaged", "_low_mood", "_near_miss"]].copy()
    return members.drop(columns=[c for c in members if c.startswith("_")]), metrics, engagement, truth


def main() -> None:
    members, metrics, engagement, truth = generate()
    engine = get_engine()
    run_sql_file(engine, "01_schema.sql")
    members.to_sql("members", engine, if_exists="append", index=False)
    metrics.to_sql("daily_metrics", engine, if_exists="append", index=False, chunksize=5000, method="multi")
    engagement.to_sql("engagement", engine, if_exists="append", index=False, chunksize=5000, method="multi")

    # The hidden "answer key" is saved ONLY for validating the project in notebooks.
    # It is not used by the pipeline or by the model.
    from .config import DATA_DIR
    DATA_DIR.mkdir(exist_ok=True)
    truth.to_csv(DATA_DIR / "synthetic_ground_truth.csv", index=False)

    synced_rate = (metrics["wearable_synced"] == "yes").mean()
    print(f"members={len(members)}  daily_metrics={len(metrics)}  engagement={len(engagement)}")
    print(f"wearable not synced: {100 * (1 - synced_rate):.1f}% of days")
    print(f"disengaging members: {int(truth['_disengaged'].sum())} ({100 * truth['_disengaged'].mean():.0f}%)")
    print("Loaded into MySQL. Next: python -m src.part2_pipeline")


PATTERNS = """
1. Weekday vs weekend: desk workers +12% steps and +0.8h sleep on Sat/Sun; field workers -35% steps, +0.5h sleep; shift workers -10% steps, no sleep bump.
2. Shift workers: sleep follows rotating 4-8 day blocks of 'day shift' (~7.0h) and 'night shift' (~5.5h), with a large daily spread (sd 1.3h).
3. Disengagement: 15% of members (75) fade. Their app opens, content views, challenges, mood check-ins, steps and syncing decline linearly from a start day (day 10-55) to zero 15-50 days later; some are fully gone before day 77, some are still partly active.
4. Missing sync: each member skips ~2-14% of days (overall ~7-9%), sometimes 2 days in a row; all four wearable values are NULL on those days.
5. Outliers: 30 impossible readings (150000 steps, 23.5h sleep, 0 bpm, 220 bpm, 1500 active minutes...).
6. Five hidden archetypes with different step / sleep / app habits (the personas Part 3 should rediscover).
7. Mood: 14 members end with repeated low check-ins (must escalate); 8 'near-miss' members have exactly two low check-ins in their last five (must NOT escalate).
"""

if __name__ == "__main__":
    main()
