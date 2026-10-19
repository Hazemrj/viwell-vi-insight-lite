"""Cleaning + feature engineering (shared by Part 2 snapshots and Part 3 ML).

The key idea: every feature is computed "as of" one day. For the member snapshot
that day is day 90. For the risk model it is day 76, so the model never peeks at
days 77-90 (that would be data leakage).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import N_DAYS, START_DATE

METRICS = ["steps", "active_minutes", "sleep_hours", "resting_hr"]

# ---- Cleaning rules (documented in README) --------------------------------
# A value outside these hard limits is physically implausible -> treated as a
# sensor error and REMOVED (set to missing). We do not "cap" it, because a capped
# fake number would still look like real data.
VALID_RANGES = {
    "steps": (0, 60_000),
    "active_minutes": (0, 600),
    "sleep_hours": (2, 14),
    "resting_hr": (35, 120),
}
ACTIVE_DAY_MINUTES = 30      # a day counts as "active" with >= 30 active minutes
LOW_MOOD = 2                 # mood <= 2 is "low"
MOOD_RECENT_N = 5            # look at the last 5 check-ins ...
MOOD_RECENT_DAYS = 14        # ... made in the last 14 days


def clean_metrics(metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rule 1: unsynced days stay missing (NaN) - never turned into 0 or filled in.
    Rule 2: values outside VALID_RANGES are removed (set to NaN)."""
    df = metrics.copy()
    report = {"rows": int(len(df)),
              "unsynced_days": int((df["wearable_synced"] == "no").sum()),
              "unsynced_pct": round(100 * float((df["wearable_synced"] == "no").mean()), 2),
              "outliers_removed": {}}
    for col, (lo, hi) in VALID_RANGES.items():
        bad = (df[col] < lo) | (df[col] > hi)
        report["outliers_removed"][col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    report["outliers_removed_total"] = int(sum(report["outliers_removed"].values()))
    return df, report


# ---- helpers ---------------------------------------------------------------
def _pivot(df: pd.DataFrame, col: str, ids: list[str]) -> np.ndarray:
    """members x 90-days matrix for one column."""
    p = df.pivot(index="member_id", columns="date", values=col).reindex(ids)
    return p.to_numpy(dtype=float)


def _nanmean(a: np.ndarray, axis=1) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _nanstd(a: np.ndarray, axis=1) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanstd(a, axis=axis)


def _pct_change(recent: np.ndarray, base: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (recent - base) / base * 100
    out[~np.isfinite(out)] = np.nan
    return out


def _streak(row: np.ndarray, ok) -> int:
    """Count back from the last day while ok(value) is True. NaN (= unknown) is skipped."""
    n = 0
    for v in row[::-1]:
        if np.isnan(v):
            continue
        if ok(v):
            n += 1
        else:
            break
    return n


def _mood_slope_per_week(days: np.ndarray, vals: np.ndarray) -> float:
    if len(vals) < 4 or len(np.unique(days)) < 2:
        return np.nan
    return float(np.polyfit(days, vals, 1)[0] * 7)


# ---- main feature function ---------------------------------------------------
def compute_features(members: pd.DataFrame, metrics_clean: pd.DataFrame,
                     engagement: pd.DataFrame, as_of_day: int = N_DAYS) -> pd.DataFrame:
    """One row per member, using ONLY days 1..as_of_day.  as_of_day is 1-based."""
    ids = sorted(members["member_id"])
    t = as_of_day            # slice end (exclusive) in 0-based indexing
    as_of_date = (pd.Timestamp(START_DATE) + pd.Timedelta(days=as_of_day - 1)).date()
    f = pd.DataFrame({"member_id": ids})
    f["as_of"] = str(as_of_date)

    # 7-day and 30-day averages + change vs own 30-day baseline ---------------
    for col in METRICS:
        m = _pivot(metrics_clean, col, ids)[:, :t]
        a7, a30 = _nanmean(m[:, -7:]), _nanmean(m[:, -30:])
        f[f"{col}_7d"], f[f"{col}_30d"] = a7, a30
        f[f"{col}_chg_pct"] = _pct_change(a7, a30)

    opens = _pivot(engagement, "app_opens", ids)[:, :t]
    views = _pivot(engagement, "content_views", ids)[:, :t]
    chal = (engagement.assign(c=(engagement["challenge_joined"] == "yes").astype(float))
            .pipe(_pivot, "c", ids))[:, :t]
    o7, o30 = opens[:, -7:].mean(axis=1), opens[:, -30:].mean(axis=1)
    f["app_opens_7d"], f["app_opens_30d"] = o7, o30
    f["app_opens_chg_pct"] = _pct_change(o7, o30)
    f["app_zero_days_14d"] = (opens[:, -14:] == 0).sum(axis=1)
    f["content_views_7d"] = views[:, -7:].sum(axis=1)
    f["content_views_30d"] = views[:, -30:].sum(axis=1)
    f["challenges_30d"] = chal[:, -30:].sum(axis=1)

    # recency & streaks ----------------------------------------------------------
    last_open_idx = np.array([np.flatnonzero(r > 0)[-1] if (r > 0).any() else -1 for r in opens])
    f["days_since_last_open"] = np.where(last_open_idx >= 0, (t - 1) - last_open_idx, t)
    f["last_active_date"] = [
        str((pd.Timestamp(START_DATE) + pd.Timedelta(days=int(i))).date()) if i >= 0 else None
        for i in last_open_idx]
    f["app_streak_days"] = [_streak(r, lambda v: v > 0) for r in opens]
    act = _pivot(metrics_clean, "active_minutes", ids)[:, :t]
    f["active_streak_days"] = [_streak(r, lambda v: v >= ACTIVE_DAY_MINUTES) for r in act]

    # consistency score 0-100: steady sleep + steady steps over 30 days ------------
    sleep30 = _pivot(metrics_clean, "sleep_hours", ids)[:, :t][:, -30:]
    steps30 = _pivot(metrics_clean, "steps", ids)[:, :t][:, -30:]
    sleep_sd = _nanstd(sleep30)
    with np.errstate(divide="ignore", invalid="ignore"):
        steps_cv = _nanstd(steps30) / _nanmean(steps30)
    s_part = 100 * (1 - np.clip(sleep_sd / 2.0, 0, 1))      # sd of 2h or more -> 0 points
    c_part = 100 * (1 - np.clip(steps_cv / 0.6, 0, 1))      # cv of 0.6 or more -> 0 points
    f["sleep_sd_30d"] = sleep_sd
    f["consistency_score"] = np.round(np.nanmean(np.vstack([s_part, c_part]), axis=0))

    # wearable sync rate -------------------------------------------------------------
    synced = _pivot(metrics_clean.assign(s=(metrics_clean["wearable_synced"] == "yes").astype(float)), "s", ids)[:, :t]
    f["sync_rate_7d"] = synced[:, -7:].mean(axis=1)
    f["sync_rate_30d"] = synced[:, -30:].mean(axis=1)

    # mood -------------------------------------------------------------------------------
    mood = _pivot(engagement, "mood_checkin", ids)[:, :t]
    avg30, slope, recent, low_n, n30 = [], [], [], [], []
    for r in mood:
        d30 = np.flatnonzero(~np.isnan(r[-30:]))
        v30 = r[-30:][d30]
        n30.append(len(v30))
        avg30.append(float(v30.mean()) if len(v30) else np.nan)
        slope.append(_mood_slope_per_week(d30, v30))
        d14 = np.flatnonzero(~np.isnan(r[-MOOD_RECENT_DAYS:]))
        last = [int(x) for x in r[-MOOD_RECENT_DAYS:][d14][-MOOD_RECENT_N:]]
        recent.append(last)
        low_n.append(sum(1 for x in last if x <= LOW_MOOD))
    f["mood_avg_30d"], f["mood_slope_per_week"], f["mood_n_30d"] = avg30, slope, n30
    f["mood_trend"] = ["insufficient_data" if np.isnan(s) else
                       ("declining" if s < -0.15 else "improving" if s > 0.15 else "stable") for s in slope]
    f["mood_recent_checkins"] = recent
    f["mood_low_recent_count"] = low_n
    return f
