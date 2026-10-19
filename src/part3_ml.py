"""PART 3 - ML signals: (a) personas via K-means, (b) disengagement-risk classifier.

Run:  python -m src.part3_ml
Writes: outputs/*.json + outputs/*.png, models/risk_model.joblib, and fills the
`ml` block of every snapshot in data/snapshots/.
"""
from __future__ import annotations

import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_DAYS, LABEL_DAYS, MODEL_DIR, N_DAYS, OUTPUT_DIR, SEED, SNAPSHOT_DIR
from .features import clean_metrics, compute_features
from .part2_pipeline import load_raw
from .db import get_engine
from .snapshot import build_snapshot, validate_snapshot, write_snapshot

# ----------------------------------------------------------------------------
# 3a. PERSONAS
# ----------------------------------------------------------------------------
PERSONA_FEATURES = ["steps_30d", "active_minutes_30d", "sleep_hours_30d", "consistency_score",
                    "app_opens_30d", "app_opens_7d", "content_views_30d", "mood_avg_30d"]
K_RANGE = range(2, 9)
K_ALLOWED = range(4, 7)      # product decision: 4-6 personas are actionable; k=2/3 are too coarse
K_TOLERANCE = 0.02           # parsimony: prefer the SMALLEST k whose silhouette is within 0.02 of the best


def scale_persona_matrix(feats: pd.DataFrame):
    X = feats[PERSONA_FEATURES].copy()
    X = X.fillna(X.median())
    scaler = StandardScaler().fit(X)
    return X, scaler, scaler.transform(X)


def evaluate_k(Xs: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Xs)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(Xs, km.labels_)})
    return pd.DataFrame(rows)


def choose_k(scores: pd.DataFrame) -> int:
    allowed = scores[scores["k"].isin(list(K_ALLOWED))]
    best = allowed["silhouette"].max()
    return int(allowed[allowed["silhouette"] >= best - K_TOLERANCE]["k"].min())


def name_personas(profile: pd.DataFrame, population: pd.DataFrame) -> dict[int, tuple[str, str]]:
    """Rule-based naming. Each cluster's average is compared with the whole population
    (z = how many standard deviations above/below the average member)."""
    mu, sd = population.mean(), population.std(ddof=0).replace(0, 1)
    names: dict[int, tuple[str, str]] = {}
    for c, r in profile.iterrows():
        z = (r - mu) / sd
        act, eng, cons = z["steps_30d"], z["app_opens_30d"], z["consistency_score"]
        fading = r["app_opens_30d"] > 0 and r["app_opens_7d"] < 0.5 * r["app_opens_30d"]
        if fading:
            names[c] = ("Fading Away", "Used the app before but use has dropped sharply in the last week - reach out first.")
        elif cons < -0.8:
            names[c] = ("Irregular Rhythms", "Unpredictable sleep and activity (typically shift workers) - offer flexible, routine-building tips.")
        elif act > 0.7 and eng > 0.7:
            names[c] = ("Engaged Achievers", "Very active, open the app almost daily and love content - ideal for challenges and streaks.")
        elif eng > 0.3 and act < 0:
            names[c] = ("Curious Desk-Sitters", "Like the app and its content but move less than average - a good fit for easy, small activity goals.")
        elif eng < -0.7:
            names[c] = ("Quiet Drifters", "Rarely open the app and are only moderately active - at highest risk of dropping off; need a simple hook.")
        elif act > 0.15 and eng < 0.3:
            names[c] = ("Active Independents", "Active and sleeping well, but use the app lightly - mostly self-driven; keep nudges rare and relevant.")
        else:
            names[c] = ("Steady Regulars", "Average activity and average app use - consistent, low-maintenance members.")
    seen: dict[str, int] = {}                     # keep names unique
    for c, (n, d) in list(names.items()):
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[c] = (f"{n} ({seen[n]})", d)
    return names


def fit_personas(feats: pd.DataFrame, k: int | None = None):
    X, scaler, Xs = scale_persona_matrix(feats)
    scores = evaluate_k(Xs)
    k = k or choose_k(scores)
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Xs)
    labels = km.labels_
    profile = X.assign(cluster=labels).groupby("cluster").mean()
    names = name_personas(profile, X)
    out = feats[["member_id"]].copy()
    out["persona_id"] = labels
    out["persona"] = [names[c][0] for c in labels]
    desc = {names[c][0]: names[c][1] for c in names}
    return out, profile.assign(persona=[names[c][0] for c in profile.index],
                               members=pd.Series(labels).value_counts().sort_index()), scores, k, desc, Xs, labels


# ----------------------------------------------------------------------------
# 3b. DISENGAGEMENT RISK
# ----------------------------------------------------------------------------
RISK_FEATURES = [
    "steps_7d", "steps_30d", "steps_chg_pct", "active_minutes_7d", "active_minutes_30d", "active_minutes_chg_pct",
    "sleep_hours_7d", "sleep_hours_30d", "sleep_hours_chg_pct", "resting_hr_7d", "resting_hr_30d", "resting_hr_chg_pct",
    "app_opens_7d", "app_opens_30d", "app_opens_chg_pct", "app_zero_days_14d", "days_since_last_open",
    "app_streak_days", "content_views_7d", "content_views_30d", "challenges_30d", "active_streak_days",
    "consistency_score", "mood_avg_30d", "mood_slope_per_week", "sync_rate_7d", "sync_rate_30d",
]  # deliberately NO age, gender, company or job_type: behaviour only (privacy + fairness)


def make_risk_dataset(members, metrics_clean, engagement):
    """Features use days 1..76 only. Label = zero app opens in days 77..90."""
    X = compute_features(members, metrics_clean, engagement, as_of_day=FEATURE_DAYS)
    ids = sorted(members["member_id"])
    future = engagement[engagement["date"] > pd.Timestamp("2026-07-01").date() + pd.Timedelta(days=FEATURE_DAYS - 1)]
    opens_future = future.groupby("member_id")["app_opens"].sum().reindex(ids)
    y = (opens_future == 0).astype(int).to_numpy()
    return X, y


def rule_baseline(X: pd.DataFrame) -> np.ndarray:
    """Simple rule: no app opens at all in the last 7 days of the feature window."""
    return (X["app_opens_7d"] == 0).astype(int).to_numpy()


def candidate_models():
    return {
        "logistic_regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                             LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5)),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
                                                              random_state=SEED, n_jobs=-1)),
    }


def cross_validated_scores(model, X, y, n_splits=5):
    """Out-of-fold predictions: every member is scored by a model that never saw them."""
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))
    for tr, va in folds:
        m = model.fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof, folds


def metrics_at(y, pred):
    return {"precision": round(float(precision_score(y, pred, zero_division=0)), 3),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 3)}


def pick_operating_threshold(y, proba, min_recall=0.80):
    """Highest threshold (=fewest false alarms) that still catches >= 80% of the members who disengage."""
    p, r, t = precision_recall_curve(y, proba)
    ok = np.flatnonzero(r[:-1] >= min_recall)
    return float(t[ok[-1]]) if len(ok) else 0.5


def fold_permutation_importance(model_factory, X, y, folds, n_repeats=15):
    imps = []
    for tr, va in folds:
        m = model_factory().fit(X.iloc[tr], y[tr])
        r = permutation_importance(m, X.iloc[va], y[va], scoring="roc_auc", n_repeats=n_repeats, random_state=SEED)
        imps.append(r.importances_mean)
    return pd.Series(np.mean(imps, axis=0), index=X.columns).sort_values(ascending=False)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    members, metrics, engagement = load_raw(get_engine())
    metrics_clean, _ = clean_metrics(metrics)

    # ---------------- 3a personas (features as of day 90) ----------------
    feats90 = compute_features(members, metrics_clean, engagement, as_of_day=N_DAYS)
    personas, profile, scores, k, desc, Xs, labels = fit_personas(feats90)
    scores.to_csv(OUTPUT_DIR / "cluster_selection.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(scores["k"], scores["inertia"], marker="o"); ax[0].set_title("Elbow: inertia vs k"); ax[0].set_xlabel("k")
    ax[1].plot(scores["k"], scores["silhouette"], marker="o"); ax[1].axvline(k, ls="--", c="grey")
    ax[1].set_title(f"Silhouette vs k (chosen k={k})"); ax[1].set_xlabel("k")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "cluster_selection.png", dpi=130); plt.close(fig)
    profile.round(2).to_csv(OUTPUT_DIR / "persona_profiles.csv")
    (OUTPUT_DIR / "personas.json").write_text(json.dumps(
        [{"persona": r["persona"], "members": int(r["members"]), "description": desc[r["persona"]]}
         for _, r in profile.iterrows()], indent=2))

    # ---------------- 3b risk model ----------------
    X76, y = make_risk_dataset(members, metrics_clean, engagement)
    X = X76[RISK_FEATURES]
    base_pred = rule_baseline(X76)
    results = {"n_members": int(len(y)), "positives": int(y.sum()), "positive_rate": round(float(y.mean()), 3)}
    results["baseline_rule_no_opens_last_7d"] = {**metrics_at(y, base_pred),
                                                  "roc_auc": round(float(roc_auc_score(y, base_pred)), 3)}
    oofs = {}
    for name, mdl in candidate_models().items():
        oof, folds = cross_validated_scores(mdl, X, y)
        oofs[name] = oof
        results[name] = {"roc_auc": round(float(roc_auc_score(y, oof)), 3),
                         "pr_auc": round(float(average_precision_score(y, oof)), 3),
                         "at_threshold_0.5": metrics_at(y, (oof >= 0.5).astype(int))}
    best = max(candidate_models(), key=lambda n: results[n]["roc_auc"])
    thr = pick_operating_threshold(y, oofs[best])
    results["chosen_model"] = best
    results["operating_threshold"] = round(thr, 3)
    results["chosen_at_operating_threshold"] = metrics_at(y, (oofs[best] >= thr).astype(int))
    results["note"] = "All numbers are 5-fold cross-validated (each member scored by a model that never saw them)."

    imp = fold_permutation_importance(lambda: candidate_models()[best], X, y, folds)
    results["top_features_permutation_importance_auc_drop"] = {k_: round(float(v), 4) for k_, v in imp.head(8).items()}
    (OUTPUT_DIR / "risk_model_results.json").write_text(json.dumps(results, indent=2))
    imp.to_csv(OUTPUT_DIR / "risk_feature_importance.csv", header=["auc_drop"])
    np.save(OUTPUT_DIR / "risk_oof_proba.npy", oofs[best]); np.save(OUTPUT_DIR / "risk_labels.npy", y)

    # final model on ALL members, then score the day-90 features ("next 14 days")
    final = candidate_models()[best].fit(X, y)
    joblib.dump({"model": final, "features": RISK_FEATURES, "threshold": thr}, MODEL_DIR / "risk_model.joblib")
    risk = final.predict_proba(feats90[RISK_FEATURES])[:, 1]
    band = np.where(risk >= thr, "high", np.where(risk >= thr / 2, "medium", "low"))

    # ---------------- put persona + risk in every snapshot ----------------
    m_idx = members.set_index("member_id")
    pmap = personas.set_index("member_id")["persona"]
    for i, row in feats90.iterrows():
        snap = build_snapshot(row, m_idx.loc[row["member_id"]], persona=pmap[row["member_id"]],
                              risk=float(risk[i]), risk_band=str(band[i]))
        validate_snapshot(snap)
        write_snapshot(snap)
    pd.DataFrame({"member_id": feats90["member_id"], "persona": personas["persona"],
                  "risk": risk, "risk_band": band}).to_sql("member_ml_signals", get_engine(), if_exists="replace", index=False)

    print(f"Personas: k={k}"); print(profile.round(1).to_string())
    print(json.dumps(results, indent=2))
    print("Updated snapshots with persona + risk. Next: python -m src.vi.app (see README)")


if __name__ == "__main__":
    main()
