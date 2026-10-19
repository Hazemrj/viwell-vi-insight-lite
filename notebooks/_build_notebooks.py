"""Builds the 3 notebooks (so they are reproducible text). Run: python notebooks/_build_notebooks.py"""
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).parent
SETUP = '''import sys, json, warnings; sys.path.insert(0, ".."); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from src.db import get_engine
from src.part2_pipeline import load_raw
from src.features import clean_metrics, compute_features
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                     "figure.dpi": 110, "axes.titleweight": "bold", "axes.titlesize": 11})
BLUE, ORANGE, GREY = "#2f6db5", "#d9822b", "#8a8f98"
members, metrics, engagement = load_raw(get_engine())
truth = pd.read_csv("../data/synthetic_ground_truth.csv")'''

def nb(name, cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    nbf.write(n, HERE / name)

nb("01_eda_and_data_quality.ipynb", [
 ("md", "# 01 - Exploring the synthetic data and checking the patterns\n**Goal:** prove that the patterns we promised in Part 1 are really in the data, and see what the cleaning step fixes.\n\nPlain-English reminder: *EDA* (exploratory data analysis) just means 'look at the data with tables and charts before you build anything'."),
 ("code", SETUP),
 ("md", "## 1. Size of the data\n500 members x 90 days = 45,000 rows in each of the two daily tables."),
 ("code", "print(len(members), len(metrics), len(engagement)); members.head()"),
 ("md", "## 2. Pattern: weekday vs weekend\nDesk workers walk more and sleep longer on weekends; field workers walk less; shift workers have no weekend sleep bump."),
 ("code", '''mc, report = clean_metrics(metrics)
d = mc.merge(members[["member_id","job_type"]], on="member_id")
d["weekend"] = pd.to_datetime(d["date"]).dt.dayofweek >= 5
t = d.groupby(["job_type","weekend"])[["steps","sleep_hours"]].mean().round(1).unstack()
display(t)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for a, col, title in zip(ax, ["steps","sleep_hours"], ["Average steps","Average sleep (hours)"]):
    p = d.groupby(["job_type","weekend"])[col].mean().unstack()
    x = np.arange(len(p)); a.bar(x-.18, p[False], .36, color=BLUE, label="weekday"); a.bar(x+.18, p[True], .36, color=ORANGE, label="weekend")
    a.set_xticks(x); a.set_xticklabels(p.index); a.set_title(title); a.legend(frameon=False)
plt.tight_layout()'''),
 ("md", "## 3. Pattern: shift workers sleep irregularly\nThe spread (standard deviation) of sleep is about twice as large for shift workers."),
 ("code", '''print(d.groupby("job_type")["sleep_hours"].std().round(2))
fig, ax = plt.subplots(figsize=(7,3)); 
for j, c in [("desk", BLUE), ("shift", ORANGE)]:
    ax.hist(d.loc[d.job_type==j, "sleep_hours"].dropna(), bins=40, alpha=.6, color=c, label=j, density=True)
ax.set_title("Nightly sleep: shift workers are far more spread out"); ax.set_xlabel("hours"); ax.legend(frameon=False)'''),
 ("md", "## 4. Pattern: ~15% of members gradually disengage\nAverage daily app opens by group (disengaging members vs the rest). The hidden answer key is only used here, to check our synthetic data."),
 ("code", '''e = engagement.merge(truth[["member_id","_disengaged"]], on="member_id")
e["day"] = (pd.to_datetime(e["date"]) - pd.Timestamp("2026-07-01")).dt.days + 1
g = e.groupby(["day","_disengaged"])["app_opens"].mean().unstack().rolling(5, min_periods=1).mean()
fig, ax = plt.subplots(figsize=(8,3.3)); ax.plot(g.index, g[False], color=GREY, label="everyone else"); ax.plot(g.index, g[True], color=ORANGE, label="disengaging members")
ax.axvline(76.5, ls="--", color="k", alpha=.4); ax.text(77, ax.get_ylim()[1]*.9, "risk label window", fontsize=8)
ax.set_title("Average app opens per day (5-day smoothing)"); ax.set_xlabel("day"); ax.legend(frameon=False)
print("disengaging members:", int(truth["_disengaged"].sum()), "of", len(truth))'''),
 ("md", "## 5. Pattern: missing wearable syncs, and outliers to clean\n"),
 ("code", '''print(json.dumps(report, indent=2))
raw = metrics[["steps","sleep_hours","resting_hr","active_minutes"]]
print("\\nRaw extremes (before cleaning):"); display(raw.describe().loc[["min","max"]])
print("After cleaning:"); display(mc[["steps","sleep_hours","resting_hr","active_minutes"]].describe().loc[["min","max"]])
sync = (metrics.assign(miss=metrics.wearable_synced=="no").groupby("member_id")["miss"].mean())
fig, ax = plt.subplots(figsize=(6,3)); ax.hist(sync*100, bins=25, color=BLUE); ax.set_title("Share of days without wearable sync, per member (%)")'''),
 ("md", "## 6. Cleaning rules (also in the README)\n1. A day with no sync stays **missing** - never replaced by 0 (a 0 would look like 'did nothing'). Averages simply skip missing days.\n2. A value outside a physically possible range is **removed** (set to missing): steps 0-60,000; active minutes 0-600; sleep 2-14 h; resting HR 35-120 bpm.\n\nWhy remove instead of cap? A capped number (e.g. 150,000 steps -> 60,000) is still a made-up number."),
])

nb("02_persona_clustering.ipynb", [
 ("md", "# 02 - Personas: choosing the number of clusters\n**K-means** groups members who behave alike. We must tell it how many groups (k). Two tools help:\n* **Elbow** (inertia): how tight the groups are. More groups always means tighter groups, so we look for where the improvement flattens.\n* **Silhouette**: how well separated the groups are (-1 to 1, higher is better)."),
 ("code", SETUP + '''
from src.part3_ml import fit_personas, PERSONA_FEATURES, K_ALLOWED
get_ipython().run_line_magic('matplotlib', 'inline')  # part3_ml uses a no-display backend; show charts here
mc, _ = clean_metrics(metrics)
feats = compute_features(members, mc, engagement, 90)
personas, profile, scores, k, desc, Xs, labels = fit_personas(feats)
display(scores.round(3))'''),
 ("code", '''fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].plot(scores.k, scores.inertia, marker="o", color=BLUE); ax[0].set_title("Elbow: inertia (lower = tighter)"); ax[0].set_xlabel("k")
ax[1].plot(scores.k, scores.silhouette, marker="o", color=BLUE); ax[1].axvline(k, ls="--", color=ORANGE); ax[1].set_title(f"Silhouette (chosen k = {k})"); ax[1].set_xlabel("k")
plt.tight_layout()'''),
 ("md", "### How we chose k (honestly)\n* There is **no sharp elbow** and silhouette scores are modest (about 0.24-0.34). Real people form a continuum, not neat islands. That is normal; we say so rather than pretend.\n* k=2 and k=3 score best, but 2-3 personas are too coarse to act on (e.g. they hide the quiet users).\n* Product rule: allow **k = 4 to 6**, then choose the **smallest k whose silhouette is within 0.02 of the best** (parsimony: fewer personas are easier to use). Result: **k = " + "4**."),
 ("md", "## The personas"),
 ("code", '''display(profile.round(1))
for p in json.load(open("../outputs/personas.json")): print(f"- {p['persona']} ({p['members']} members): {p['description']}")'''),
 ("code", '''from sklearn.decomposition import PCA
pc = PCA(2, random_state=0).fit_transform(Xs)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
pal = [BLUE, ORANGE, "#3a9d78", "#9a5fb4", GREY, "#c0463d"]
for c, name in profile["persona"].items(): ax.scatter(*pc[labels==c].T, s=14, color=pal[c], label=name, alpha=.75)
ax.set_title("Members on a 2-D map (PCA of the 8 behaviour features)"); ax.legend(frameon=False, fontsize=8)'''),
 ("md", "## Sanity check against the hidden archetypes\nThe generator secretly created 5 habit types. The clusters should line up with them reasonably (never perfectly - people vary and the types overlap).\n"),
 ("code", '''x = personas.merge(truth[["member_id","_archetype"]], on="member_id")
display(pd.crosstab(x["persona"], x["_archetype"]))
print("Job-type mix per persona:"); display(pd.crosstab(personas["persona"], members.set_index("member_id").loc[personas.member_id,"job_type"].values, normalize="index").round(2))'''),
])

nb("03_risk_model_evaluation.ipynb", [
 ("md", "# 03 - Disengagement risk model\n**Question:** using only days 1-76, can we predict which members will open the app **zero times** in days 77-90?\n\n**Why split by time?** In real life you predict the future from the past. If we computed features using days 77-90 too, the model would 'cheat' (this is called *data leakage*)."),
 ("code", SETUP + '''
from src.part3_ml import *
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
mc, _ = clean_metrics(metrics)
X76, y = make_risk_dataset(members, mc, engagement)
X = X76[RISK_FEATURES]
print("members:", len(y), "| will disengage (label=1):", y.sum(), f"({y.mean():.1%})")'''),
 ("md", "## Baseline vs models\n**Baseline rule:** 'no app opens in the last 7 days' -> predict disengaged.\n**Models:** logistic regression and random forest.\n**Evaluation:** 5-fold cross-validation - the members are split into 5 groups; each group is scored by a model trained on the other 4. With only ~66 positives, one single train/test split would be too noisy."),
 ("code", '''res = json.load(open("../outputs/risk_model_results.json"))
rows = [("Rule: no opens in last 7 days", res["baseline_rule_no_opens_last_7d"]["precision"], res["baseline_rule_no_opens_last_7d"]["recall"], res["baseline_rule_no_opens_last_7d"]["roc_auc"])]
for m in ["logistic_regression", "random_forest"]:
    rows.append((m + " @0.5", res[m]["at_threshold_0.5"]["precision"], res[m]["at_threshold_0.5"]["recall"], res[m]["roc_auc"]))
c = res["chosen_at_operating_threshold"]; rows.append((f"{res['chosen_model']} @ operating threshold {res['operating_threshold']}", c["precision"], c["recall"], res[res["chosen_model"]]["roc_auc"]))
display(pd.DataFrame(rows, columns=["approach","precision","recall","roc_auc"]))'''),
 ("md", "### Reading the metrics\n* **Precision** = of the members we flag, how many really disengage (low precision = many false alarms).\n* **Recall** = of the members who really disengage, how many we caught (low recall = missed people).\n* **ROC-AUC** = how well the score *ranks* at-risk members above safe ones, regardless of threshold (1.0 perfect, 0.5 coin flip).\n\n**Which matters most for VI? Recall, held at an acceptable precision.** The cost of missing a member who is about to drift away (they lose a wellbeing benefit, the client loses value) is higher than the cost of one unnecessary friendly nudge. But we cannot flag everyone, because nudge fatigue makes people ignore VI. So we pick the threshold that keeps **recall >= 80%** and then look for the best precision. ROC-AUC is used to compare models; it is not the product metric."),
 ("code", '''from sklearn.metrics import roc_curve, precision_recall_curve
oof = np.load("../outputs/risk_oof_proba.npy"); yy = np.load("../outputs/risk_labels.npy")
fpr, tpr, _ = roc_curve(yy, oof); p, r, _ = precision_recall_curve(yy, oof)
base = (X76["app_opens_7d"] == 0).astype(int).to_numpy()
fb, tb, _ = roc_curve(yy, base)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(fpr, tpr, color=BLUE, label="model"); ax[0].plot(fb, tb, color=ORANGE, marker="o", label="baseline rule"); ax[0].plot([0,1],[0,1], color=GREY, ls=":")
ax[0].set_title("ROC curve"); ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("recall"); ax[0].legend(frameon=False)
ax[1].plot(r, p, color=BLUE); ax[1].set_title("Precision vs recall"); ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision"); plt.tight_layout()'''),
 ("md", "## What drives risk?\nPermutation importance: shuffle one feature and see how much the ROC-AUC drops - a big drop means the model relies on it."),
 ("code", '''imp = pd.read_csv("../outputs/risk_feature_importance.csv", index_col=0)["auc_drop"].head(10)[::-1]
fig, ax = plt.subplots(figsize=(7, 3.6)); ax.barh(imp.index, imp.values, color=BLUE); ax.set_title("Top features driving risk (drop in ROC-AUC when shuffled)")
plt.tight_layout()'''),
 ("md", "### Honest reading of the results\n* The model beats the baseline mainly on **ranking** (higher ROC-AUC) and on precision at equal recall - the rule is a strong baseline because app-use drop-off is itself the signal.\n* Top drivers are all **recent app behaviour** (days since last open, content views, change in opens). Wearable and mood features add little here **because the synthetic data was generated that way** - in real data we would expect more signal from them. We deliberately did **not** feed age, gender, company or job type to the model.\n* The data is synthetic and the disengagement pattern is simple, so the high scores will not carry over to real members. The value of this notebook is the *method*: time-based split, baseline comparison, cross-validation, threshold chosen from a product goal."),
])
