# VI Insight Lite

An end-to-end prototype of VIWELL's **VI** coaching flow: synthetic member data -> per-member snapshot -> ML signals -> a locally hosted open-source LLM that returns **grounded, safe, structured** coaching insights through a small REST API.

```
 MySQL (3 raw tables)        Python pipeline            ML signals               VI service (FastAPI)
 members / daily_metrics --> clean + features  -->  personas (K-means)  -->  POST /vi/insight
 / engagement                -> JSON snapshot        risk (classifier)         snapshot -> prompt -> Ollama
   Part 1                       Part 2                  Part 3                 -> validate -> retry/fallback   Part 4
```

## Quick start (one command per part)

Prerequisites: Python 3.10+, MySQL 8 (MariaDB also works), [Ollama](https://ollama.com).

```bash
pip install -r requirements.txt            # (optional, to re-run notebooks: pip install -r requirements-notebooks.txt)
mysql -u root -p < sql/00_create_database.sql      # once: creates DB `viwell` and user viwell/viwell

python -m src.part1_generate_data    # Part 1  (~10 s)  fake data -> MySQL
python -m src.part2_pipeline         # Part 2  (~10 s)  clean + features -> data/snapshots/*.json
python -m src.part3_ml               # Part 3  (~1 min) personas + risk -> added to snapshots

ollama pull qwen2.5:3b               # Part 4: download the model once (~2 GB)
uvicorn src.vi.app:app --port 8000   # Part 4: start the API (leave running)
python -m src.vi.evaluate --save-samples    # in a 2nd terminal: 20-member evaluation + sample files
pytest -q                            # unit tests (grounding, escalation, retry/fallback)
```

**Windows / old GPU driver?** If Ollama crashes with a `CUDA error`, force CPU-only inference: `set OLLAMA_NUM_GPU=0` (cmd) or `$env:OLLAMA_NUM_GPU=0` (PowerShell) in the window that starts `uvicorn`. This is how the results below were produced.

Try it by hand:
```bash
curl -X POST localhost:8000/vi/insight -H 'content-type: application/json' -d '{"member_id":"M0042"}'
```
No model available? `VI_LLM_BACKEND=mock uvicorn src.vi.app:app --port 8000` runs the service with an offline test double (a scripted stand-in, **not** an LLM, used only for development and CI; its responses are labelled `"backend": "mock"`). Settings (DB URL, model name) are environment variables, see `.env.example`.

## Repository map
| Path | What |
|---|---|
| `src/part1_generate_data.py` | synthetic data generator (all patterns created here) |
| `src/features.py`, `src/part2_pipeline.py`, `src/snapshot.py` | cleaning, feature engineering, snapshot builder + validation |
| `src/part3_ml.py` | K-means personas, risk classifier, baseline, evaluation |
| `src/vi/` | `guardrails.py` (all safety checks), `prompt.py`, `llm.py` (Ollama + mock), `service.py`, `app.py`, `evaluate.py` |
| `schemas/snapshot.schema.json` | JSON Schema of the snapshot |
| `notebooks/` | 01 EDA + data quality, 02 persona/cluster selection, 03 risk model evaluation (executed, with charts) |
| `samples/` | 5 snapshots, 10 VI responses (2 escalations + 1 near-miss), evaluation table |
| `tests/` | 34 unit tests |
| `docs/SUMMARY.md` | one-page summary |

## Part 1 - Synthetic data (MySQL)
500 members, 8 companies, 90 days (2026-07-01 .. 2026-09-28). Tables exactly as specified (`sql/01_schema.sql`; `yes/no` columns are ENUMs, `mood_checkin` is nullable). Generated with NumPy/pandas and a fixed seed (42) so every run is identical. Faker is not used: nothing in the data is a name or an email, so there is nothing for it to fake.

**Patterns built on purpose** (measured in `notebooks/01_eda_and_data_quality.ipynb`):
1. *Weekday vs weekend* - desk workers walk +12% and sleep +0.8 h on weekends; field workers walk -35%, sleep +0.5 h; shift workers -10% steps, no sleep bump. (Measured: desk 7,828 -> 8,757 steps and 7.04 -> 7.84 h.)
2. *Shift workers* - sleep follows rotating 4-8 day blocks of day shift (~7 h) / night shift (~5.5 h); sleep standard deviation 1.5 h vs 0.7 h for others.
3. *Disengagement* - 75 members (15%) fade: app opens, content, challenges, mood check-ins, steps and syncing decline linearly from a random start day to zero; some are gone before day 77, some still partly active at day 90.
4. *Missing data* - each member skips about 2-14% of days (7.8% overall), sometimes 2 in a row; all four wearable values are NULL on those days.
5. *Outliers* - 30 impossible readings (e.g. 150,000 steps, 23.5 h sleep, 0 or 220 bpm).
6. *Five hidden habit types* (achievers, steady, desk-curious, silent movers, dabblers) - what the clustering should rediscover.
7. *Mood* - 14 members end with repeated low check-ins (must escalate) and 8 "near-miss" members with exactly two low in the last five (must *not* escalate).

## Part 2 - Pipeline and snapshot
`python -m src.part2_pipeline` reads MySQL, cleans, engineers features **as of day 90**, stores `member_features` in MySQL and writes `data/snapshots/Mxxxx.json`.

**Cleaning rules**
1. A day without sync stays *missing* (never 0, never forward-filled). Averages skip missing days; `sync_rate_30d` tells the model how complete the data is.
2. A value outside a physically possible range is *removed* (set missing): steps 0-60,000; active minutes 0-600; sleep 2-14 h; resting HR 35-120 bpm. Removed rather than capped, because a capped value is still an invented value. Result: 30 values removed (`outputs/cleaning_report.json`).

**Features per member** - 7-day and 30-day averages of steps, active minutes, sleep, resting HR and app opens; **change vs own baseline** `chg_pct = (7d - 30d) / 30d x 100` (baseline = the member's own 30-day average); activity streak (consecutive days with >= 30 active minutes; unsynced days are skipped, not counted as breaks); app streak; last active date and days since last open; consistency score 0-100 (steady sleep + steady steps over 30 days); mood average, trend (slope of check-ins, per week) and the last 5 check-ins from the last 14 days.

### Snapshot schema (`schemas/snapshot.schema.json`)
| Field | Meaning |
|---|---|
| `member_id`, `as_of`, `windows_days` | pseudonymous id; last included day; `[7, 30]` |
| `profile.age_band`, `profile.job_type` | the only profile fields (needed for tone/context) |
| `metrics.{steps,active_minutes,sleep_hours,resting_hr}` | each `{7d, 30d, chg_pct}`; `null` if no valid data |
| `engagement` | `app_opens_per_day{7d,30d,chg_pct}`, `days_since_last_open`, `last_active_date`, `app_streak_days`, `content_views_7d`, `challenges_joined_30d` |
| `activity` | `active_streak_days`, `consistency_score` |
| `mood` | `avg_30d`, `trend`, `recent_checkins` (<= 5, oldest first), `low_count_recent` |
| `data_quality.sync_rate_30d` | share of days the wearable synced |
| `ml` | `persona`, `disengagement_risk` (0-1), `risk_band` |

**Privacy by design:** no name, email, date of birth, gender, company or join date - only the pseudonymous `member_id`. The schema has `additionalProperties: false`, so an extra field cannot slip in, and every snapshot is validated when written. **Size:** ~260 estimated tokens (limit 1,500; estimated pessimistically as characters / 3 and asserted in code).

## Part 3 - ML signals
**3a Personas.** K-means on 8 standardised behaviour features (steps, active minutes, sleep, consistency, app opens 30d and 7d, content views, mood). Elbow + silhouette for k = 2..8 (`notebooks/02`). There is no sharp elbow and silhouette is modest (0.24-0.34), so I say so. k = 2-3 score highest but are too coarse to act on; I allowed 4-6 and took the smallest k within 0.02 of the best silhouette -> **k = 4**:
| Persona | Members | One-liner |
|---|---|---|
| Engaged Achievers | 91 | Very active, open the app almost daily and love content - ideal for challenges and streaks. |
| Curious Desk-Sitters | 155 | Like the app and its content but move less than average - good fit for easy, small activity goals. |
| Irregular Rhythms | 81 | Unpredictable sleep and activity (typically shift workers) - offer flexible, routine-building tips. |
| Quiet Drifters | 173 | Rarely open the app and are only moderately active - highest drop-off risk; need a simple hook. |

Names come from rules applied to each cluster's profile (`name_personas`), so they stay stable on re-runs.

**3b Disengagement risk.** Label: zero app opens in days 77-90 (66 of 500 members, 13%). Features: days 1-76 only (no leakage), behaviour only - age, gender, company and job type are deliberately excluded. Evaluated with 5-fold cross-validation (each member scored by a model that never saw them; one split would leave only ~15 positives in the test set).

| Approach | Precision | Recall | ROC-AUC |
|---|---|---|---|
| Rule: no app opens in the last 7 days | 0.71 | 0.82 | 0.88 |
| Logistic regression @ 0.5 | 0.71 | 0.91 | 0.97 |
| Random forest @ 0.5 | 0.85 | 0.79 | 0.98 |
| **Random forest @ chosen threshold 0.49** | **0.86** | **0.80** | 0.98 |

**Which metric matters most for VI: recall, at an acceptable precision.** Missing a member who is about to drift away costs more than one unnecessary friendly nudge; but flagging everyone causes nudge fatigue. So the threshold is the highest one that still keeps recall >= 80%. ROC-AUC is used to compare models, not as the product metric. **Top drivers:** days since last open, content views (7d), change in app opens, app opens (7d), zero-open days in the last 14. **Caveat:** the baseline is strong and results are high because the synthetic disengagement pattern is simple; real data would be noisier.

The final model is trained on all members, applied to the day-90 features, and writes `persona`, `disengagement_risk` and `risk_band` (high >= 0.49, medium >= 0.245) into each snapshot.

## Part 4 - VI coaching layer
`POST /vi/insight {"member_id": "M0042"}` -> `{member_id, insight, evidence[], actions[2], tone, safety_flag, meta}`. (`meta` = source, attempts, backend, model, latency; the Node backend can ignore it.) `GET /health` reports backend and model. Unknown member -> 404; malformed id -> 422.

**Flow** (`src/vi/service.py`): load snapshot -> **escalation check (code)** -> build prompt -> call model -> validate -> on any failure retry **once** with the reasons appended -> otherwise return a **safe fallback**. If the model server is down the service still answers (fallback) instead of crashing.

**Guardrails (all enforced in code, `src/vi/guardrails.py`):**
* **Grounding.** Every number in `insight` and `evidence` must exist in the snapshot. Sign is ignored ("down 18.4%" matches -18.4) and a number may match a snapshot value rounded to its own precision ("18%" matches 18.4). Digits of the member id do not count, nor do dates not in the snapshot. In `actions`, a number is a *suggested goal* ("walk 10 minutes"), not a claim about the member, so it is allowed only when attached to a unit (minutes, hours, days, steps...). Written-out numbers ("three") are *not* checked - a known gap.
* **No medical claims.** A blocklist of diagnosis/condition/medication wording (insomnia, depression, blood pressure, melatonin, "diagnos*", dosage, treatment...). Output containing any is rejected like invalid JSON.
* **Escalation.** If 3 or more of the last 5 check-ins (within 14 days) are <= 2, the service returns a fixed, human-reviewed supportive message with `safety_flag: true` suggesting a qualified professional - **without calling the model at all** and with no coaching tips. A safety rule should not depend on a model's mood. The rule is recomputed from the raw check-in list, not from a precomputed flag.
* **Valid output.** JSON parse (tolerates code fences) -> schema (2 actions, <= 2 sentences, allowed tone, bounded lengths). The service, not the model, sets `member_id` and `safety_flag`.

**Prompt design** (`src/vi/prompt.py`): the *code* selects the 1-3 biggest week-vs-baseline changes and pre-writes them in the exact evidence format; the model only writes the friendly wording around them. That is far more reliable on a 3B model than asking it to find and compute facts. The prompt contains hard rules, the compact snapshot, the output shape and a suggested tone.

**Model:** `qwen2.5:3b` via Ollama (`OLLAMA_MODEL` to change; `llama3.2:3b` also works). Temperature 0.2, `format: json`.

**Evaluation** (`python -m src.vi.evaluate`): 20 members chosen on purpose - 4 escalation cases, 3 near-misses (2 low check-ins, must *not* escalate) and 13 ordinary members spread across personas. The script calls the real HTTP endpoint and re-checks every response independently (schema, grounding, medical wording, escalation rule).

**Results** - real model `qwen2.5:3b` via Ollama, CPU-only laptop (`outputs/eval_results_ollama.md`):

| Metric | Result |
|---|---|
| Valid JSON rate (final response, after retry/fallback) | 100% |
| Grounding pass rate (every number in the snapshot + no medical wording) | 100% |
| Escalation correctness (safety_flag == rule) | 100% |
| Average latency | 13,109 ms (p95 18,528 ms) |
| Model answer accepted on the 1st try (non-escalation cases) | 100% |
| Retries used / fallbacks used | 0 / 0 |
| Escalation template used | 4 (no model call) |

**How to read these results honestly**
* **The guardrails did not trigger in this run** (0 retries, 0 fallbacks), so this run alone does not prove they work. That is what the **34 unit tests** are for: they force invalid JSON, invented numbers, medical wording, model crashes and escalation edge cases and check the retry, fallback and escalation behaviour (`pytest -q`).
* **20 members is a small sample.** The result is "100% on these 20", not a general guarantee. A larger run and human ratings would be the next step.
* **Grounding proves the numbers are real, not that the advice is good.** The check cannot tell whether an insight is insightful or whether the tone is right; a proper evaluation would add human ratings of usefulness and tone.
* **The high first-try pass rate is mostly a design effect**: the code pre-selects the facts and writes the evidence lines, so a small model only has to write friendly wording around them.
* **Latency was measured on a CPU-only laptop** (the GPU driver was too old for Ollama's CUDA build). A GPU, or a self-hosted vLLM server, would be much faster; 13 s per insight is fine for a weekly insight generated in the background, but not for an interactive UI without caching.

Sample outputs: `samples/snapshots/` (5), `samples/vi_insights/` (10, including 2 escalations and 1 near-miss), `samples/eval_results_ollama.md`.

## Assumptions
* Weekend = Saturday/Sunday (UAE federal weekend); KSA's Friday/Saturday would only shift the pattern.
* "Recent check-ins" for escalation = the last 5 check-ins made in the last 14 days (the brief did not define it). Not consecutive: 3 of the last 5.
* Baseline for % change = the member's own 30-day average (which includes the last 7 days; a purer baseline would use days 8-37 - see "next").
* Snapshots are files; in production they would be rows in a store the service can read.
* Token count is estimated (no tokenizer download); the estimate is deliberately pessimistic.

## Key decisions
Escalate before the model; code picks the facts, model writes the words; validate in code and never trust the model's `safety_flag`/`member_id`; remove rather than cap impossible values; no demographics in the risk model; cross-validation and a strong baseline instead of one flattering split; one-shot retry then deterministic fallback so the API always returns a valid, safe answer.

## Privacy and scaling to many client companies
Health data is sensitive and regulated (UAE PDPL / health-data rules, KSA PDPL): the snapshot is pseudonymous and minimal, and the model runs on VIWELL's own infrastructure (Ollama here; in production vLLM or similar on a GPU host in-region). Member data contains no free text, so there is no prompt-injection surface from the data. Logs contain member id, source and latency only - never snapshot values. For many companies: partition snapshots and keys per tenant, never mix tenants in one prompt or cache, enforce authentication between the Node backend and this service (not included), apply retention limits and consent checks upstream, and show company-level aggregates only for groups of 10+ members. Model-hosting production sketch: same FastAPI service, `OllamaClient` swapped for an OpenAI-compatible client pointing at a self-hosted vLLM server; the API contract does not change.

## What I would do with more time
Larger evaluation set with human ratings of usefulness and tone; GPU / vLLM serving for lower latency; real tokenizer for the size check; a purer baseline window; number-word grounding; per-metric evidence-key checks; authentication and rate limiting; snapshot store instead of files; calibrated risk probabilities and a proper time-split backtest; human review of the blocklist and escalation copy by clinicians; Arabic output (`lang: "ar"`); Docker Compose; a larger evaluation set with human ratings of tone and usefulness.

## AI tool usage
I used an AI coding assistant (Claude, by Anthropic, via Claude Code) to generate the first version of the code, the notebooks and this documentation from the task brief. I then installed and ran every part on my own machine (MySQL, Python, Ollama with `qwen2.5:3b`), debugged the setup problems that came up (Python packages, Ollama's GPU/CUDA error, which led to the CPU-only `OLLAMA_NUM_GPU=0` switch in `src/vi/llm.py`), regenerated the sample outputs and evaluation results with the real local model, and checked the results. The design decisions listed above (escalation before the model, code-selected facts, grounding in code, removing rather than capping outliers, no demographics in the risk model) are ones I can explain and defend.

Optional bonus implemented: **tests** (`tests/`, 34 unit tests for the grounding check, escalation rule, output validation, retry and fallback, and the Ollama request format).
