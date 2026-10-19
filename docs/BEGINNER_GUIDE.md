# Beginner's guide: do the whole task from zero

You do not need to know data science. Follow the steps in order. Anything in a grey box is something you type into a terminal (Windows: "PowerShell"; Mac: "Terminal").

---
## 0. What are we even building? (2 minutes)

Imagine a fitness app. Each person has a **member record** (age group, job), **daily wearable numbers** (steps, sleep...), and **app activity** (opens, mood check-ins). Your company wants a friendly AI coach called **VI** that says things like *"Your sleep dropped this week - try an earlier bedtime."*

Four jobs, one per part:

| Part | Plain-English job | Analogy |
|---|---|---|
| 1 | Invent realistic fake data and put it in a database | Building a practice town to test in |
| 2 | Clean the data and shrink each person to a one-page summary ("snapshot") | Writing a patient-style summary card per person |
| 3 | Let the computer find **groups** of similar people (personas) and **predict who will stop using the app** | Sorting customers into types; spotting who's about to leave |
| 4 | Give the summary to an AI chatbot running **on your own computer**, and make it safe | A coach who may only quote the card, never invent facts |

Key words:
* **Database (MySQL):** an organised spreadsheet-storage program. Tables = sheets.
* **Pipeline:** a script that takes data in, transforms it step by step, and outputs a result - same steps every time.
* **Feature:** a number we calculate to describe a person, e.g. "average sleep last 7 days".
* **Snapshot:** all the useful features for one person in one small JSON file.
* **Machine learning (ML):** letting a program learn patterns from examples instead of writing rules by hand.
* **LLM:** a "large language model" - the chatbot brain. **Ollama** is a free program that runs one on your laptop.
* **API / endpoint:** a way for one program to ask another for something over the network. `POST /vi/insight` = "please give me an insight for this member".
* **Grounding:** every fact in the AI's answer must come from the data. **Guardrails:** code that blocks unsafe/wrong output.

---
## 1. Install the tools (30-45 minutes, one time)

1. **Python 3.10 or newer** - python.org -> Downloads. On Windows tick **"Add python.exe to PATH"**. Check: `python --version`
2. **MySQL** - mysql.com -> MySQL Community Server (installer on Windows includes **MySQL Workbench**, a window to run SQL). Remember the root password you set. Check it runs: `mysql --version`
3. **Ollama** - ollama.com -> Download. Then download a small model (about 2 GB):
   `ollama pull qwen2.5:3b`
   Test: `ollama run qwen2.5:3b "say hi"` (type `/bye` to exit). If your laptop is very slow or has <8 GB RAM, the brief allows a hosted free-tier API; tell me and I'll adapt it.
4. **Git + a GitHub account** - git-scm.com, github.com.
5. A code editor, e.g. **VS Code**.

## 2. Get the project running

1. Unzip the project folder I gave you; open a terminal **inside** it (VS Code: Terminal -> New Terminal).
2. Install the Python libraries:
   `pip install -r requirements.txt`
3. Create the database (once). Mac/Linux/Windows cmd: `mysql -u root -p < sql/00_create_database.sql`. In PowerShell the `<` doesn't work; instead open the file in MySQL Workbench and press the lightning-bolt button. It creates a database `viwell` and a user `viwell` / password `viwell`.

## 3. Run each part and understand what happened

### Part 1 - fake data
`python -m src.part1_generate_data`

**What it does** (file `src/part1_generate_data.py`): invents 500 people. Each gets a "habit type" (e.g. very active & loves the app, or active but never opens the app). For every day it generates steps, sleep etc. using random numbers around that person's habits, then adds realistic quirks the task asked for:
* weekends differ from weekdays; shift workers sleep irregularly;
* 75 people (15%) slowly stop using the app;
* ~8% of days the wearable "didn't sync" (value left empty);
* 30 impossible numbers (like 150,000 steps) planted so we can clean them.
It saves everything to MySQL. A "seed" (42) makes the randomness repeatable, so you and the reviewer get identical data.
**Check:** it prints `members=500 daily_metrics=45000 ...`. In MySQL Workbench: `SELECT * FROM members LIMIT 5;`

### Part 2 - clean + snapshot
`python -m src.part2_pipeline`

**What it does** (`src/features.py`, `src/part2_pipeline.py`):
1. *Missing days stay empty* - we don't pretend 0 steps happened.
2. *Impossible values are deleted* (steps over 60,000, sleep over 14 h, heart rate over 120...).
3. *Features:* average of the last 7 days vs the last 30 days; **% change = (7-day - 30-day) / 30-day x 100** (is this week better or worse than *their own* normal?); streaks; a consistency score; last time they opened the app; mood trend.
4. Writes one small file per person into `data/snapshots/` (open `M0042.json`). It contains **no names or emails** - privacy - and is checked against a schema (`schemas/snapshot.schema.json`) and a size limit.

### Part 3 - machine learning
`python -m src.part3_ml`

**3a Personas (clustering).** *K-means* puts similar people in the same group. We must choose the number of groups. Two charts help: the **elbow** (more groups always fits better; look for where the gain levels off) and the **silhouette** (how clearly separated the groups are, higher = better). There was no clear elbow, so I used a documented rule and got **4 personas**: Engaged Achievers, Curious Desk-Sitters, Irregular Rhythms, Quiet Drifters. Open `notebooks/02_persona_clustering.ipynb` to see the charts.

**3b Risk prediction (classification).** Question: who will open the app **zero times** in days 77-90? We "teach" the model using only days 1-76 (never peeking at the future - peeking is called *leakage* and would make the results fake). We compare against a dumb rule ("no opens in the last 7 days"):
* **Precision:** when we say "at risk", how often are we right?
* **Recall:** of everyone truly at risk, how many did we catch?
* **ROC-AUC:** overall ability to rank risky people above safe ones (0.5 = coin flip, 1 = perfect).
For VI **recall matters most** (missing someone about to leave is worse than one extra friendly nudge), but not at the price of spamming everyone. To test fairly we use **cross-validation**: split people into 5 groups, test each group with a model that never saw them.
The result is saved into each snapshot (`persona`, `disengagement_risk`).

### Part 4 - the AI coach
1. Terminal A: `uvicorn src.vi.app:app --port 8000` (leave it running)
2. Terminal B: `curl -X POST localhost:8000/vi/insight -H "content-type: application/json" -d "{\"member_id\":\"M0042\"}"` (Windows PowerShell: use `curl.exe`; or just open `http://localhost:8000/docs` in the browser and click "Try it out" - easiest)

**What happens inside** (`src/vi/service.py`) - this is the part the reviewers weigh most (25%):
1. Load that person's snapshot.
2. **Safety check #1 (escalation):** if 3 of their last 5 mood check-ins are 2 or lower, **skip the AI** and return a fixed, kind message suggesting they talk to a professional, with `safety_flag: true`.
3. Otherwise build a prompt: rules + the snapshot + 1-3 ready-made "facts" (picked by code) the AI may quote.
4. Send it to the local model (Ollama).
5. **Safety check #2-4 in code** (`src/vi/guardrails.py`): is it valid JSON in the right shape? Does **every number appear in the snapshot** (stops made-up statistics)? Any medical words ("insomnia", "depression", medication)? If anything fails: ask once more; if it fails again, return a safe generic message. The API therefore *always* answers with something valid and safe.

Run the tests: `pytest -q` (34 should pass). **Evaluate** on 20 members and write the sample files:
`python -m src.vi.evaluate --save-samples`
It prints the results table (valid-JSON rate, grounding pass rate, escalation correctness, average latency) and refreshes `samples/` with **real** model output.

> **Important:** the sample outputs I put in the project were made with a fake "mock" model, because my environment couldn't download a real one. Step 3 above overwrites them with real results. Do not skip it, and check `outputs/eval_results_ollama.md` exists. If the real model fails many checks, that is OK - report the numbers honestly and note the retry/fallback caught it.

## 4. Finish the deliverables

1. Open `README.md`: delete the "Status of the LLM results" note, paste your real evaluation table into the Part 4 section, and edit "AI tool usage" so it's honest (you used an AI assistant; say how).
2. `notebooks/` are already run. (Re-run if you change data: `jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb`.)
3. `docs/SUMMARY.md` is your one-page summary. Add your own "what I learned" sentences in your voice.
4. Put it on GitHub:
   * github.com -> New repository -> name `viwell-vi-insight-lite`, **Public**, no README.
   * In the project folder:
     `git init` / `git add .` / `git commit -m "VI Insight Lite"` / `git branch -M main` / `git remote add origin https://github.com/YOUR_USERNAME/viwell-vi-insight-lite.git` / `git push -u origin main`
   * Open the link in a private/incognito window to confirm anyone can see it.
5. **Reply to the email** that carried the brief, before **9:00 AM Dubai time today (Mon 19 Oct)**, with the repo URL and 2-3 lines.

## 5. Be ready to explain it (they said you must)
Practise saying these out loud:
* *Why remove outliers instead of capping?* A capped value is still invented data.
* *Why escalate before calling the model?* Safety shouldn't depend on an AI's behaviour; the rule is simple and testable.
* *How do you stop made-up numbers?* Code extracts every number in the reply and checks it against the snapshot; failures retry once, then fall back.
* *Why is recall the key metric?* Missing a leaving member costs more than an extra nudge; precision is held acceptable to avoid nudge fatigue.
* *Why not feed age/gender to the model?* Fairness and privacy; behaviour is enough.
* *Why did you choose k=4?* No sharp elbow; 2-3 groups too coarse to act on; parsimony within 0.02 silhouette of best.
* *What are the weaknesses?* Synthetic data makes results look better than real life; number-words aren't checked; no auth yet; mock vs real model.
* *How would it scale?* Per-company data separation, self-hosted model server (vLLM) in-region, minimum group size 10 for aggregates, no values in logs.

## 6. If something breaks
| Problem | Fix |
|---|---|
| `Can't connect to MySQL` | MySQL service not running, or wrong password: re-run the `00_create_database.sql` step |
| `ModuleNotFoundError` | run `pip install -r requirements.txt` in the same terminal |
| Insights all have `source: fallback` | Ollama isn't running or the model isn't downloaded: `ollama list`; check `curl localhost:11434` |
| Very slow answers | try `OLLAMA_MODEL=llama3.2:1b` (smaller) or close other apps |
| `uvicorn` not found | `python -m uvicorn src.vi.app:app --port 8000` |
