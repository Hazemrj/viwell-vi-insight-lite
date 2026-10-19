# VI Insight Lite - one-page summary

**What I built.** A working end-to-end prototype of VIWELL's VI flow: (1) a reproducible generator that loads 500 synthetic members x 90 days into MySQL with real-life patterns (weekend effects, irregular shift sleep, 15% gradual disengagement, ~8% missing syncs, 30 planted outliers); (2) a cleaning + feature pipeline producing a privacy-safe JSON snapshot per member (~260 tokens, schema-validated, no identifiers beyond a pseudonymous id); (3) ML signals - 4 product-ready personas (K-means, k chosen with elbow + silhouette) and a disengagement-risk classifier compared with a rule baseline; (4) a FastAPI microservice, `POST /vi/insight`, that prompts a local open-source model (Ollama) and enforces guardrails in code.

**Results.** Risk model (5-fold CV): ROC-AUC 0.98 vs 0.88 for the "no opens in 7 days" rule; at the chosen threshold precision 0.86 / recall 0.80 vs 0.71 / 0.82 for the rule. Recall is the metric that matters most for VI (missing a member about to drift costs more than one extra nudge), held at an acceptable precision. 34 unit tests cover grounding, the escalation rule, retry and fallback. The LLM evaluation script reports valid-JSON rate, grounding pass rate, escalation correctness and latency on 20 members (see `outputs/eval_results_*.md`).

**What I learned.**
1. *Make the code do what code is good at.* Choosing which facts to cite, checking numbers, and deciding to escalate are all deterministic; the model only writes friendly wording. Small models become far more reliable this way.
2. *A safety rule must not depend on the model.* Escalation runs before the LLM and returns a vetted message.
3. *Honest evaluation beats a high score.* The baseline is strong, the clusters overlap (silhouette 0.24-0.34, no sharp elbow), and the data is synthetic - I report that instead of hiding it.
4. *Privacy is a design input, not a final step*: minimum fields, no demographics in the model, no free text in snapshots, no values in logs.

**Limits / next.** Sample LLM outputs must be regenerated with the real model on the target laptop; number-words are not grounding-checked; baseline window includes the last 7 days; no auth or snapshot store yet; Arabic and Docker Compose not done.
