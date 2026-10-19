**Backend:** `ollama`  **Model:** `qwen2.5:3b`  **Members tested:** 20

| Metric | Result |
|---|---|
| Valid JSON rate (final response, after retry/fallback) | 100% |
| Grounding pass rate (every number in the snapshot + no medical wording) | 100% |
| Escalation correctness (safety_flag == rule) | 100% |
| Average latency | 13109 ms (p95 18528 ms) |
| Model answer accepted on the 1st try (non-escalation cases) | 100% |
| Retries used / fallbacks used | 0 / 0 |
| Escalation template used | 4 |
