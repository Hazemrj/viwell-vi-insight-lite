"""LLM clients. Two backends with the same tiny interface:  client.chat(messages) -> str

* OllamaClient - the REAL thing: a locally hosted open-source model served by Ollama.
* MockClient   - an offline test double that writes a plausible answer from the prompt.
                 It is NOT a language model. It exists so the guardrails, API and evaluation
                 can be tested on a machine without a model (and in CI). Responses carry
                 meta.backend = "mock" so it can never be mistaken for the real model.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

import requests

from ..config import LLM_BACKEND, OLLAMA_MODEL, OLLAMA_URL


class OllamaClient:
    name = "ollama"

    def __init__(self, url: str = OLLAMA_URL, model: str = OLLAMA_MODEL, timeout: int = 120):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout

    def chat(self, messages: list[dict]) -> str:
        options = {"temperature": 0.2, "num_predict": 400}
        # Set OLLAMA_NUM_GPU=0 to force CPU-only (works around broken GPU drivers).
        if os.getenv("OLLAMA_NUM_GPU") not in (None, ""):
            options["num_gpu"] = int(os.environ["OLLAMA_NUM_GPU"])
        r = requests.post(
            f"{self.url}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False, "format": "json",
                  "options": options},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]


class MockClient:
    """Deterministic fake model. FAULT_RATE injects realistic mistakes so the guardrails can be exercised."""
    name = "mock"
    model = "mock-template-writer"
    FAULTS = ["bad_json", "invented_number", "medical_claim", "too_long"]

    def __init__(self, fault_rate: float | None = None):
        self.fault_rate = float(os.getenv("MOCK_FAULT_RATE", "0.2")) if fault_rate is None else fault_rate
        self.calls = 0

    def chat(self, messages: list[dict]) -> str:
        self.calls += 1
        user = messages[-1]["content"]
        snap = json.loads(re.search(r"SNAPSHOT:\n(\{.*\})", user).group(1))
        facts = re.findall(r"^- (\w+_7d: .*)$", user, flags=re.M)
        tone = re.search(r"Suggested tone: (\w+)", user).group(1)
        retry = "previous answer was rejected" in user
        h = int(hashlib.md5(f"{snap['member_id']}{retry}".encode()).hexdigest(), 16)
        fault = None
        if not retry and (h % 1000) / 1000 < self.fault_rate:      # faults only on the first attempt
            fault = self.FAULTS[h % len(self.FAULTS)]
        out = self._write(snap, facts, tone)
        if fault == "bad_json":
            return "Sure! Here is the insight: " + json.dumps(out)[:-1]
        if fault == "invented_number":
            out["insight"] = "Your sleep dropped by 37% this week, so let's protect your evenings."
        if fault == "medical_claim":
            out["insight"] = "Your pattern may point to insomnia, so consider a sleep medication."
        if fault == "too_long":
            out["insight"] = "One. Two. Three sentences are too many here."
        return json.dumps(out)

    @staticmethod
    def _write(snap: dict, facts: list[str], tone: str) -> dict:
        if facts:
            metric = facts[0].split("_7d")[0].replace("_", " ")
            worse = "-" in facts[0].split("(")[-1] if metric != "resting hr" else "+" in facts[0].split("(")[-1]
            insight = (f"Your {metric} was a little below your usual level this week - a small reset can help."
                       if worse else f"Nice work - your {metric} is ahead of your usual level this week.")
        else:
            insight = "Your week looks steady and close to your usual level - nicely consistent."
        return {
            "member_id": snap["member_id"], "insight": insight,
            "evidence": facts[:2] or [f"app_opens_per_day_30d: {snap['engagement']['app_opens_per_day']['30d']}"],
            "actions": ["Take a 10 minute walk after one meal on three days.",
                        "Pick a regular bedtime and keep it for the next few nights."],
            "tone": tone, "safety_flag": False,
        }


def get_client(backend: str | None = None):
    backend = backend or LLM_BACKEND
    if backend == "mock":
        return MockClient()
    if backend == "ollama":
        return OllamaClient()
    raise ValueError(f"unknown VI_LLM_BACKEND: {backend}")
