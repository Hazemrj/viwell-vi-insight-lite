"""FastAPI microservice.   Run:  uvicorn src.vi.app:app --port 8000

POST /vi/insight   {"member_id": "M0042"}  ->  insight JSON
GET  /health
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .llm import get_client
from .service import MemberNotFound, generate_insight

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = FastAPI(title="VI Insight Lite", version="1.0",
              description="Grounded, guard-railed coaching insights from a member snapshot.")
client = get_client()


class InsightRequest(BaseModel):
    member_id: str = Field(pattern=r"^M\d{4}$", examples=["M0042"])


class InsightResponse(BaseModel):
    member_id: str
    insight: str
    evidence: list[str]
    actions: list[str]
    tone: str
    safety_flag: bool
    meta: dict


@app.get("/health")
def health():
    return {"status": "ok", "backend": client.name, "model": getattr(client, "model", None)}


@app.post("/vi/insight", response_model=InsightResponse)
def insight(req: InsightRequest):
    try:
        return generate_insight(req.member_id, client)
    except MemberNotFound:
        raise HTTPException(status_code=404, detail="member snapshot not found")
