"""
Phase 5: Memory - thread history + lightweight recall of past decisions.

- Two different kinds of "memory" that people conflate:
  1. Thread memory: the actual back-and-forth of ONE email conversation (needed so replies aren't context - blind)
  2. Episodic/Decision Memory: a log of what the agent decided across ALL emails, so you (or the agent) can later
  ask "how did I usually handle emails like this?" - this is a RAG-adjacent piece.
- This is intentionally a simple JSON-backed store, not a vector DB. Once you've got this working, swapping in
  Chroma/FAISS for semantic search over 'episodic_log' is a drop-in upgrade - but you should feel the *need* for
  embeddings before reaching for them.
"""

import json
import os
from datetime import datetime, timezone
from typing import List, Optional

STORE_PATH = "memory_store"
EPISODIC_LOG = os.path.join(STORE_PATH, "episodic_log.jsonl")

def _ensure_store():
    os.makedirs(STORE_PATH, exist_ok = True)
    if not os.path.exists(EPISODIC_LOG):
        open(EPISODIC_LOG, "a").close()

def log_decision(thread_id: str, sender: str, subject: str, category: str,
                 reasoning: str, action: str, confidence: str, result: str):
    """Append one processed-email record to the episodic log. This is what makes the agent's history auditable and,
    later, queryable."""

    _ensure_store()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "sender": sender,
        "subject": subject,
        "category": category,
        "action": action,
        "reasoning": reasoning,
        "confidence": confidence,
        "result": result,
    }
    with open(EPISODIC_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

def get_sender_history(sender: str, limit: int = 5) -> List[dict]:
    """Naive Recall: How has the agent handled this sender before? This is the RAG-lite step -- filtering by exact
    sender match. The natural next upgrade is embedding 'subject + reasoning' and doing similarity search instead of
    exact match, so 'emails like this' works even from senders you've never seen."""
    _ensure_store()
    records = []
    with open(EPISODIC_LOG) as f:
        for line in f:
            rec = json.loads(line)
            if rec["sender"] == sender:
                records.append(rec)
    return records[-limit:]

def summarize_history_for_prompt(sender: str) -> Optional[str]:
    """Render sender history as a short context block to inject into the decide_node prompt. Returns None if no 
    history exists for the respective sender (this is a common case)."""
    history = get_sender_history(sender)
    if not history:
        return None
    lines = [f"Past handling of emails from {sender}:"]
    for rec in history:
        lines.append(f"- '{rec['subject']}' -> {rec['action']} ({rec['reasoning'][:80]})")
    return "\n".join(lines)