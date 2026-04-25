---
title: Executive Conflict Resolution OpenEnv
emoji: ⚡
colorFrom: indigo
colorTo: orange
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - conflict-resolution
  - personalized-tasks
app_port: 7860
---

# Executive Conflict Resolution OpenEnv

A dynamic real-world environment where an LLM agent acts as an executive assistant resolving workplace conflicts — scheduling clashes, deadline crises, delegation decisions, and social obligation conflicts.

**Theme: 3.2 Personalized Tasks** (also touches Theme 2: Long-Horizon Planning)

## Why This Is Hard

Unlike static email classification, every episode generates a **fresh inbox** with randomized participants, times, and stakes. The agent must:
- Understand cross-item dependencies (can't reschedule meeting A without knowing about meeting B)
- Choose the *right* resolution strategy, not just label the problem
- Draft an actual actionable message — not a template

## Tasks

| Task | Difficulty | Objective |
|------|-----------|-----------|
| `easy` | Easy | Classify each conflict: `scheduling`, `deadline`, `delegation`, `social` |
| `medium` | Medium | Classify + choose resolution: `reschedule`, `decline`, `delegate`, `accept`, `escalate` |
| `hard` | Hard | Classify + resolve + draft the actual message to send |

## Action Space

```json
{
  "item_id": "string",
  "conflict_type": "scheduling | deadline | delegation | social",
  "resolution": "reschedule | decline | delegate | accept | escalate",
  "message": "string (hard task only)"
}
```

## Observation Space

```json
{
  "task_id": "easy | medium | hard",
  "items": [{"id", "type", "title", "description", "participants", "time_window", "urgency"}],
  "context": "executive schedule summary string",
  "step": 0,
  "instructions": "string"
}
```

## Reward Function

- Per-step reward after each conflict resolution (0.0–1.0)
- Type accuracy: exact match (0.99 correct / 0.01 wrong)
- Resolution accuracy: exact + acceptable alternatives get full credit
- Message quality: keyword coverage scoring against expected resolution keywords
- Unanswered items penalize final score

## Dynamic Inbox Generation

Each `reset()` call samples 5 conflict templates from a pool of 8, fills in randomized names, companies, times, and amounts. No two episodes are identical.

## Setup

```bash
pip install -r requirements.txt
python server.py
```

## Docker

```bash
docker build -t conflict-resolution-env .
docker run -p 7860:7860 conflict-resolution-env
```

## Inference

```bash
export HF_TOKEN=your_token
export ENV_URL=http://localhost:7860
python inference.py
```

## Training (Round 2 — Onsite)

Training notebook: `training/train_trl.ipynb` *(to be completed onsite with HF compute credits)*

Expected reward improvement after fine-tuning:

| Task | Before | After (expected) |
|------|--------|-----------------|
| easy | ~0.80 | ~0.95 |
| medium | ~0.65 | ~0.85 |
| hard | ~0.45 | ~0.70 |

## Blog / Video

*Link to be added after onsite training run.*
