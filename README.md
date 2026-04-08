---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - email
  - nlp
app_port: 7860
---

# Email Triage OpenEnv

A real-world email triage environment for RL agent evaluation. Agents must classify, prioritize, and respond to emails — tasks humans perform daily.

## Overview

The environment presents an inbox of 5 emails. The agent must process each email according to the task difficulty. Rewards are given per action based on accuracy of classification, priority scoring, and reply quality.

## Tasks

| Task | Difficulty | Objective |
|------|-----------|-----------|
| `easy` | Easy | Classify each email: `spam`, `urgent`, `normal`, `newsletter` |
| `medium` | Medium | Classify + assign priority score (0.0–1.0) |
| `hard` | Hard | Classify + priority + draft a contextually appropriate reply |

## Action Space

```json
{
  "email_id": "string",
  "label": "spam | urgent | normal | newsletter",
  "priority": 0.0,
  "reply": "string (hard task only)"
}
```

## Observation Space

```json
{
  "task_id": "easy | medium | hard",
  "emails": [{"id", "subject", "sender", "body", "timestamp"}],
  "step": 0,
  "instructions": "string"
}
```

## Reward Function

- Per-step reward after each email action (0.0–1.0)
- Label accuracy: exact match scoring
- Priority accuracy: `1 - |predicted - expected|`
- Reply quality: keyword coverage scoring
- Unanswered emails penalize final score

## Setup

```bash
pip install -r requirements.txt
python server.py          # starts env server on :7860
```

## Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```

## Inference

```bash
export HF_TOKEN=your_token
export ENV_URL=http://localhost:7860
python inference.py
```

## Baseline Scores (Llama-3.1-8B-Instruct)

| Task | Score |
|------|-------|
| easy | ~0.85 |
| medium | ~0.72 |
| hard | ~0.58 |
| **average** | **~0.72** |
