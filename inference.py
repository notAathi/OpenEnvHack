import os
import uuid
import json
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are an expert executive assistant resolving workplace conflicts and scheduling issues.

For each conflict item, respond with a JSON object (nothing else) with these fields:
- item_id: the id of the conflict item
- conflict_type: one of "scheduling", "deadline", "delegation", "social"
- resolution: one of "reschedule", "decline", "delegate", "accept", "escalate"  (required for medium/hard)
- message: the actual message to send to resolve this conflict (required for hard task only)

Conflict type guide:
- scheduling: overlapping meetings or time blocks
- deadline: time-sensitive deliverable or contract
- delegation: task needs to be assigned or approved by someone
- social: personal/team social obligation conflicting with work

Resolution guide:
- reschedule: move one of the conflicting items to a new time
- decline: politely refuse one obligation
- delegate: assign the task/meeting to someone else
- accept: confirm attendance or approval
- escalate: flag to a higher authority immediately

Example response:
{"item_id": "c1", "conflict_type": "scheduling", "resolution": "reschedule", "message": "Hi team, I have a conflict at that time. Can we reschedule to 4 PM?"}
"""


def call_llm(messages: list) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()


def parse_action(raw: str, task_id: str) -> dict:
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        data = json.loads(raw[start:end])
        action = {
            "item_id": data.get("item_id", ""),
            "conflict_type": data.get("conflict_type"),
        }
        if task_id in ("medium", "hard"):
            action["resolution"] = data.get("resolution")
        if task_id == "hard":
            action["message"] = data.get("message", "")
        return action
    except Exception:
        return None


def run_task(task_id: str):
    session_id = str(uuid.uuid4())

    resp = requests.post(f"{ENV_URL}/reset", json={"session_id": session_id, "task_id": task_id})
    resp.raise_for_status()
    obs = resp.json()

    print(f"[START] task={task_id} env=executive-conflict-resolution model={MODEL_NAME}")

    step_num = 0
    all_rewards = []
    last_error = None
    done = False

    while not done and obs.get("items"):
        item = obs["items"][0]
        instructions = obs["instructions"]
        context = obs.get("context", "")

        user_msg = (
            f"{context}\n\n{instructions}\n\n"
            f"Resolve this conflict:\n"
            f"ID: {item['id']}\n"
            f"Title: {item['title']}\n"
            f"Description: {item['description']}\n"
            f"Participants: {', '.join(item['participants'])}\n"
            f"Urgency: {item['urgency']}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        try:
            raw = call_llm(messages)
            action = parse_action(raw, task_id)
            if action is None:
                action = {"item_id": item["id"], "conflict_type": "scheduling", "resolution": "reschedule", "message": ""}
                last_error = "parse_failed"
            else:
                last_error = None
        except Exception as e:
            action = {"item_id": item["id"], "conflict_type": "scheduling", "resolution": "reschedule", "message": ""}
            last_error = str(e)

        step_resp = requests.post(f"{ENV_URL}/step", json={"session_id": session_id, "action": action})
        step_resp.raise_for_status()
        result = step_resp.json()

        obs = result["observation"]
        reward_val = result["reward"]["value"]
        done = result["done"]
        step_num += 1
        all_rewards.append(reward_val)

        action_str = f"type={action.get('conflict_type')},resolution={action.get('resolution', 'N/A')}"
        error_str = last_error if last_error else "null"
        print(f"[STEP] step={step_num} action={action_str} reward={reward_val:.2f} done={str(done).lower()} error={error_str}")

    score_resp = requests.get(f"{ENV_URL}/score/{session_id}")
    final_score = score_resp.json().get("final_score", 0.0) if score_resp.ok else 0.0

    rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
    success = final_score >= 0.5
    print(f"[END] success={str(success).lower()} steps={step_num} score={final_score:.2f} rewards={rewards_str}")
    return final_score


if __name__ == "__main__":
    scores = {}
    for task in TASKS:
        try:
            score = run_task(task)
            scores[task] = score
        except Exception as e:
            print(f"[END] success=false steps=0 score=0.00 rewards=")
            scores[task] = 0.0

    print("\n=== Baseline Results ===")
    for task, score in scores.items():
        print(f"  {task}: {score:.4f}")
    print(f"  average: {sum(scores.values()) / len(scores):.4f}")
