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

SYSTEM_PROMPT = """You are an expert email triage assistant. You will be given emails and must process them.

For each email, respond with a JSON object (and nothing else) with these fields:
- email_id: the id of the email
- label: one of "spam", "urgent", "normal", "newsletter"
- priority: a float 0.0 (low) to 1.0 (high) — required for medium/hard tasks
- reply: a short reply string — required for hard task only, empty string if no reply needed

Example response:
{"email_id": "e1", "label": "spam", "priority": 0.0, "reply": ""}
"""


def call_llm(messages: list) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def parse_action(raw: str, task_id: str) -> dict:
    # Extract JSON from response
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        data = json.loads(raw[start:end])
        action = {"email_id": data.get("email_id", ""), "label": data.get("label")}
        if task_id in ("medium", "hard"):
            action["priority"] = float(data.get("priority", 0.5))
        if task_id == "hard":
            action["reply"] = data.get("reply", "")
        return action
    except Exception:
        return None


def run_task(task_id: str):
    session_id = str(uuid.uuid4())

    # Reset
    resp = requests.post(f"{ENV_URL}/reset", json={"session_id": session_id, "task_id": task_id})
    resp.raise_for_status()
    obs = resp.json()

    print(f"[START] task={task_id} env=email-triage model={MODEL_NAME}")

    step_num = 0
    all_rewards = []
    last_error = None
    done = False

    while not done and obs.get("emails"):
        email = obs["emails"][0]
        instructions = obs["instructions"]

        user_msg = f"{instructions}\n\nProcess this email:\nID: {email['id']}\nFrom: {email['sender']}\nSubject: {email['subject']}\nBody: {email['body']}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        try:
            raw = call_llm(messages)
            action = parse_action(raw, task_id)
            if action is None:
                action = {"email_id": email["id"], "label": "normal", "priority": 0.5, "reply": ""}
                last_error = "parse_failed"
            else:
                last_error = None
        except Exception as e:
            action = {"email_id": email["id"], "label": "normal", "priority": 0.5, "reply": ""}
            last_error = str(e)

        step_resp = requests.post(f"{ENV_URL}/step", json={"session_id": session_id, "action": action})
        step_resp.raise_for_status()
        result = step_resp.json()

        obs = result["observation"]
        reward_val = result["reward"]["value"]
        done = result["done"]
        step_num += 1
        all_rewards.append(reward_val)

        action_str = f"label={action.get('label')},priority={action.get('priority', 'N/A')}"
        error_str = last_error if last_error else "null"
        print(f"[STEP] step={step_num} action={action_str} reward={reward_val:.2f} done={str(done).lower()} error={error_str}")

    # Final score
    score_resp = requests.get(f"{ENV_URL}/score/{session_id}")
    final_score = score_resp.json().get("final_score", 0.0) if score_resp.ok else 0.0

    rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
    success = final_score >= 0.5
    print(f"[END] success={str(success).lower()} steps={step_num} rewards={rewards_str}")
    return final_score


if __name__ == "__main__":
    scores = {}
    for task in TASKS:
        try:
            score = run_task(task)
            scores[task] = score
        except Exception as e:
            print(f"[END] success=false steps=0 rewards=")
            scores[task] = 0.0

    print("\n=== Baseline Results ===")
    for task, score in scores.items():
        print(f"  {task}: {score:.4f}")
    print(f"  average: {sum(scores.values()) / len(scores):.4f}")
