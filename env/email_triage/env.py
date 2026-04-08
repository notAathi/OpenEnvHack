from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class Email(BaseModel):
    id: str
    subject: str
    sender: str
    body: str
    timestamp: str


class Observation(BaseModel):
    task_id: str
    emails: List[Email]
    step: int
    instructions: str


class Action(BaseModel):
    email_id: str
    label: Optional[str] = None          # spam | urgent | normal | newsletter
    priority: Optional[float] = None     # 0.0 - 1.0
    reply: Optional[str] = None


class Reward(BaseModel):
    value: float
    breakdown: Dict[str, float]
    message: str


INBOX: List[Dict] = [
    {
        "id": "e1",
        "subject": "WINNER! Claim your prize now!!!",
        "sender": "noreply@prize-scam.com",
        "body": "You have won $1,000,000! Click here to claim.",
        "timestamp": "2024-01-15T08:00:00",
        "label": "spam",
        "priority": 0.0,
        "reply_keywords": [],
    },
    {
        "id": "e2",
        "subject": "Production server is DOWN",
        "sender": "alerts@company.com",
        "body": "CRITICAL: The production API server has been unreachable for 10 minutes. Immediate action required.",
        "timestamp": "2024-01-15T08:05:00",
        "label": "urgent",
        "priority": 1.0,
        "reply_keywords": ["acknowledge", "investigating", "team"],
    },
    {
        "id": "e3",
        "subject": "Monthly newsletter - January 2024",
        "sender": "news@techdigest.com",
        "body": "This month in tech: AI breakthroughs, new frameworks, and more.",
        "timestamp": "2024-01-15T07:00:00",
        "label": "newsletter",
        "priority": 0.1,
        "reply_keywords": [],
    },
    {
        "id": "e4",
        "subject": "Team lunch tomorrow?",
        "sender": "colleague@company.com",
        "body": "Hey, are you free for lunch tomorrow at noon? We could go to the new place downtown.",
        "timestamp": "2024-01-15T09:00:00",
        "label": "normal",
        "priority": 0.3,
        "reply_keywords": ["yes", "no", "tomorrow", "lunch", "available"],
    },
    {
        "id": "e5",
        "subject": "URGENT: Contract renewal deadline TODAY",
        "sender": "legal@partner.com",
        "body": "The contract expires at 5 PM today. Please sign and return immediately or service will be interrupted.",
        "timestamp": "2024-01-15T08:30:00",
        "label": "urgent",
        "priority": 0.95,
        "reply_keywords": ["sign", "contract", "confirm", "deadline"],
    },
]


def _label_score(predicted: str, expected: str) -> float:
    return 1.0 if predicted == expected else 0.0


def _priority_score(predicted: float, expected: float) -> float:
    return max(0.0, 1.0 - abs(predicted - expected))


def _reply_score(reply: str, keywords: List[str]) -> float:
    if not keywords:
        return 1.0 if (reply is None or reply == "") else 0.8
    if not reply:
        return 0.0
    reply_lower = reply.lower()
    hits = sum(1 for kw in keywords if kw in reply_lower)
    return hits / len(keywords)


class EmailTriageEnv:
    def __init__(self, task_id: str = "easy"):
        assert task_id in ("easy", "medium", "hard"), f"Unknown task: {task_id}"
        self.task_id = task_id
        self._emails = [Email(**{k: v for k, v in e.items() if k in Email.model_fields}) for e in INBOX]
        self._ground_truth = {e["id"]: e for e in INBOX}
        self._actions_taken: Dict[str, Action] = {}
        self._step = 0
        self._done = False

    def reset(self) -> Observation:
        self._actions_taken = {}
        self._step = 0
        self._done = False
        return self._make_obs()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, Dict]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset().")

        self._step += 1
        self._actions_taken[action.email_id] = action

        reward = self._grade_action(action)
        all_done = len(self._actions_taken) >= len(self._emails)
        if all_done:
            self._done = True

        obs = self._make_obs()
        return obs, reward, self._done, {"step": self._step}

    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "step": self._step,
            "done": self._done,
            "actions_taken": {k: v.model_dump() for k, v in self._actions_taken.items()},
        }

    def _make_obs(self) -> Observation:
        instructions = {
            "easy": "Classify each email. Set 'label' to one of: spam, urgent, normal, newsletter.",
            "medium": "Classify each email AND set 'priority' (0.0=low, 1.0=high).",
            "hard": "Classify, set priority, AND write a short 'reply' where appropriate.",
        }[self.task_id]
        pending = [e for e in self._emails if e.id not in self._actions_taken]
        return Observation(
            task_id=self.task_id,
            emails=pending,
            step=self._step,
            instructions=instructions,
        )

    def _grade_action(self, action: Action) -> Reward:
        gt = self._ground_truth.get(action.email_id)
        if gt is None:
            return Reward(value=0.0, breakdown={}, message="Unknown email id")

        breakdown: Dict[str, float] = {}
        weights: Dict[str, float] = {}

        # Label always graded
        if action.label is not None:
            breakdown["label"] = _label_score(action.label, gt["label"])
            weights["label"] = 1.0

        if self.task_id in ("medium", "hard") and action.priority is not None:
            breakdown["priority"] = _priority_score(action.priority, gt["priority"])
            weights["priority"] = 1.0

        if self.task_id == "hard":
            breakdown["reply"] = _reply_score(action.reply or "", gt["reply_keywords"])
            weights["reply"] = 1.0

        if not breakdown:
            return Reward(value=0.0, breakdown={}, message="No gradable fields in action")

        total_weight = sum(weights.values())
        value = sum(breakdown[k] * weights[k] for k in breakdown) / total_weight

        # Penalize if label missing when it should be provided
        if action.label is None and self.task_id != "":
            value *= 0.5

        return Reward(
            value=round(value, 4),
            breakdown=breakdown,
            message=f"Graded email {action.email_id}",
        )

    def final_score(self) -> float:
        """Aggregate score 0.0-1.0 over all emails."""
        if not self._actions_taken:
            return 0.0
        scores = []
        for email_id, action in self._actions_taken.items():
            r = self._grade_action(action)
            scores.append(r.value)
        # Penalize unanswered emails
        unanswered = len(self._emails) - len(self._actions_taken)
        penalty = unanswered / len(self._emails)
        return round(max(0.0, sum(scores) / len(self._emails) - penalty * 0.1), 4)
