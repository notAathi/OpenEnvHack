from typing import Any, Dict, Tuple
from environment.models import Action, Observation, State, Email
from environment.tasks import easy_task, medium_task, hard_task

INBOX = [
    {"id": "e1", "subject": "WINNER! Claim your prize now!!!", "sender": "noreply@prize-scam.com",
     "body": "You have won $1,000,000! Click here to claim.", "timestamp": "2024-01-15T08:00:00"},
    {"id": "e2", "subject": "Production server is DOWN", "sender": "alerts@company.com",
     "body": "CRITICAL: The production API server has been unreachable for 10 minutes. Immediate action required.", "timestamp": "2024-01-15T08:05:00"},
    {"id": "e3", "subject": "Monthly newsletter - January 2024", "sender": "news@techdigest.com",
     "body": "This month in tech: AI breakthroughs, new frameworks, and more.", "timestamp": "2024-01-15T07:00:00"},
    {"id": "e4", "subject": "Team lunch tomorrow?", "sender": "colleague@company.com",
     "body": "Hey, are you free for lunch tomorrow at noon? We could go to the new place downtown.", "timestamp": "2024-01-15T09:00:00"},
    {"id": "e5", "subject": "URGENT: Contract renewal deadline TODAY", "sender": "legal@partner.com",
     "body": "The contract expires at 5 PM today. Please sign and return immediately or service will be interrupted.", "timestamp": "2024-01-15T08:30:00"},
]

GRADERS = {"easy": easy_task.grade, "medium": medium_task.grade, "hard": hard_task.grade}

INSTRUCTIONS = {
    "easy":   "Classify each email. Set 'label' to one of: spam, urgent, normal, newsletter.",
    "medium": "Classify each email AND set 'priority' (0.0=low, 1.0=high).",
    "hard":   "Classify, set priority, AND write a short 'reply' where appropriate.",
}


class EmailTriageEnv:
    def __init__(self):
        self.task_id = "easy"
        self._emails = [Email(**e) for e in INBOX]
        self._actions_taken: Dict[str, Action] = {}
        self._step = 0
        self._done = False

    def reset(self, task_level: str = "easy") -> Observation:
        assert task_level in GRADERS, f"Unknown task: {task_level}"
        self.task_id = task_level
        self._actions_taken = {}
        self._step = 0
        self._done = False
        return self._make_obs()

    def state(self) -> State:
        return State(
            task_id=self.task_id,
            step=self._step,
            done=self._done,
            actions_taken={k: v.model_dump() for k, v in self._actions_taken.items()},
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Episode done. Call reset().")
        self._step += 1
        self._actions_taken[action.email_id] = action
        reward = GRADERS[self.task_id](action)
        self._done = len(self._actions_taken) >= len(self._emails)
        return self._make_obs(), reward, self._done, {"step": self._step}

    def final_score(self) -> float:
        if not self._actions_taken:
            return 0.0
        scores = [GRADERS[self.task_id](a) for a in self._actions_taken.values()]
        unanswered = len(self._emails) - len(self._actions_taken)
        penalty = (unanswered / len(self._emails)) * 0.1
        return round(max(0.0, sum(scores) / len(self._emails) - penalty), 4)

    def _make_obs(self) -> Observation:
        pending = [e for e in self._emails if e.id not in self._actions_taken]
        return Observation(task_id=self.task_id, emails=pending, step=self._step, instructions=INSTRUCTIONS[self.task_id])
