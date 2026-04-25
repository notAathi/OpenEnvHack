import random
from typing import Any, Dict, List, Tuple
from environment.models import Action, ConflictItem, Observation, State
from environment.tasks import easy_task, medium_task, hard_task

GRADERS = {"easy": easy_task.grade, "medium": medium_task.grade, "hard": hard_task.grade}

INSTRUCTIONS = {
    "easy":   "Identify the conflict type for each item. Set 'conflict_type' to one of: scheduling, deadline, delegation, social.",
    "medium": "Identify the conflict type AND set 'resolution' to one of: reschedule, decline, delegate, accept, escalate.",
    "hard":   "Identify conflict type, set resolution, AND write the 'message' to send to resolve it.",
}

# Templates: (title, description, type, urgency, resolution, message_keywords, participants_pool)
_TEMPLATES = [
    {
        "type": "scheduling",
        "urgency": "high",
        "resolution": "reschedule",
        "title": "Board meeting overlaps with client demo",
        "description": "Your {time} board meeting was just moved to overlap with the {client} client demo you confirmed last week. Both require your presence.",
        "participants": ["{client} team", "Board members"],
        "message_keywords": ["reschedule", "conflict", "available", "alternative"],
    },
    {
        "type": "deadline",
        "urgency": "critical",
        "resolution": "escalate",
        "title": "Contract signing deadline in 2 hours",
        "description": "Legal flagged that the {partner} partnership contract must be signed by {time} or the deal lapses. You're in back-to-back calls.",
        "participants": ["Legal", "{partner}"],
        "message_keywords": ["urgent", "contract", "sign", "deadline", "priority"],
    },
    {
        "type": "delegation",
        "urgency": "medium",
        "resolution": "delegate",
        "title": "Team lead requests approval for {project} budget",
        "description": "{lead} needs sign-off on a ${amount}K budget increase for {project} before EOD. You're traveling tomorrow.",
        "participants": ["{lead}", "Finance"],
        "message_keywords": ["approve", "delegate", "authority", "proceed", "budget"],
    },
    {
        "type": "social",
        "urgency": "low",
        "resolution": "decline",
        "title": "Dinner invite conflicts with late investor call",
        "description": "{colleague} invited you to a team dinner at {time}, but you have a critical investor call that runs until {end_time}.",
        "participants": ["{colleague}", "Investor team"],
        "message_keywords": ["sorry", "conflict", "rain check", "appreciate", "another time"],
    },
    {
        "type": "scheduling",
        "urgency": "medium",
        "resolution": "reschedule",
        "title": "Two 1:1s booked at the same slot",
        "description": "Both {person_a} and {person_b} have 1:1s scheduled at {time}. Your assistant double-booked.",
        "participants": ["{person_a}", "{person_b}"],
        "message_keywords": ["reschedule", "apologize", "new time", "available", "slot"],
    },
    {
        "type": "deadline",
        "urgency": "high",
        "resolution": "escalate",
        "title": "Quarterly report due while you're on a flight",
        "description": "The Q{quarter} report must be submitted to the board by {time}. Your flight lands 30 minutes after the deadline.",
        "participants": ["Board", "Finance team"],
        "message_keywords": ["delegate", "report", "deadline", "submit", "cover"],
    },
    {
        "type": "delegation",
        "urgency": "high",
        "resolution": "delegate",
        "title": "Client escalation needs immediate owner",
        "description": "{client} is threatening to churn. Their account manager {lead} is on leave. Someone needs to own this call in the next hour.",
        "participants": ["{client}", "{lead}"],
        "message_keywords": ["assign", "escalate", "owner", "handle", "urgent"],
    },
    {
        "type": "social",
        "urgency": "medium",
        "resolution": "accept",
        "title": "CEO wants a quick sync — no agenda",
        "description": "The CEO's EA just pinged asking if you're free at {time} for an informal chat. You have a soft block but nothing confirmed.",
        "participants": ["CEO", "EA"],
        "message_keywords": ["confirm", "available", "happy", "sync", "free"],
    },
]

_NAMES = ["Alex", "Jordan", "Morgan", "Taylor", "Casey", "Riley", "Drew", "Quinn"]
_COMPANIES = ["Acme Corp", "Vertex AI", "NovaTech", "BlueSky", "Meridian"]
_PROJECTS = ["Phoenix", "Atlas", "Orion", "Nexus", "Titan"]
_TIMES = ["9:00 AM", "10:30 AM", "2:00 PM", "3:30 PM", "5:00 PM", "6:00 PM"]
_END_TIMES = ["7:00 PM", "8:00 PM", "9:00 PM"]


def _fill(template: dict) -> dict:
    subs = {
        "time": random.choice(_TIMES),
        "end_time": random.choice(_END_TIMES),
        "client": random.choice(_COMPANIES),
        "partner": random.choice(_COMPANIES),
        "lead": random.choice(_NAMES),
        "colleague": random.choice(_NAMES),
        "person_a": random.choice(_NAMES),
        "person_b": random.choice(_NAMES),
        "project": random.choice(_PROJECTS),
        "amount": random.randint(50, 500),
        "quarter": random.randint(1, 4),
    }
    return {
        "type": template["type"],
        "urgency": template["urgency"],
        "resolution": template["resolution"],
        "title": template["title"].format(**subs),
        "description": template["description"].format(**subs),
        "participants": [p.format(**subs) for p in template["participants"]],
        "message_keywords": template["message_keywords"],
    }


def _generate_inbox(n: int = 5) -> List[dict]:
    chosen = random.sample(_TEMPLATES, min(n, len(_TEMPLATES)))
    items = []
    for i, tmpl in enumerate(chosen):
        filled = _fill(tmpl)
        filled["id"] = f"c{i+1}"
        items.append(filled)
    return items


class ConflictResolutionEnv:
    def __init__(self):
        self.task_id = "easy"
        self._inbox: List[dict] = []
        self._items: List[ConflictItem] = []
        self._actions_taken: Dict[str, Action] = {}
        self._step = 0
        self._done = False

    def reset(self, task_level: str = "easy") -> Observation:
        assert task_level in GRADERS
        self.task_id = task_level
        self._inbox = _generate_inbox(5)
        self._items = [
            ConflictItem(
                id=d["id"], type=d["type"], title=d["title"],
                description=d["description"], participants=d["participants"],
                time_window="today", urgency=d["urgency"],
            )
            for d in self._inbox
        ]
        self._actions_taken = {}
        self._step = 0
        self._done = False
        return self._make_obs()

    def state(self) -> State:
        return State(
            task_id=self.task_id, step=self._step, done=self._done,
            actions_taken={k: v.model_dump() for k, v in self._actions_taken.items()},
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Episode done. Call reset().")
        self._step += 1
        self._actions_taken[action.item_id] = action
        gt = next((d for d in self._inbox if d["id"] == action.item_id), None)
        reward = GRADERS[self.task_id](action, gt)
        self._done = len(self._actions_taken) >= len(self._items)
        return self._make_obs(), reward, self._done, {"step": self._step}

    def final_score(self) -> float:
        if not self._actions_taken:
            return 0.01
        scores = []
        for item_id, action in self._actions_taken.items():
            gt = next((d for d in self._inbox if d["id"] == item_id), None)
            scores.append(GRADERS[self.task_id](action, gt))
        unanswered = len(self._items) - len(self._actions_taken)
        penalty = (unanswered / len(self._items)) * 0.1
        return round(min(0.99, max(0.01, sum(scores) / len(self._items) - penalty)), 4)

    def _make_obs(self) -> Observation:
        pending = [it for it in self._items if it.id not in self._actions_taken]
        context = (
            "Executive context: Back-to-back schedule today. "
            f"{len(self._items)} conflict items require resolution. "
            f"{len(self._actions_taken)} resolved so far."
        )
        return Observation(
            task_id=self.task_id, items=pending,
            context=context, step=self._step,
            instructions=INSTRUCTIONS[self.task_id],
        )
