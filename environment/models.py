from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ConflictItem(BaseModel):
    id: str
    type: str          # "scheduling" | "deadline" | "delegation" | "social"
    title: str
    description: str
    participants: List[str]
    time_window: str
    urgency: str       # "low" | "medium" | "high" | "critical"


class Observation(BaseModel):
    task_id: str
    items: List[ConflictItem]
    context: str       # executive's current schedule summary
    step: int
    instructions: str


class Action(BaseModel):
    item_id: str
    conflict_type: Optional[str] = None    # easy+
    resolution: Optional[str] = None       # medium+: reschedule|decline|delegate|accept|escalate
    message: Optional[str] = None          # hard: actual message to send


class State(BaseModel):
    task_id: str
    step: int
    done: bool
    actions_taken: Dict[str, Any]
