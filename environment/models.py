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
    label: Optional[str] = None       # spam | urgent | normal | newsletter
    priority: Optional[float] = None  # 0.0 - 1.0
    reply: Optional[str] = None       # hard task only


class State(BaseModel):
    task_id: str
    step: int
    done: bool
    actions_taken: Dict[str, Any]
