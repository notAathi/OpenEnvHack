from environment.models import Action
from environment.tasks.easy_task import _clamp
from environment.tasks.medium_task import grade as medium_grade, _ACCEPTABLE


def _message_score(message: str, keywords: list) -> float:
    if not keywords:
        return 0.8
    if not message:
        return 0.01
    msg = message.lower()
    hits = sum(1 for kw in keywords if kw in msg)
    return round(max(0.01, min(0.99, hits / len(keywords))), 4)


def grade(action: Action, gt: dict) -> float:
    if gt is None:
        return 0.01
    type_score = 0.99 if action.conflict_type == gt["type"] else 0.01
    acceptable = _ACCEPTABLE.get(gt["resolution"], {gt["resolution"]})
    res_score = 0.99 if (action.resolution or "") in acceptable else 0.3
    msg_score = _message_score(action.message or "", gt.get("message_keywords", []))
    return _clamp((type_score + res_score + msg_score) / 3)
