from environment.models import Action
from environment.tasks.easy_task import GROUND_TRUTH, _clamp


def _reply_score(reply: str, keywords: list) -> float:
    if not keywords:
        return 0.8 if not reply else 0.7
    if not reply:
        return 0.01
    reply_lower = reply.lower()
    hits = sum(1 for kw in keywords if kw in reply_lower)
    return round(max(0.01, min(0.99, hits / len(keywords))), 4)


def grade(action: Action) -> float:
    gt = GROUND_TRUTH.get(action.email_id)
    if gt is None:
        return 0.01

    label_score = 0.99 if action.label == gt["label"] else 0.01
    priority_score = 0.01 if action.priority is None else max(0.01, 0.99 - abs(action.priority - gt["priority"]))
    reply_score = _reply_score(action.reply or "", gt["reply_keywords"])

    return _clamp((label_score + priority_score + reply_score) / 3)
