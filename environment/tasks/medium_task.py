from environment.models import Action
from environment.tasks.easy_task import GROUND_TRUTH, _clamp


def grade(action: Action) -> float:
    gt = GROUND_TRUTH.get(action.email_id)
    if gt is None:
        return 0.01

    label_score = 0.99 if action.label == gt["label"] else 0.01

    if action.priority is None:
        priority_score = 0.01
    else:
        priority_score = max(0.01, 0.99 - abs(action.priority - gt["priority"]))

    return _clamp((label_score + priority_score) / 2)
