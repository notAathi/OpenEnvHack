from environment.models import Action
from environment.tasks.easy_task import GROUND_TRUTH


def grade(action: Action) -> float:
    gt = GROUND_TRUTH.get(action.email_id)
    if gt is None:
        return 0.0

    label_score = 1.0 if action.label == gt["label"] else 0.0

    if action.priority is None:
        priority_score = 0.0
    else:
        priority_score = max(0.0, 1.0 - abs(action.priority - gt["priority"]))

    return round((label_score + priority_score) / 2, 4)
