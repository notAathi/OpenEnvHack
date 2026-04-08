from environment.models import Action

GROUND_TRUTH = {
    "e1": {"label": "spam",       "priority": 0.05, "reply_keywords": []},
    "e2": {"label": "urgent",     "priority": 0.95, "reply_keywords": ["acknowledge", "investigating", "team"]},
    "e3": {"label": "newsletter", "priority": 0.1,  "reply_keywords": []},
    "e4": {"label": "normal",     "priority": 0.3,  "reply_keywords": ["yes", "no", "tomorrow", "lunch", "available"]},
    "e5": {"label": "urgent",     "priority": 0.9,  "reply_keywords": ["sign", "contract", "confirm", "deadline"]},
}


def _clamp(score: float) -> float:
    return round(min(0.99, max(0.01, score)), 4)


def grade(action: Action) -> float:
    gt = GROUND_TRUTH.get(action.email_id)
    if gt is None:
        return 0.01
    if action.label is None:
        return 0.01
    return _clamp(0.99 if action.label == gt["label"] else 0.01)
