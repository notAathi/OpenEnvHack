from environment.models import Action


def _clamp(score: float) -> float:
    return round(min(0.99, max(0.01, score)), 4)


def grade(action: Action, gt: dict) -> float:
    if gt is None or action.conflict_type is None:
        return 0.01
    return _clamp(0.99 if action.conflict_type == gt["type"] else 0.01)
