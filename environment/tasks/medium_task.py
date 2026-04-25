from environment.models import Action
from environment.tasks.easy_task import _clamp

# Some resolutions are acceptable alternatives
_ACCEPTABLE = {
    "reschedule": {"reschedule", "decline"},
    "escalate":   {"escalate", "delegate"},
    "delegate":   {"delegate", "escalate"},
    "decline":    {"decline", "reschedule"},
    "accept":     {"accept"},
}


def grade(action: Action, gt: dict) -> float:
    if gt is None:
        return 0.01
    type_score = 0.99 if action.conflict_type == gt["type"] else 0.01
    if action.resolution is None:
        res_score = 0.01
    else:
        acceptable = _ACCEPTABLE.get(gt["resolution"], {gt["resolution"]})
        res_score = 0.99 if action.resolution in acceptable else 0.3
    return _clamp((type_score + res_score) / 2)
