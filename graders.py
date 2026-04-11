"""Episodic graders that produce strictly-bounded (0, 1) scores."""

from __future__ import annotations

import logging
from typing import Dict

from tasks import TaskConfig, TASKS

logger = logging.getLogger(__name__)

_SCORE_MIN = 0.1
_SCORE_MAX = 0.99


# Calibrated bounds per task (total cumulative wait in seconds).
# worst_wait: approximate total wait under a random-switching policy.
# best_wait: approximate total wait under a greedy heuristic.
# These are estimated values that will be refined after running calibration.
GRADER_BOUNDS: Dict[str, Dict[str, float]] = {
    "easy": {"worst_wait": 5_000.0, "best_wait": 1_200.0},
    "medium": {"worst_wait": 60_000.0, "best_wait": 25_000.0},
    "hard": {"worst_wait": 300_000.0, "best_wait": 80_000.0},
}


def compute_score(task_name: str, total_cumulative_wait: float) -> float:
    """Compute a score in [0.1, 0.99] for an episode.

    The raw [0, 1] interpolation between worst/best wait bounds is
    linearly mapped into [_SCORE_MIN, _SCORE_MAX] so the output can
    never touch 0.0 or 1.0.

    Args:
        task_name: "easy", "medium", or "hard".
        total_cumulative_wait: Sum of all vehicle wait-seconds over the episode.

    Returns:
        Score in [0.1, 0.99].
    """
    bounds = GRADER_BOUNDS.get(task_name)
    if bounds is None:
        logger.warning("[GRADER] task=%s — unknown task, returning min=%s", task_name, _SCORE_MIN)
        return _SCORE_MIN

    worst = bounds["worst_wait"]
    best = bounds["best_wait"]

    if worst <= best:
        result = _SCORE_MAX if total_cumulative_wait <= best else _SCORE_MIN
        logger.info("[GRADER] task=%s wait=%.2f worst=%.2f best=%.2f (degenerate bounds) score=%s",
                     task_name, total_cumulative_wait, worst, best, result)
        return result

    # raw in [0, 1]: 0 = worst performance, 1 = best performance
    raw = (worst - total_cumulative_wait) / (worst - best)
    raw = max(0.0, min(1.0, raw))

    # Map [0, 1] -> [_SCORE_MIN, _SCORE_MAX]
    final = round(_SCORE_MIN + raw * (_SCORE_MAX - _SCORE_MIN), 2)
    # Safety clamp (guards against float drift after rounding)
    final = max(_SCORE_MIN, min(_SCORE_MAX, final))

    logger.info("[GRADER] task=%s wait=%.2f worst=%.2f best=%.2f raw=%.6f final=%s",
                 task_name, total_cumulative_wait, worst, best, raw, final)
    return final
