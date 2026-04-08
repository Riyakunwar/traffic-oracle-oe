"""Episodic graders that produce strictly-bounded (0, 1) scores."""

from __future__ import annotations

from typing import Dict

from tasks import TaskConfig, TASKS

_SCORE_EPSILON = 1e-6


# Calibrated bounds per task (total cumulative wait in seconds).
# worst_wait: approximate total wait under a random-switching policy.
# best_wait: approximate total wait under a greedy heuristic.
# These are estimated values that will be refined after running calibration.
GRADER_BOUNDS: Dict[str, Dict[str, float]] = {
    "easy": {"worst_wait": 45_000.0, "best_wait": 8_000.0},
    "medium": {"worst_wait": 325_000.0, "best_wait": 105_000.0},
    "hard": {"worst_wait": 1_650_000.0, "best_wait": 390_000.0},
}


def compute_score(task_name: str, total_cumulative_wait: float) -> float:
    """Compute a strictly-bounded (0, 1) score for an episode.

    Score is linearly interpolated between worst and best wait bounds,
    then clamped to the open interval (_SCORE_EPSILON, 1.0 - _SCORE_EPSILON).

    A score of 1.0 means the agent achieved the best known wait time.
    A score of 0.0 means the agent performed as badly as random switching.

    Args:
        task_name: "easy", "medium", or "hard".
        total_cumulative_wait: Sum of all vehicle wait-seconds over the episode.

    Returns:
        Score strictly between 0.0 and 1.0.
    """
    bounds = GRADER_BOUNDS.get(task_name)
    if bounds is None:
        return _SCORE_EPSILON

    worst = bounds["worst_wait"]
    best = bounds["best_wait"]

    if worst <= best:
        return 1.0 - _SCORE_EPSILON if total_cumulative_wait <= best else _SCORE_EPSILON

    score = (worst - total_cumulative_wait) / (worst - best)
    return max(_SCORE_EPSILON, min(1.0 - _SCORE_EPSILON, score))
