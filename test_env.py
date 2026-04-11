"""
Test script to manually run the traffic environment without an AI agent.

Uses the greedy queue-based heuristic for phase decisions.

Usage:
    python test_env.py
"""

from __future__ import annotations

import logging
import sys
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(message)s")

from server.traffic_environment import TrafficEnvironment
from models import TrafficAction
from tasks import TASKS

BENCHMARK = "traffic-oracle"
TASK_NAMES = ["easy", "medium", "hard"]
SEED = 42


# ---------------------------------------------------------------------------
# Logging helpers (same format as inference.py)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str) -> None:
    print(f"[START] task={task} env={env} model=greedy-heuristic", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Greedy heuristic (no LLM)
# ---------------------------------------------------------------------------

def greedy_phases(obs) -> List[int]:
    """Fast queue-based greedy heuristic."""
    phases = []
    for inter in obs.intersections:
        ns = inter.queue_north + inter.queue_south
        ew = inter.queue_east + inter.queue_west
        ns += (inter.occupancy_north + inter.occupancy_south) * 5
        ew += (inter.occupancy_east + inter.occupancy_west) * 5
        phases.append(0 if ns >= ew else 1)
    return phases


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_task(task: str) -> None:
    """Run a full episode for one task using only the greedy heuristic."""
    config = TASKS[task]
    env = TrafficEnvironment()
    obs = env.reset(task=task, seed=SEED)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    last_error: Optional[str] = None

    log_start(task=task, env=BENCHMARK)

    try:
        for step in range(1, config.episode_duration + 1):
            phases = greedy_phases(obs)

            action = TrafficAction(phases=phases)
            obs = env.step(action)

            reward = obs.reward
            done = obs.done
            rewards.append(reward)
            steps_taken = step
            last_error = None

            # Log every 500 steps to avoid flooding stdout
            if step % 500 == 0 or done:
                action_str = f"set_phases({phases[:6]}{'...' if len(phases) > 6 else ''})"
                log_step(
                    step=step,
                    action=action_str,
                    reward=reward,
                    done=done,
                    error=last_error,
                )

            if done:
                break

        metadata = obs.metadata if obs.metadata else {}
        score = metadata.get("grader_score", 1e-6)
        success = score > 0.0

        print(f"\n--- Task '{task}' Summary ---", flush=True)
        print(f"  Grader score:          {metadata.get('grader_score', 'N/A')}", flush=True)
        print(f"  Total cumulative wait: {metadata.get('total_cumulative_wait', 'N/A')}", flush=True)
        print(f"  Total departed:        {metadata.get('total_departed', 'N/A')}", flush=True)
        print(f"  Total spawned:         {metadata.get('total_spawned', 'N/A')}", flush=True)
        print(f"  Avg reward/step:       {sum(rewards) / len(rewards):.4f}", flush=True)
        print()

    except Exception as exc:
        last_error = str(exc)
        log_step(
            step=steps_taken + 1,
            action="error",
            reward=0.0,
            done=True,
            error=last_error,
        )

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    tasks = sys.argv[1:] if len(sys.argv) > 1 else TASK_NAMES
    for task in tasks:
        if task not in TASKS:
            print(f"Unknown task: {task}. Available: {list(TASKS.keys())}")
            continue
        run_task(task)


if __name__ == "__main__":
    main()
