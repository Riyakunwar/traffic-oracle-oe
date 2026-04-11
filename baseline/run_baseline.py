"""Run the fixed-time baseline agent across all tasks and report scores.

Usage:
    python -m baseline.run_baseline
"""

from __future__ import annotations

import time

from server.traffic_environment import TrafficEnvironment
from models import TrafficAction
from tasks import TASKS
from baseline.fixed_time_agent import FixedTimeAgent


def run_task(task_name: str, seed: int = 42) -> dict:
    """Run one full episode with the fixed-time baseline."""
    env = TrafficEnvironment()
    obs = env.reset(task=task_name, seed=seed)

    config = TASKS[task_name]
    agent = FixedTimeAgent(
        num_intersections=config.num_intersections,
        cycle_length=30,
    )

    start_time = time.time()
    total_reward = 0.0

    for step in range(config.episode_duration):
        action = agent.act(obs)
        obs = env.step(action)
        total_reward += obs.reward

        if step % 1000 == 0 and step > 0:
            elapsed = time.time() - start_time
            print(
                f"  [{task_name}] step {step}/{config.episode_duration} "
                f"| wait={obs.total_cumulative_wait:.0f} "
                f"| departed={obs.total_vehicles_departed} "
                f"| active={obs.total_vehicles_active} "
                f"| elapsed={elapsed:.1f}s"
            )

    elapsed = time.time() - start_time
    # Keep fallback strictly inside (0, 1) for strict score validators.
    grader_score = obs.metadata.get("grader_score", 0.1) if obs.metadata else 0.1

    return {
        "task": task_name,
        "grader_score": grader_score,
        "total_reward": total_reward,
        "total_cumulative_wait": obs.total_cumulative_wait,
        "total_departed": obs.total_vehicles_departed,
        "total_spawned": obs.metadata.get("total_spawned", 0) if obs.metadata else 0,
        "elapsed_seconds": elapsed,
    }


def main():
    print("=" * 60)
    print("Traffic Signal Optimization — Fixed-Time Baseline")
    print("=" * 60)

    results = []
    for task_name in ["easy", "medium", "hard"]:
        print(f"\nRunning task: {task_name}")
        print("-" * 40)
        result = run_task(task_name)
        results.append(result)
        print(f"\n  Score: {result['grader_score']:.3f}")
        print(f"  Total wait: {result['total_cumulative_wait']:.0f}s")
        print(f"  Departed: {result['total_departed']}")
        print(f"  Runtime: {result['elapsed_seconds']:.1f}s")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        print(f"  {r['task']:8s} | score={r['grader_score']:.3f} | wait={r['total_cumulative_wait']:.0f}")


if __name__ == "__main__":
    main()
