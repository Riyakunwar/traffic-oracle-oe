"""
Inference Script for Traffic Oracle — Traffic Signal Optimization
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script emits exactly three line types to stdout:

    [START] task=<task_name> env=traffic-oracle model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

Usage:
    API_BASE_URL=... MODEL_NAME=... HF_TOKEN=... python inference.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(message)s")

from openai import OpenAI

from server.traffic_environment import TrafficEnvironment
from models import TrafficAction
from tasks import TASKS

# ---------------------------------------------------------------------------
# Configuration (environment variables per hackathon guidelines)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

BENCHMARK = "traffic-oracle"
TASK_NAMES = ["easy", "medium", "hard"]
SEED = 42
EPISODE_STEPS = 7200
LLM_CALL_INTERVAL = 500  # ask LLM every N steps for strategy update

SYSTEM_PROMPT = textwrap.dedent("""
You are a traffic signal controller for a grid road network.
Each intersection has two phases: 0=NS_GREEN (north-south traffic flows)
and 1=EW_GREEN (east-west traffic flows).

Given the current queue lengths and occupancy at each intersection,
decide the optimal phase for each intersection to minimize total vehicle wait time.

Rules:
- If an intersection has more north+south queue pressure, prefer phase 0.
- If an intersection has more east+west queue pressure, prefer phase 1.
- Consider occupancy ratios as tie-breakers.

Respond ONLY with a JSON array of integers (0 or 1), one per intersection.
No explanation, no markdown, just the array. Example: [0,1,0,1,1,0,0,1]
""").strip()


# ---------------------------------------------------------------------------
# Logging helpers (exact required format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM-assisted decisions
# ---------------------------------------------------------------------------

def build_user_prompt(task: str, obs, step: int) -> str:
    """Build a prompt describing current traffic state for the LLM."""
    intersections = obs.intersections
    n = len(intersections)

    lines = []
    for inter in intersections:
        ns_q = inter.queue_north + inter.queue_south
        ew_q = inter.queue_east + inter.queue_west
        ns_occ = inter.occupancy_north + inter.occupancy_south
        ew_occ = inter.occupancy_east + inter.occupancy_west
        lines.append(
            f"node {inter.node_id}: NS_q={ns_q} EW_q={ew_q} "
            f"NS_occ={ns_occ:.1f} EW_occ={ew_occ:.1f}"
        )

    # Cap lines for token efficiency
    state_block = "\n".join(lines[:30])
    if len(lines) > 30:
        state_block += f"\n... ({len(lines) - 30} more)"

    return (
        f"Task: {task}, Step: {step}/{EPISODE_STEPS}, "
        f"Intersections: {n}, "
        f"Active vehicles: {obs.total_vehicles_active}, "
        f"Cumulative wait: {obs.total_cumulative_wait:.0f}s\n\n"
        f"Current state:\n{state_block}\n\n"
        f"Return a JSON array of {n} phase choices (0 or 1):"
    )


def get_llm_phases(client: OpenAI, task: str, obs, step: int) -> Optional[List[int]]:
    """Ask LLM for phase decisions. Returns list of phases or None on failure."""
    user_prompt = build_user_prompt(task, obs, step)
    n = len(obs.intersections)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=1024,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Extract JSON array
        if "[" in text and "]" in text:
            text = text[text.index("["):text.rindex("]") + 1]
        phases = json.loads(text)
        if isinstance(phases, list) and len(phases) == n and all(p in (0, 1) for p in phases):
            return phases
    except Exception:
        pass
    return None


def greedy_phases(obs) -> List[int]:
    """Fast queue-based greedy heuristic — no LLM call needed."""
    phases = []
    for inter in obs.intersections:
        ns = inter.queue_north + inter.queue_south
        ew = inter.queue_east + inter.queue_west
        # Weight occupancy as tie-breaker
        ns += (inter.occupancy_north + inter.occupancy_south) * 5
        ew += (inter.occupancy_east + inter.occupancy_west) * 5
        phases.append(0 if ns >= ew else 1)
    return phases


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_task(task: str, client: OpenAI) -> None:
    """Run a full episode for one task with required logging format."""
    config = TASKS[task]
    env = TrafficEnvironment()
    obs = env.reset(task=task, seed=SEED)

    n = config.num_intersections
    rewards: List[float] = []
    steps_taken = 0
    success = False
    last_error: Optional[str] = None

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, config.episode_duration + 1):
            # Decide phases: LLM periodically, greedy otherwise
            if step % LLM_CALL_INTERVAL == 1:
                llm_result = get_llm_phases(client, task, obs, step)
                if llm_result:
                    phases = llm_result
                else:
                    phases = greedy_phases(obs)
            else:
                phases = greedy_phases(obs)

            action = TrafficAction(phases=phases)
            obs = env.step(action)

            reward = obs.reward
            done = obs.done
            rewards.append(reward)
            steps_taken = step
            last_error = None

            # Format action compactly for log
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

        # Determine success from grader score
        metadata = obs.metadata if obs.metadata else {}
        score = float(metadata.get("grader_score", 0.1))
        # Explicit safety clamp — validator requires strictly (0, 1)
        if score <= 0.0:
            score = 0.01
        elif score >= 1.0:
            score = 0.99
        success = score >= 0.5

    except Exception as exc:
        last_error = str(exc)
        score = 0.01
        log_step(
            step=steps_taken + 1,
            action="error",
            reward=0.0,
            done=True,
            error=last_error,
        )

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    for task in TASK_NAMES:
        run_task(task, client)


if __name__ == "__main__":
    main()
