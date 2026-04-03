# OpenEnv Traffic Lights: RL Environment for Congestion Optimization

This project provides a blueprint for creating an **OpenEnv-compatible reinforcement learning environment** that simulates traffic lights and trains an agent to reduce road congestion.

The goal is to learn a traffic signal policy that improves:
- average vehicle waiting time
- queue lengths per lane
- throughput at intersections

---

## 1) Problem Statement

Traditional traffic light schedules are fixed (e.g., 30s green / 30s red) and do not adapt to changing traffic demand.

In this environment, an RL agent observes intersection traffic state and decides how to control traffic lights to minimize congestion over time.

---

## 2) Environment Design

### Core Concepts

- **Intersection**: one or more signalized intersections (start with one).
- **Approaches/Lanes**: incoming lanes such as North, South, East, West.
- **Signal Phases**: allowed movement groups (e.g., NS green, EW green, all red transition).
- **Vehicles**: entities with arrival time, lane, and movement intent.
- **Time Step**: discrete simulation tick (e.g., 1 second).

### Recommended First Version (MVP)

Start simple:
- Single intersection
- 4 incoming approaches
- 2 signal phases: `NS_GREEN`, `EW_GREEN`
- Stochastic vehicle arrivals (Poisson-like process)
- Fixed yellow/all-red transition penalty period

---

## 3) Observation Space

At each step, the agent should receive a compact state representation such as:

- queue length per incoming lane
- mean waiting time per lane
- current active phase
- time elapsed in current phase
- optional: arrival rate estimate

Example state vector:
```text
[q_n, q_s, q_e, q_w, wait_n, wait_s, wait_e, wait_w, phase_id, phase_time]
```

Keep observations normalized for stable training.

---

## 4) Action Space

Common action choices:

1. **Discrete phase select**
   - `0 = keep current phase`
   - `1 = switch to NS_GREEN`
   - `2 = switch to EW_GREEN`

2. **Binary switch**
   - `0 = hold`
   - `1 = toggle phase`

For safety and realism, enforce:
- minimum green duration
- yellow/all-red transition before switching

---

## 5) Reward Function

A good reward aligns with congestion reduction:

```text
reward_t = - (alpha * total_queue_length + beta * total_waiting_time) + gamma * vehicles_departed
```

Typical behavior:
- penalize long queues and waiting time
- reward throughput (vehicles that clear the intersection)
- optionally penalize frequent phase switching

Example:
```text
reward_t = -0.5 * queue_sum - 0.3 * wait_sum + 1.0 * passed - 0.1 * switched
```

Tune coefficients (`alpha`, `beta`, `gamma`) experimentally.

---

## 6) Episode Definition

An episode can represent a fixed horizon, for example:
- 1 simulated hour
- 3600 steps if 1 step = 1 second

Reset should:
- clear all vehicles
- reset phase and timer
- reseed or regenerate traffic demand pattern

Use multiple demand profiles (light, medium, peak) to avoid overfitting.

---

## 7) OpenEnv Interface (Conceptual)

Your environment should expose standard RL methods:

- `reset(seed=None) -> obs, info`
- `step(action) -> obs, reward, terminated, truncated, info`
- `render()`
- `close()`

`info` may include useful metrics:
- `avg_wait`
- `queue_sum`
- `throughput`
- `switch_count`

---

## 8) Suggested Project Structure

```text
rl-openenv-ai/
├── README.md
├── openenv_traffic/
│   ├── __init__.py
│   ├── env.py                # environment class
│   ├── traffic_model.py      # vehicle arrival + movement logic
│   ├── signal_controller.py  # phase/switch constraints
│   └── metrics.py            # congestion metrics
├── train.py                  # RL training entry point
├── evaluate.py               # policy evaluation script
└── requirements.txt
```

---

## 9) Training Pipeline

1. Build and validate environment dynamics.
2. Run a random policy baseline.
3. Train with a baseline RL algorithm (e.g., PPO, DQN).
4. Compare against fixed-time and actuated heuristic controls.
5. Evaluate on unseen traffic demand profiles.

Track:
- mean reward
- average delay per vehicle
- max and mean queue length
- throughput (vehicles/hour)

---

## 10) Baselines to Compare Against

To prove RL value, compare with:

- **Fixed-time control**: static cycle durations
- **Vehicle-actuated heuristic**: extend green if queue threshold exceeded
- **Random switching**: sanity check lower bound

RL should outperform at least fixed-time under variable demand.

---

## 11) Evaluation Checklist

- [ ] Deterministic seed reproducibility
- [ ] No invalid phase transitions
- [ ] Reward does not explode/vanish
- [ ] Training curve improves over random baseline
- [ ] RL policy generalizes across traffic profiles

---

## 12) Minimal Pseudocode

```python
obs, info = env.reset(seed=42)
done = False
while not done:
    action = agent.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

---

## 13) Future Extensions

- Multi-intersection coordination (grid network)
- Priority lanes (bus/emergency vehicles)
- Pedestrian phase integration
- Weather/event-based demand shifts
- Offline imitation pretraining from heuristic controller logs

---

## 14) Quick Start Notes

If you are implementing this from scratch:
1. Start with one intersection and deterministic arrivals.
2. Verify reward behavior and phase constraints.
3. Add stochastic traffic.
4. Train PPO as a first benchmark.
5. Scale complexity only after stable learning.

This staged approach avoids debugging simulation complexity and RL instability at the same time.
