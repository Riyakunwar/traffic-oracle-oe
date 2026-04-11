# Running the Traffic Oracle Environment

## Prerequisites

```bash
uv sync
```

## Task Overview

| Task   | Grid | Intersections | Arrival Rate | Vehicle Types | Episode Steps |
|--------|------|---------------|--------------|---------------|---------------|
| easy   | 2x2  | 4             | 0.5/sec      | 2 (2-wheeler, small car) | 7200 |
| medium | 3x3  | 9             | 2.0/sec      | 4 (all types) | 7200 |
| hard   | 4x4  | 16            | 6.0/sec      | 4 (all types, time-varying, corridors) | 7200 |

Each intersection has 2 signal phases:
- `0` = NS_GREEN (north-south traffic flows)
- `1` = EW_GREEN (east-west traffic flows)

## 1. Run the Test Script (No AI Agent)

Uses a greedy queue-based heuristic. No API keys needed.

```bash
# All tasks
uv run python test_env.py

# Single task
uv run python test_env.py easy
uv run python test_env.py medium
uv run python test_env.py hard
```

## 2. Run the Inference Script (With LLM Agent)

```bash
HF_TOKEN=your_token_here uv run python inference.py
```

Optional overrides:

```bash
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
HF_TOKEN=your_token_here \
uv run python inference.py
```

## 3. Run via the Server (HTTP API)

Start the server:

```bash
uv run uvicorn server.app:app --port 8000
```

### Reset (start an episode)

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy", "seed": 42}'
```

### Step (send an action)

**Easy (2x2, 4 intersections):**

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"phases": [0, 1, 0, 1]}}'
```

**Medium (3x3, 9 intersections):**

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"phases": [0, 1, 0, 1, 0, 1, 0, 1, 0]}}'
```

**Hard (4x4, 16 intersections):**

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"phases": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}}'
```

### Get state

```bash
curl http://localhost:8000/state
```

### Get schema

```bash
curl http://localhost:8000/schema
```

### Health check

```bash
curl http://localhost:8000/health
```

## 4. Run via Docker

```bash
docker build -t traffic-oracle .
docker run -p 8000:8000 traffic-oracle
```

Then use the HTTP API calls above.

## 5. Run Programmatically (Python)

```python
from server.traffic_environment import TrafficEnvironment
from models import TrafficAction

env = TrafficEnvironment()

# Easy task: 2x2 grid, 4 intersections
obs = env.reset(task="easy", seed=42)
print(f"Intersections: {len(obs.intersections)}")

for step in range(1, 7201):
    # All NS_GREEN
    action = TrafficAction(phases=[0, 0, 0, 0])
    obs = env.step(action)

    if obs.done:
        print(f"Episode done at step {step}")
        print(f"Score: {obs.reward}")
        print(f"Metadata: {obs.metadata}")
        break
```

## Scoring

The grader returns a score strictly in (0, 1) based on total cumulative wait time:
- Score near **0.9**: agent performs close to the best known heuristic
- Score near **0.1**: agent performs close to random signal switching
- The final step's `obs.reward` is set to the grader score
- The `obs.metadata` dict contains `grader_score`, `total_cumulative_wait`, `total_departed`, `total_spawned`
