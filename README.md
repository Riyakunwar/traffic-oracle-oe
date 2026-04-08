# traffic-oracle — Traffic Signal Optimization (OpenEnv)

An **OpenEnv-compatible environment** simulating adaptive traffic signal control on a city road grid. An RL agent controls traffic lights at every intersection to minimize vehicle wait times during peak hours.

Vehicles submit itineraries (origin, destination, departure time, vehicle category). Roads have limited capacity and intersections switch between two signal phases (NS_GREEN / EW_GREEN) with enforced minimum green and yellow transition periods.

---

## Action Space

```python
TrafficAction(phases=[0, 1, 0, ...])  # one int per intersection: 0=NS_GREEN, 1=EW_GREEN
```

One phase choice per intersection (sorted by node ID). Signal changes enforce a 5-second minimum green and a 3-second yellow transition.

## Observation Space

`TrafficObservation` contains:

| Field | Type | Description |
|---|---|---|
| `current_time` | int | Simulation second (0–7199) |
| `task` | str | Active task name |
| `intersections` | list | Per-intersection observations (see below) |
| `total_vehicles_active` | int | Vehicles currently in the network |
| `total_vehicles_departed` | int | Vehicles that reached their destination |
| `total_vehicles_waiting` | int | Vehicles waiting at signals this step |
| `total_cumulative_wait` | float | Sum of all vehicle wait-seconds so far |

Each `IntersectionObs`:

| Field | Description |
|---|---|
| `node_id`, `current_phase`, `phase_timer`, `in_yellow` | Signal state |
| `queue_north/south/east/west` | Vehicles queued per approach |
| `occupancy_north/south/east/west` | Load/capacity ratio per approach |

## Vehicle Categories

| Category | Road space |
|---|---|
| 2-wheeler | 0.5 car-equivalents |
| 3-wheeler | 0.75 |
| Small car | 1.0 |
| Large vehicle | 2.0 |

## Tasks

| | Easy | Medium | Hard |
|---|---|---|---|
| Grid | 4x4 (16 intersections) | 7x7 (49 intersections) | 10x10 (100 intersections) |
| Road capacity | 25 car-equiv | 18 car-equiv | 12 car-equiv |
| Vehicle types | 2-wheeler, small car | All 4 | All 4, heavy on large |
| Traffic volume | ~3,600 vehicles/2h | ~14,400 vehicles/2h | ~43,000 vehicles/2h |
| Demand pattern | Uniform | Moderate | Time-varying + commuter corridors |

Task is selected via `reset(task="easy")`. Episode length: 7,200 steps (2 simulated hours).

## Reward

**Per-step (dense):**
```
reward = -1.0 * wait_penalty + 0.5 * departed_bonus - 0.05 * switch_penalty
```

**Episodic grader (0.0–1.0):** Linearly interpolated between calibrated worst/best wait bounds. Returned in `metadata["grader_score"]` on the final step (`done=True`).

## Baseline Scores (Fixed-Time 30s Cycle, seed=42)

| Task | Grader Score | Total Wait (s) |
|---|---|---|
| Easy | 0.037 | 43,636 |
| Medium | 0.014 | 322,015 |
| Hard | 0.027 | 1,616,050 |

An RL agent should significantly outperform these.

---

## Setup

**Prerequisites:** Python 3.10+, [`uv`](https://github.com/astral-sh/uv)

```bash
git clone <repo-url>
cd rl-openenv-ai

# Install dependencies
uv sync

# Run tests
python -m pytest tests/ -v

# Run the fixed-time baseline (all 3 tasks)
python -m baseline.run_baseline
```

---

## Running the Server Locally

```bash
# Start the FastAPI server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Or run directly
python -m server.app
```

The server exposes:

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start a new episode |
| `/step` | POST | Execute an action |
| `/state` | GET | Current episode metadata |
| `/schema` | GET | Action/observation JSON schemas |
| `/ws` | WebSocket | Persistent session (low-latency) |
| `/health` | GET | Health check |

---

## Docker

```bash
# Build (from project root)
docker build -t traffic-oracle .

# Run
docker run -p 8000:8000 traffic-oracle

# Test it
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" \
     -d '{"task": "easy", "seed": 42}'
```

---

## Running Inference

```bash
# Set required environment variables
export HF_TOKEN="your-huggingface-token"
export API_BASE_URL="https://router.huggingface.co/v1"   # optional, has default
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"            # optional, has default

# Run inference across all tasks
python inference.py
```

---

## Deploy to Hugging Face Spaces

```bash
pip install openenv
huggingface-cli login

# One-command deploy
openenv push --repo-id <your-hf-username>/traffic-oracle
```

---

## Project Structure

```
traffic-oracle/
├── inference.py             # LLM inference script (root, required by guidelines)
├── Dockerfile               # Docker build from project root
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Package metadata + dependencies
├── models.py                # Pydantic Action/Observation types
├── network.py               # Road network graph + grid factory + BFS routing
├── vehicles.py              # VehicleCategory, Itinerary, Vehicle
├── simulation.py            # TrafficSimulator — discrete-time tick engine
├── tasks.py                 # Easy/medium/hard TaskConfig + itinerary generation
├── graders.py               # Episodic 0.0–1.0 scoring
├── client.py                # TrafficEnv(EnvClient) HTTP/WebSocket client
├── server/
│   ├── app.py               # FastAPI app via create_app()
│   └── traffic_environment.py  # Environment subclass (reset/step/state)
├── baseline/
│   ├── fixed_time_agent.py  # Fixed-cycle 30s baseline agent
│   └── run_baseline.py      # Reproducible scoring across all tasks
└── tests/
    ├── test_network.py
    ├── test_vehicles.py
    ├── test_simulation.py
    └── test_environment.py
```
