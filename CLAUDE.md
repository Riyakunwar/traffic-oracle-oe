# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**traffic-oracle** is an **OpenEnv-compatible traffic signal optimization environment**. An RL agent controls traffic lights across a grid road network to minimize vehicle wait times during peak hours. Built on the OpenEnv framework (HuggingFace/Meta).

## Project Structure

```
traffic-oracle/
├── inference.py             # LLM inference script (root, required by hackathon guidelines)
├── Dockerfile               # Docker build from project root
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Package metadata + dependencies
├── models.py                # TrafficAction, TrafficObservation (Pydantic, extends openenv types)
├── network.py               # RoadNetwork graph, grid factory, BFS shortest paths, signal phases
├── vehicles.py              # VehicleCategory enum, Itinerary, Vehicle dataclasses
├── simulation.py            # TrafficSimulator — tick-by-tick engine (spawn/signal/move/depart)
├── tasks.py                 # TaskConfig for easy/medium/hard, itinerary generation
├── graders.py               # Episodic scoring (0.0–1.0) with calibrated bounds
├── client.py                # TrafficEnv(EnvClient) for connecting to the server
├── server/
│   ├── traffic_environment.py  # Environment subclass (reset/step/state)
│   └── app.py                  # FastAPI via create_app()
├── baseline/                # Fixed-time agent + run script
└── tests/                   # pytest tests for all modules
```

## Key Design Decisions

- **Grid network**: Node ID = `row * cols + col`. Edges are bidirectional between adjacent nodes.
- **Signal phases**: 2 phases per intersection (NS_GREEN=0, EW_GREEN=1). Min green=5s, yellow=3s.
- **Vehicle sizes**: 2-wheeler=0.5, 3-wheeler=0.75, small_car=1.0, large_vehicle=2.0 car-equivalents.
- **Road capacity**: Measured in car-equivalents. Vehicles can't enter a full road.
- **Routing**: BFS shortest path, pre-computed at reset for all OD pairs.
- **Tasks**: easy (2x2), medium (3x3), hard (4x4). Selected via `reset(task="easy")`.
- **Grader**: Linear interpolation between worst/best wait bounds, clamped [0,1].
- **Episode**: 7200 steps = 2 simulated hours.

## Commands

```bash
# Install deps
uv sync

# Run tests
python -m pytest tests/ -v

# Run baseline scoring
python -m baseline.run_baseline

# Start server
uvicorn server.app:app --port 8000

# Docker build & run
docker build -t traffic-oracle .
docker run -p 8000:8000 traffic-oracle

# Run inference
HF_TOKEN=xxx python inference.py
```

## OpenEnv Pattern

The environment follows the standard OpenEnv structure:
- `models.py` — Pydantic types extending `openenv.core.env_server.types.Action` / `Observation`
- `server/traffic_environment.py` — extends `openenv.core.env_server.interfaces.Environment`
- `server/app.py` — `create_app()` from `openenv.core.env_server.http_server`
- `client.py` — extends `openenv.core.EnvClient[Action, Observation, State]`
