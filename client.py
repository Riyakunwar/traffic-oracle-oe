"""Client for the Traffic Signal OpenEnv environment."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import TrafficAction, TrafficObservation, IntersectionObs


class TrafficEnv(EnvClient[TrafficAction, TrafficObservation, State]):
    """Client for the Traffic Signal Optimization environment.

    Example:
        >>> with TrafficEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset(task="easy")
        ...     for _ in range(100):
        ...         action = TrafficAction(phases=[0] * 16)
        ...         result = client.step(action)
        ...         print(result.observation.total_cumulative_wait)
    """

    def _step_payload(self, action: TrafficAction) -> Dict:
        return {"phases": action.phases}

    def _parse_result(self, payload: Dict) -> StepResult[TrafficObservation]:
        obs_data = payload.get("observation", {})

        intersections = [
            IntersectionObs(**i) for i in obs_data.get("intersections", [])
        ]

        observation = TrafficObservation(
            current_time=obs_data.get("current_time", 0),
            task=obs_data.get("task", "easy"),
            intersections=intersections,
            total_vehicles_active=obs_data.get("total_vehicles_active", 0),
            total_vehicles_departed=obs_data.get("total_vehicles_departed", 0),
            total_vehicles_waiting=obs_data.get("total_vehicles_waiting", 0),
            total_cumulative_wait=obs_data.get("total_cumulative_wait", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
