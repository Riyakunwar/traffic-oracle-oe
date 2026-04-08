"""OpenEnv Environment implementation for traffic signal optimization."""

from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import TrafficAction, TrafficObservation, IntersectionObs
from network import RoadNetwork
from simulation import TrafficSimulator
from tasks import TASKS, TaskConfig, generate_itineraries
from graders import compute_score


class TrafficEnvironment(Environment):
    """Traffic signal optimization environment.

    The agent controls traffic light phases at all intersections in a grid
    road network to minimize total vehicle wait time.

    Tasks:
        - "easy": 4x4 grid, light traffic, 2 vehicle types
        - "medium": 7x7 grid, moderate traffic, 4 vehicle types
        - "hard": 10x10 grid, heavy peak-hour traffic with commuter corridors
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.task_config: TaskConfig = TASKS["easy"]
        self.simulator: TrafficSimulator | None = None
        self._prev_phases: list[int] = []

    def reset(self, **kwargs) -> TrafficObservation:
        """Reset the environment and start a new episode.

        Keyword Args:
            task: Task difficulty ("easy", "medium", "hard"). Default: "easy".
            seed: Random seed for itinerary generation. Default: task's default seed.
        """
        task_name = kwargs.get("task", "easy")
        seed = kwargs.get("seed", None)

        self.task_config = TASKS.get(task_name, TASKS["easy"])
        effective_seed = seed if seed is not None else self.task_config.seed

        # Build the road network
        network = RoadNetwork.grid(
            rows=self.task_config.grid_rows,
            cols=self.task_config.grid_cols,
            road_capacity=self.task_config.road_capacity,
            road_length=self.task_config.road_length,
            min_green=self.task_config.min_green,
            yellow_duration=self.task_config.yellow_duration,
        )

        # Generate vehicle itineraries
        itineraries = generate_itineraries(self.task_config, effective_seed)

        # Create simulator
        self.simulator = TrafficSimulator(network, itineraries)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._prev_phases = [0] * self.task_config.num_intersections

        return self._build_observation(reward=0.0, done=False)

    def step(self, action: TrafficAction) -> TrafficObservation:  # type: ignore[override]
        """Execute one simulation step.

        Args:
            action: TrafficAction with phase choices for all intersections.

        Returns:
            TrafficObservation with updated state, reward, and done flag.
        """
        if self.simulator is None:
            return self.reset()

        # Validate action length
        num_intersections = self.task_config.num_intersections
        phases = action.phases
        if len(phases) != num_intersections:
            # Pad or truncate to match
            if len(phases) < num_intersections:
                phases = phases + [0] * (num_intersections - len(phases))
            else:
                phases = phases[:num_intersections]

        # Run simulation tick
        metrics = self.simulator.tick(phases)
        self._state.step_count += 1

        # Compute per-step reward (dense signal)
        reward = self._compute_reward(metrics, phases)
        self._prev_phases = list(phases)

        # Check episode termination
        done = self._state.step_count >= self.task_config.episode_duration

        obs = self._build_observation(reward=reward, done=done)

        # On final step, include grader score in metadata
        if done:
            grader_score = compute_score(
                self.task_config.name,
                self.simulator.total_cumulative_wait,
            )
            obs.metadata = {
                "grader_score": grader_score,
                "total_cumulative_wait": self.simulator.total_cumulative_wait,
                "total_departed": self.simulator.total_departed,
                "total_spawned": self.simulator.total_spawned,
            }

        return obs

    @property
    def state(self) -> State:
        return self._state

    def _compute_reward(self, metrics, phases: list[int]) -> float:
        """Compute dense per-step reward."""
        num_intersections = self.task_config.num_intersections

        # Normalize wait penalty
        max_wait_per_step = max(1, self.simulator.total_spawned)
        wait_penalty = metrics.wait_increment / max(1.0, max_wait_per_step * 0.1)

        # Throughput bonus
        departed_bonus = metrics.vehicles_departed

        # Switching penalty
        switches = sum(
            1 for i, p in enumerate(phases)
            if i < len(self._prev_phases) and p != self._prev_phases[i]
        )
        switch_penalty = switches / max(1, num_intersections)

        reward = -1.0 * wait_penalty + 0.5 * departed_bonus - 0.05 * switch_penalty
        return reward

    def _build_observation(self, reward: float, done: bool) -> TrafficObservation:
        """Build a TrafficObservation from current simulator state."""
        if self.simulator is None:
            return TrafficObservation(
                done=done,
                reward=reward,
                task=self.task_config.name,
            )

        queues = self.simulator.get_queue_lengths()
        occupancies = self.simulator.get_edge_occupancies()

        intersection_obs = []
        for node_id in sorted(self.simulator.network.intersections.keys()):
            inter = self.simulator.network.intersections[node_id]
            q = queues.get(node_id, {"north": 0, "south": 0, "east": 0, "west": 0})
            o = occupancies.get(node_id, {"north": 0.0, "south": 0.0, "east": 0.0, "west": 0.0})

            intersection_obs.append(
                IntersectionObs(
                    node_id=node_id,
                    current_phase=int(inter.current_phase),
                    phase_timer=inter.phase_timer,
                    in_yellow=inter.in_yellow,
                    queue_north=q["north"],
                    queue_south=q["south"],
                    queue_east=q["east"],
                    queue_west=q["west"],
                    occupancy_north=round(o["north"], 3),
                    occupancy_south=round(o["south"], 3),
                    occupancy_east=round(o["east"], 3),
                    occupancy_west=round(o["west"], 3),
                )
            )

        # Count currently waiting vehicles
        total_waiting = sum(
            sum(q.values()) for q in queues.values()
        )

        return TrafficObservation(
            current_time=self.simulator.current_time,
            task=self.task_config.name,
            intersections=intersection_obs,
            total_vehicles_active=len(self.simulator.vehicles),
            total_vehicles_departed=self.simulator.total_departed,
            total_vehicles_waiting=total_waiting,
            total_cumulative_wait=float(self.simulator.total_cumulative_wait),
            done=done,
            reward=reward,
        )
