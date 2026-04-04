"""Task configurations and itinerary generation for easy/medium/hard difficulties."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .vehicles import Itinerary, VehicleCategory


@dataclass
class TaskConfig:
    """Configuration for a single task difficulty level."""

    name: str
    grid_rows: int
    grid_cols: int
    road_capacity: float
    road_length: int  # free-flow traversal time in seconds
    episode_duration: int  # total simulation steps
    min_green: int
    yellow_duration: int
    vehicle_categories: List[VehicleCategory]
    category_weights: List[float]
    arrival_rate_per_second: float  # Poisson lambda
    seed: int = 42
    # For hard task: commuter corridors as (origin, dest) pairs with boosted probability
    corridors: List[Tuple[int, int]] = field(default_factory=list)
    corridor_boost: float = 1.0  # multiplier for corridor OD pairs
    time_varying: bool = False  # whether arrival rate varies over time

    @property
    def num_intersections(self) -> int:
        return self.grid_rows * self.grid_cols


TASKS: Dict[str, TaskConfig] = {
    "easy": TaskConfig(
        name="easy",
        grid_rows=4,
        grid_cols=4,
        road_capacity=25.0,
        road_length=12,
        episode_duration=7200,
        min_green=5,
        yellow_duration=3,
        vehicle_categories=[VehicleCategory.TWO_WHEELER, VehicleCategory.SMALL_CAR],
        category_weights=[0.3, 0.7],
        arrival_rate_per_second=0.5,
    ),
    "medium": TaskConfig(
        name="medium",
        grid_rows=7,
        grid_cols=7,
        road_capacity=18.0,
        road_length=8,
        episode_duration=7200,
        min_green=5,
        yellow_duration=3,
        vehicle_categories=[
            VehicleCategory.TWO_WHEELER,
            VehicleCategory.THREE_WHEELER,
            VehicleCategory.SMALL_CAR,
            VehicleCategory.LARGE_VEHICLE,
        ],
        category_weights=[0.2, 0.15, 0.45, 0.2],
        arrival_rate_per_second=2.0,
    ),
    "hard": TaskConfig(
        name="hard",
        grid_rows=10,
        grid_cols=10,
        road_capacity=12.0,
        road_length=6,
        episode_duration=7200,
        min_green=5,
        yellow_duration=3,
        vehicle_categories=[
            VehicleCategory.TWO_WHEELER,
            VehicleCategory.THREE_WHEELER,
            VehicleCategory.SMALL_CAR,
            VehicleCategory.LARGE_VEHICLE,
        ],
        category_weights=[0.15, 0.1, 0.4, 0.35],
        arrival_rate_per_second=6.0,
        time_varying=True,
        corridor_boost=5.0,
    ),
}


def _get_hard_corridors(rows: int, cols: int) -> List[Tuple[int, int]]:
    """Generate commuter corridors for the hard task.

    Creates corridors from edges to opposite edges (simulating commuter flows).
    """
    corridors = []
    # Left-to-right corridors (top and bottom rows)
    for r in [0, rows - 1]:
        origin = r * cols + 0
        dest = r * cols + (cols - 1)
        corridors.append((origin, dest))
        corridors.append((dest, origin))
    # Top-to-bottom corridors (left and right columns)
    for c in [0, cols - 1]:
        origin = 0 * cols + c
        dest = (rows - 1) * cols + c
        corridors.append((origin, dest))
        corridors.append((dest, origin))
    return corridors


def _arrival_rate_profile(t: int, base_rate: float, episode_duration: int) -> float:
    """Time-varying arrival rate for the hard task.

    Ramps up for first 25%, sustains peak for 50%, ramps down for last 25%.
    """
    ramp_duration = episode_duration * 0.25
    if t < ramp_duration:
        return base_rate * (0.3 + 0.7 * t / ramp_duration)
    elif t < episode_duration - ramp_duration:
        return base_rate
    else:
        remaining = episode_duration - t
        return base_rate * (0.3 + 0.7 * remaining / ramp_duration)


def generate_itineraries(config: TaskConfig, seed: int | None = None) -> List[Itinerary]:
    """Generate vehicle itineraries for a task configuration.

    Args:
        config: Task configuration.
        seed: Random seed. If None, uses config.seed.

    Returns:
        List of Itinerary objects sorted by start_time.
    """
    rng = random.Random(seed if seed is not None else config.seed)
    num_nodes = config.num_intersections
    itineraries: List[Itinerary] = []

    # Set up corridors for hard task
    corridors = []
    if config.name == "hard":
        corridors = _get_hard_corridors(config.grid_rows, config.grid_cols)
        config = TaskConfig(**{**config.__dict__, "corridors": corridors})

    for t in range(config.episode_duration):
        # Determine arrival rate for this second
        if config.time_varying:
            rate = _arrival_rate_profile(t, config.arrival_rate_per_second, config.episode_duration)
        else:
            rate = config.arrival_rate_per_second

        # Sample number of arrivals (Poisson)
        num_arrivals = _poisson_sample(rng, rate)

        for _ in range(num_arrivals):
            # Pick origin and destination
            if corridors and rng.random() < 0.4:
                # 40% chance to use a commuter corridor
                origin, destination = rng.choice(corridors)
            else:
                origin = rng.randint(0, num_nodes - 1)
                destination = rng.randint(0, num_nodes - 1)
                while destination == origin:
                    destination = rng.randint(0, num_nodes - 1)

            # Pick vehicle category
            category = rng.choices(config.vehicle_categories, weights=config.category_weights, k=1)[0]

            itineraries.append(
                Itinerary(
                    start_time=t,
                    origin=origin,
                    destination=destination,
                    category=category,
                )
            )

    return itineraries


def _poisson_sample(rng: random.Random, lam: float) -> int:
    """Sample from Poisson distribution using inverse transform."""
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p < L:
            break
    return k - 1
