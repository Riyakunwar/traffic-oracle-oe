"""Pydantic data models for the Traffic Signal OpenEnv environment."""

from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class IntersectionObs(BaseModel):
    """Per-intersection observation."""

    node_id: int = Field(description="Intersection node ID")
    current_phase: int = Field(description="0=NS_GREEN, 1=EW_GREEN")
    phase_timer: int = Field(description="Seconds in current phase")
    in_yellow: bool = Field(description="Currently in yellow transition")

    queue_north: int = Field(default=0, description="Vehicles queued from north")
    queue_south: int = Field(default=0, description="Vehicles queued from south")
    queue_east: int = Field(default=0, description="Vehicles queued from east")
    queue_west: int = Field(default=0, description="Vehicles queued from west")

    occupancy_north: float = Field(default=0.0, description="Load/capacity ratio, north approach")
    occupancy_south: float = Field(default=0.0, description="Load/capacity ratio, south approach")
    occupancy_east: float = Field(default=0.0, description="Load/capacity ratio, east approach")
    occupancy_west: float = Field(default=0.0, description="Load/capacity ratio, west approach")


class TrafficAction(Action):
    """Agent's signal control decision for all intersections.

    phases is a list of ints, one per intersection (sorted by node ID).
    0 = NS_GREEN, 1 = EW_GREEN.
    """

    phases: List[int] = Field(..., description="Phase choice per intersection: 0=NS_GREEN, 1=EW_GREEN")


class TrafficObservation(Observation):
    """Full observation returned each step."""

    current_time: int = Field(default=0, description="Simulation second (0..7199)")
    task: str = Field(default="easy", description="Current task name")
    intersections: List[IntersectionObs] = Field(
        default_factory=list, description="Per-intersection observations"
    )
    total_vehicles_active: int = Field(default=0, description="Vehicles currently in network")
    total_vehicles_departed: int = Field(default=0, description="Vehicles that reached destination")
    total_vehicles_waiting: int = Field(default=0, description="Vehicles waiting at signals this step")
    total_cumulative_wait: float = Field(default=0.0, description="Sum of all wait-seconds so far")
