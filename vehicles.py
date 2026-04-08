"""Vehicle types, itineraries, and runtime vehicle state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class VehicleCategory(str, Enum):
    TWO_WHEELER = "2-wheeler"
    THREE_WHEELER = "3-wheeler"
    SMALL_CAR = "small_car"
    LARGE_VEHICLE = "large_vehicle"


VEHICLE_SIZE: Dict[VehicleCategory, float] = {
    VehicleCategory.TWO_WHEELER: 0.5,
    VehicleCategory.THREE_WHEELER: 0.75,
    VehicleCategory.SMALL_CAR: 1.0,
    VehicleCategory.LARGE_VEHICLE: 2.0,
}


@dataclass
class Itinerary:
    """A trip request: when and where a vehicle wants to travel."""

    start_time: int  # simulation second when vehicle enters the network
    origin: int  # node ID
    destination: int  # node ID
    category: VehicleCategory


@dataclass
class Vehicle:
    """Runtime state of an active vehicle in the simulation."""

    id: int
    itinerary: Itinerary
    path: List[int]  # sequence of node IDs from origin to destination
    current_edge_index: int = 0  # index into path; edge is (path[i], path[i+1])
    wait_time: int = 0  # total seconds spent waiting
    time_on_edge: int = 0  # seconds spent traversing current edge
    departed: bool = False

    @property
    def size(self) -> float:
        return VEHICLE_SIZE[self.itinerary.category]

    @property
    def current_node(self) -> int:
        """The node the vehicle is heading toward (downstream end of current edge)."""
        return self.path[self.current_edge_index + 1]

    @property
    def previous_node(self) -> int:
        """The node the vehicle came from (upstream end of current edge)."""
        return self.path[self.current_edge_index]

    @property
    def current_edge(self) -> tuple[int, int]:
        return (self.path[self.current_edge_index], self.path[self.current_edge_index + 1])

    @property
    def at_destination(self) -> bool:
        return self.current_edge_index >= len(self.path) - 2 and self.departed
