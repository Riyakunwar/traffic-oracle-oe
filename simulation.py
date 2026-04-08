"""Traffic simulation engine — core tick-by-tick logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from network import RoadNetwork, SignalPhase
from vehicles import Itinerary, Vehicle, VEHICLE_SIZE


@dataclass
class StepMetrics:
    """Metrics produced by a single simulation tick."""

    wait_increment: int = 0  # total wait-seconds added this tick
    vehicles_waiting: int = 0  # vehicles currently waiting at signals
    vehicles_departed: int = 0  # vehicles that reached destination this tick
    vehicles_active: int = 0  # vehicles currently in the network
    switches: int = 0  # number of intersections that initiated a phase switch


class TrafficSimulator:
    """Discrete-time traffic simulation on a road network.

    Each call to tick() advances the simulation by 1 second.
    """

    def __init__(self, network: RoadNetwork, itineraries: List[Itinerary]) -> None:
        self.network = network
        self.itineraries = itineraries
        # Group itineraries by start_time for efficient lookup
        self._itineraries_by_time: Dict[int, List[Itinerary]] = {}
        for itin in itineraries:
            self._itineraries_by_time.setdefault(itin.start_time, []).append(itin)

        self.current_time: int = 0
        self.vehicles: List[Vehicle] = []  # active vehicles
        self.next_vehicle_id: int = 0
        self.total_departed: int = 0
        self.total_cumulative_wait: int = 0
        self.total_spawned: int = 0

    def tick(self, phase_actions: List[int]) -> StepMetrics:
        """Advance simulation by one second.

        Args:
            phase_actions: one int per intersection, 0=NS_GREEN, 1=EW_GREEN.

        Returns:
            StepMetrics for this tick.
        """
        metrics = StepMetrics()

        # 1. Apply signal actions
        metrics.switches = self._apply_signals(phase_actions)

        # 2. Update yellow countdowns
        self._update_yellows()

        # 3. Spawn new vehicles
        self._spawn_vehicles()

        # 4. Move vehicles
        departed_ids = self._move_vehicles(metrics)

        # 5. Remove departed vehicles
        self.vehicles = [v for v in self.vehicles if v.id not in departed_ids]

        # 6. Update edge loads
        self._update_edge_loads()

        # 7. Advance phase timers
        for intersection in self.network.intersections.values():
            intersection.phase_timer += 1

        metrics.vehicles_active = len(self.vehicles)
        self.current_time += 1

        return metrics

    def _apply_signals(self, phase_actions: List[int]) -> int:
        """Apply requested phase changes. Returns number of switches initiated."""
        switches = 0
        nodes = sorted(self.network.intersections.keys())
        for i, node_id in enumerate(nodes):
            if i >= len(phase_actions):
                break
            intersection = self.network.intersections[node_id]
            requested = SignalPhase(phase_actions[i])

            if intersection.in_yellow:
                # Already transitioning, ignore action
                continue

            if requested != intersection.current_phase:
                if intersection.phase_timer >= intersection.min_green:
                    # Start yellow transition
                    intersection.yellow_countdown = intersection.yellow_duration
                    intersection.pending_phase = requested
                    switches += 1
                # else: ignore, min green not met

        return switches

    def _update_yellows(self) -> None:
        """Decrement yellow countdowns and complete phase switches."""
        for intersection in self.network.intersections.values():
            if intersection.yellow_countdown > 0:
                intersection.yellow_countdown -= 1
                if intersection.yellow_countdown == 0 and intersection.pending_phase is not None:
                    intersection.current_phase = intersection.pending_phase
                    intersection.pending_phase = None
                    intersection.phase_timer = 0

    def _spawn_vehicles(self) -> None:
        """Spawn vehicles whose start_time matches current_time."""
        arrivals = self._itineraries_by_time.get(self.current_time, [])
        for itin in arrivals:
            path = self.network.shortest_path(itin.origin, itin.destination)
            if path is None or len(path) < 2:
                continue  # unreachable or same-node trip
            vehicle = Vehicle(
                id=self.next_vehicle_id,
                itinerary=itin,
                path=path,
            )
            self.next_vehicle_id += 1
            self.total_spawned += 1
            self.vehicles.append(vehicle)

    def _move_vehicles(self, metrics: StepMetrics) -> set:
        """Process vehicle movement. Returns set of departed vehicle IDs."""
        departed_ids: set = set()

        for vehicle in self.vehicles:
            edge = vehicle.current_edge
            road = self.network.edges[edge]

            if vehicle.time_on_edge < road.length:
                # Still traversing the edge
                vehicle.time_on_edge += 1
                continue

            # Vehicle has reached the downstream intersection
            to_node = vehicle.current_node

            # Check if this is the final destination
            if vehicle.current_edge_index >= len(vehicle.path) - 2:
                # At destination
                vehicle.departed = True
                departed_ids.add(vehicle.id)
                metrics.vehicles_departed += 1
                self.total_departed += 1
                continue

            # Try to advance to next edge
            next_from = vehicle.path[vehicle.current_edge_index + 1]
            next_to = vehicle.path[vehicle.current_edge_index + 2]

            # Check signal
            if not self.network.is_green_for_edge(vehicle.previous_node, to_node):
                vehicle.wait_time += 1
                metrics.wait_increment += 1
                metrics.vehicles_waiting += 1
                self.total_cumulative_wait += 1
                continue

            # Check capacity on next edge
            if not self.network.has_capacity(next_from, next_to, vehicle.size):
                vehicle.wait_time += 1
                metrics.wait_increment += 1
                metrics.vehicles_waiting += 1
                self.total_cumulative_wait += 1
                continue

            # Advance to next edge
            vehicle.current_edge_index += 1
            vehicle.time_on_edge = 0

        return departed_ids

    def _update_edge_loads(self) -> None:
        """Recompute current_load for all edges based on vehicle positions."""
        # Reset all loads
        for edge in self.network.edges.values():
            edge.current_load = 0.0

        # Sum vehicle sizes on each edge
        for vehicle in self.vehicles:
            edge_key = vehicle.current_edge
            edge = self.network.edges.get(edge_key)
            if edge is not None:
                edge.current_load += vehicle.size

    def get_queue_lengths(self) -> Dict[int, Dict[str, int]]:
        """Get per-intersection, per-direction queue lengths (vehicles waiting at signal)."""
        queues: Dict[int, Dict[str, int]] = {}
        for node_id in self.network.intersections:
            queues[node_id] = {"north": 0, "south": 0, "east": 0, "west": 0}

        for vehicle in self.vehicles:
            edge = vehicle.current_edge
            road = self.network.edges[edge]
            to_node = vehicle.current_node

            # Only count as queued if vehicle has reached the intersection end
            if vehicle.time_on_edge >= road.length:
                direction = self.network.edge_direction.get(edge)
                if direction and to_node in queues:
                    queues[to_node][direction] += 1

        return queues

    def get_edge_occupancies(self) -> Dict[int, Dict[str, float]]:
        """Get per-intersection, per-direction occupancy ratios for incoming edges."""
        occupancies: Dict[int, Dict[str, float]] = {}
        for node_id in self.network.intersections:
            occupancies[node_id] = {"north": 0.0, "south": 0.0, "east": 0.0, "west": 0.0}

        for (from_node, to_node), edge in self.network.edges.items():
            direction = self.network.edge_direction.get((from_node, to_node))
            if direction and to_node in occupancies:
                occupancies[to_node][direction] = (
                    edge.current_load / edge.capacity if edge.capacity > 0 else 0.0
                )

        return occupancies
