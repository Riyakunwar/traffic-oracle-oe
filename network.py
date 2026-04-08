"""Road network graph: intersections (nodes), road segments (edges), signal phases."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple


class SignalPhase(IntEnum):
    NS_GREEN = 0
    EW_GREEN = 1


class Direction(str):
    """Cardinal direction from which a vehicle approaches an intersection."""
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


# Map direction to which phase allows it
DIRECTION_TO_PHASE: Dict[str, SignalPhase] = {
    Direction.NORTH: SignalPhase.NS_GREEN,
    Direction.SOUTH: SignalPhase.NS_GREEN,
    Direction.EAST: SignalPhase.EW_GREEN,
    Direction.WEST: SignalPhase.EW_GREEN,
}


@dataclass
class RoadSegment:
    """A directed road segment between two intersections."""

    from_node: int
    to_node: int
    capacity: float  # max car-equivalents
    length: int  # free-flow traversal time in seconds
    current_load: float = 0.0  # current car-equivalents on this segment


@dataclass
class Intersection:
    """A signalized intersection (node in the graph)."""

    node_id: int
    row: int
    col: int
    current_phase: SignalPhase = SignalPhase.NS_GREEN
    phase_timer: int = 0  # seconds in current phase
    min_green: int = 5
    yellow_duration: int = 3
    yellow_countdown: int = 0  # >0 means currently in yellow transition
    pending_phase: Optional[SignalPhase] = None  # phase to switch to after yellow

    @property
    def in_yellow(self) -> bool:
        return self.yellow_countdown > 0


class RoadNetwork:
    """Directed graph of intersections and road segments."""

    def __init__(self) -> None:
        self.intersections: Dict[int, Intersection] = {}
        self.edges: Dict[Tuple[int, int], RoadSegment] = {}
        self.adjacency: Dict[int, List[int]] = {}  # node -> list of neighbor node IDs
        # For each intersection, map (from_node, to_node) to the approach direction
        self.edge_direction: Dict[Tuple[int, int], str] = {}
        self.rows: int = 0
        self.cols: int = 0
        self._path_cache: Dict[Tuple[int, int], List[int]] = {}

    def incoming_edges(self, node_id: int) -> List[Tuple[int, int]]:
        """Return all edges (u, node_id) incoming to this node."""
        return [(u, node_id) for u in self.adjacency if node_id in self.adjacency.get(u, [])]

    def is_green_for_edge(self, from_node: int, to_node: int) -> bool:
        """Check if the signal at to_node allows traffic from from_node."""
        intersection = self.intersections[to_node]
        if intersection.in_yellow:
            return False
        direction = self.edge_direction.get((from_node, to_node))
        if direction is None:
            return False
        required_phase = DIRECTION_TO_PHASE[direction]
        return intersection.current_phase == required_phase

    def has_capacity(self, from_node: int, to_node: int, vehicle_size: float) -> bool:
        """Check if the road segment can accommodate an additional vehicle."""
        edge = self.edges.get((from_node, to_node))
        if edge is None:
            return False
        return edge.current_load + vehicle_size <= edge.capacity

    def shortest_path(self, origin: int, destination: int) -> Optional[List[int]]:
        """Get the cached shortest path between two nodes. Returns None if unreachable."""
        return self._path_cache.get((origin, destination))

    def compute_all_paths(self) -> None:
        """Pre-compute shortest paths between all node pairs using BFS."""
        nodes = list(self.intersections.keys())
        for origin in nodes:
            # BFS from origin
            visited: Dict[int, int] = {origin: -1}  # node -> predecessor
            queue: deque[int] = deque([origin])
            while queue:
                current = queue.popleft()
                for neighbor in self.adjacency.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = current
                        queue.append(neighbor)
            # Reconstruct paths
            for dest in nodes:
                if dest == origin:
                    continue
                if dest not in visited:
                    continue
                path: List[int] = []
                node = dest
                while node != -1:
                    path.append(node)
                    node = visited[node]
                path.reverse()
                self._path_cache[(origin, dest)] = path

    @classmethod
    def grid(
        cls,
        rows: int,
        cols: int,
        road_capacity: float = 20.0,
        road_length: int = 10,
        min_green: int = 5,
        yellow_duration: int = 3,
    ) -> "RoadNetwork":
        """Create a grid road network.

        Node (r, c) has ID r * cols + c.
        Edges connect adjacent nodes in all 4 cardinal directions.
        """
        net = cls()
        net.rows = rows
        net.cols = cols

        # Create intersections
        for r in range(rows):
            for c in range(cols):
                nid = r * cols + c
                net.intersections[nid] = Intersection(
                    node_id=nid,
                    row=r,
                    col=c,
                    min_green=min_green,
                    yellow_duration=yellow_duration,
                )
                net.adjacency[nid] = []

        # Create edges (bidirectional)
        for r in range(rows):
            for c in range(cols):
                nid = r * cols + c
                # North neighbor (r-1, c) -> edge from (r-1,c) to (r,c) means
                # vehicle approaches from the north
                if r > 0:
                    north_id = (r - 1) * cols + c
                    # Edge: nid -> north_id (vehicle going north)
                    net.adjacency[nid].append(north_id)
                    net.edges[(nid, north_id)] = RoadSegment(
                        from_node=nid, to_node=north_id,
                        capacity=road_capacity, length=road_length,
                    )
                    # Approaching north_id from the south (coming from nid which is south)
                    net.edge_direction[(nid, north_id)] = Direction.SOUTH

                # South neighbor
                if r < rows - 1:
                    south_id = (r + 1) * cols + c
                    net.adjacency[nid].append(south_id)
                    net.edges[(nid, south_id)] = RoadSegment(
                        from_node=nid, to_node=south_id,
                        capacity=road_capacity, length=road_length,
                    )
                    # Approaching south_id from the north
                    net.edge_direction[(nid, south_id)] = Direction.NORTH

                # West neighbor
                if c > 0:
                    west_id = r * cols + (c - 1)
                    net.adjacency[nid].append(west_id)
                    net.edges[(nid, west_id)] = RoadSegment(
                        from_node=nid, to_node=west_id,
                        capacity=road_capacity, length=road_length,
                    )
                    # Approaching west_id from the east
                    net.edge_direction[(nid, west_id)] = Direction.EAST

                # East neighbor
                if c < cols - 1:
                    east_id = r * cols + (c + 1)
                    net.adjacency[nid].append(east_id)
                    net.edges[(nid, east_id)] = RoadSegment(
                        from_node=nid, to_node=east_id,
                        capacity=road_capacity, length=road_length,
                    )
                    # Approaching east_id from the west
                    net.edge_direction[(nid, east_id)] = Direction.WEST

        net.compute_all_paths()
        return net
