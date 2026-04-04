"""Tests for the road network graph."""

import pytest
from redgrid.network import RoadNetwork, SignalPhase, Direction


class TestGridCreation:
    def test_2x2_grid_has_4_nodes(self):
        net = RoadNetwork.grid(2, 2)
        assert len(net.intersections) == 4

    def test_2x2_grid_has_8_edges(self):
        """2x2 grid: 4 horizontal edges + 4 vertical edges = 8 total."""
        net = RoadNetwork.grid(2, 2)
        assert len(net.edges) == 8

    def test_4x4_grid_has_16_nodes(self):
        net = RoadNetwork.grid(4, 4)
        assert len(net.intersections) == 16

    def test_4x4_grid_edges(self):
        """4x4 grid: 3*4*2 horizontal + 4*3*2 vertical = 48 edges."""
        net = RoadNetwork.grid(4, 4)
        assert len(net.edges) == 48

    def test_node_ids_sequential(self):
        net = RoadNetwork.grid(3, 3)
        assert set(net.intersections.keys()) == set(range(9))

    def test_custom_capacity_and_length(self):
        net = RoadNetwork.grid(2, 2, road_capacity=15.0, road_length=8)
        for edge in net.edges.values():
            assert edge.capacity == 15.0
            assert edge.length == 8


class TestShortestPaths:
    def test_adjacent_nodes(self):
        net = RoadNetwork.grid(3, 3)
        path = net.shortest_path(0, 1)
        assert path == [0, 1]

    def test_diagonal_path(self):
        """From (0,0) to (2,2) in a 3x3 grid — should be 4 hops."""
        net = RoadNetwork.grid(3, 3)
        path = net.shortest_path(0, 8)
        assert path is not None
        assert len(path) == 5  # 0 -> 1 -> 2 -> 5 -> 8 or similar
        assert path[0] == 0
        assert path[-1] == 8

    def test_same_node_no_path(self):
        net = RoadNetwork.grid(3, 3)
        path = net.shortest_path(4, 4)
        assert path is None

    def test_all_pairs_reachable(self):
        net = RoadNetwork.grid(4, 4)
        for i in range(16):
            for j in range(16):
                if i != j:
                    assert net.shortest_path(i, j) is not None


class TestSignalLogic:
    def test_default_phase_is_ns_green(self):
        net = RoadNetwork.grid(2, 2)
        for inter in net.intersections.values():
            assert inter.current_phase == SignalPhase.NS_GREEN

    def test_green_for_ns_direction(self):
        net = RoadNetwork.grid(3, 3)
        # Edge from node 1 (row=0,col=1) to node 4 (row=1,col=1)
        # This approaches node 4 from the NORTH → needs NS_GREEN
        assert net.is_green_for_edge(1, 4) is True

    def test_red_for_ew_direction_when_ns_green(self):
        net = RoadNetwork.grid(3, 3)
        # Edge from node 3 (row=1,col=0) to node 4 (row=1,col=1)
        # This approaches node 4 from the WEST → needs EW_GREEN
        assert net.is_green_for_edge(3, 4) is False

    def test_capacity_check(self):
        net = RoadNetwork.grid(2, 2, road_capacity=10.0)
        assert net.has_capacity(0, 1, 5.0) is True
        net.edges[(0, 1)].current_load = 9.5
        assert net.has_capacity(0, 1, 1.0) is False
