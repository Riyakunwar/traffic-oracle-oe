"""Tests for the traffic simulation engine."""

import pytest
from redgrid.network import RoadNetwork, SignalPhase
from redgrid.vehicles import Itinerary, VehicleCategory
from redgrid.simulation import TrafficSimulator


class TestSpawning:
    def test_vehicle_spawns_at_correct_time(self):
        net = RoadNetwork.grid(2, 2, road_length=5)
        itins = [Itinerary(start_time=3, origin=0, destination=3, category=VehicleCategory.SMALL_CAR)]
        sim = TrafficSimulator(net, itins)

        # Tick 0, 1, 2: no vehicles
        for t in range(3):
            sim.tick([0, 0, 0, 0])
            assert len(sim.vehicles) == 0

        # Tick 3: vehicle spawns
        sim.tick([0, 0, 0, 0])
        assert len(sim.vehicles) == 1

    def test_unreachable_destination_skipped(self):
        net = RoadNetwork.grid(2, 2, road_length=5)
        # Same origin and destination — should be skipped
        itins = [Itinerary(start_time=0, origin=0, destination=0, category=VehicleCategory.SMALL_CAR)]
        sim = TrafficSimulator(net, itins)
        sim.tick([0, 0, 0, 0])
        assert len(sim.vehicles) == 0


class TestMovement:
    def test_vehicle_traverses_edge(self):
        net = RoadNetwork.grid(2, 2, road_length=3)
        # Vehicle goes 0 -> 1 (adjacent, east direction)
        itins = [Itinerary(start_time=0, origin=0, destination=1, category=VehicleCategory.SMALL_CAR)]
        sim = TrafficSimulator(net, itins)

        # Tick 0: spawn and move in same tick, time_on_edge=1
        sim.tick([0, 0, 0, 0])
        assert len(sim.vehicles) == 1
        assert sim.vehicles[0].time_on_edge == 1

        # Tick 1: time_on_edge=2
        sim.tick([0, 0, 0, 0])
        assert sim.vehicles[0].time_on_edge == 2

    def test_vehicle_departs_at_destination(self):
        net = RoadNetwork.grid(2, 2, road_length=2)
        # Vehicle goes 0 -> 1 (destination is node 1)
        # Edge (0,1) approaches node 1 from the WEST → needs EW_GREEN at node 1
        itins = [Itinerary(start_time=0, origin=0, destination=1, category=VehicleCategory.SMALL_CAR)]
        sim = TrafficSimulator(net, itins)

        # Set node 1 to EW_GREEN so vehicle can depart
        net.intersections[1].current_phase = SignalPhase.EW_GREEN

        # Tick 0: spawn
        sim.tick([0, 1, 0, 0])  # keep EW at node 1
        # Tick 1: traversing
        sim.tick([0, 1, 0, 0])
        # Tick 2: time_on_edge == road_length, at destination -> depart
        metrics = sim.tick([0, 1, 0, 0])
        assert metrics.vehicles_departed == 1
        assert sim.total_departed == 1
        assert len(sim.vehicles) == 0


class TestSignalTransitions:
    def test_yellow_transition(self):
        net = RoadNetwork.grid(2, 2, road_length=5, min_green=2, yellow_duration=2)
        sim = TrafficSimulator(net, [])

        # All start NS_GREEN. Request switch at tick 0.
        # min_green=2, so first we need to wait.

        # Tick 0: phase_timer starts at 0, min_green=2, switch ignored
        sim.tick([1, 0, 0, 0])
        assert net.intersections[0].current_phase == SignalPhase.NS_GREEN

        # Tick 1: phase_timer=1, still < min_green
        sim.tick([1, 0, 0, 0])
        assert net.intersections[0].current_phase == SignalPhase.NS_GREEN

        # Tick 2: phase_timer=2, >= min_green. Yellow starts.
        sim.tick([1, 0, 0, 0])
        assert net.intersections[0].in_yellow is True
        assert net.intersections[0].yellow_countdown == 1  # decremented from 2 to 1

        # Tick 3: yellow countdown finishes
        sim.tick([1, 0, 0, 0])
        assert net.intersections[0].in_yellow is False
        assert net.intersections[0].current_phase == SignalPhase.EW_GREEN


class TestMetrics:
    def test_cumulative_wait_tracked(self):
        net = RoadNetwork.grid(3, 3, road_length=2)
        # Vehicle going 0 -> 2 (path: 0->1->2). At node 1, approaches from west (needs EW_GREEN).
        # Keep NS_GREEN so vehicle waits at node 1 trying to advance to edge (1,2).
        itins = [Itinerary(start_time=0, origin=0, destination=2, category=VehicleCategory.SMALL_CAR)]
        sim = TrafficSimulator(net, itins)

        # Keep NS_GREEN at all nodes — vehicle approaching node 1 from west needs EW_GREEN
        for _ in range(10):
            sim.tick([0] * 9)

        # Vehicle should be waiting at node 1 (red signal for EW direction)
        assert sim.total_cumulative_wait > 0

    def test_queue_lengths(self):
        net = RoadNetwork.grid(3, 3, road_length=1)
        # Vehicle going 0 -> 2 (path: 0->1->2). At node 1, approaches from west (needs EW_GREEN).
        itins = [
            Itinerary(start_time=0, origin=0, destination=2, category=VehicleCategory.SMALL_CAR),
        ]
        sim = TrafficSimulator(net, itins)

        # Tick 0: spawn and traverse first edge (road_length=1 so time_on_edge reaches 1)
        sim.tick([0] * 9)
        # Tick 1: at intersection node 1, needs EW_GREEN but has NS_GREEN -> queued
        sim.tick([0] * 9)

        queues = sim.get_queue_lengths()
        # Vehicle approaches node 1 from the west
        assert queues[1]["west"] >= 1
