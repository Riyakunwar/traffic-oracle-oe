"""Tests for the OpenEnv TrafficEnvironment."""

import pytest

# These tests import the environment directly (no openenv server needed)
from network import RoadNetwork
from vehicles import Itinerary, VehicleCategory
from simulation import TrafficSimulator
from tasks import TASKS, generate_itineraries
from graders import compute_score


class TestTaskConfigs:
    def test_all_tasks_exist(self):
        assert "easy" in TASKS
        assert "medium" in TASKS
        assert "hard" in TASKS

    def test_easy_is_4x4(self):
        assert TASKS["easy"].grid_rows == 4
        assert TASKS["easy"].grid_cols == 4

    def test_medium_is_7x7(self):
        assert TASKS["medium"].grid_rows == 7
        assert TASKS["medium"].grid_cols == 7

    def test_hard_is_10x10(self):
        assert TASKS["hard"].grid_rows == 10
        assert TASKS["hard"].grid_cols == 10

    def test_episode_duration(self):
        for config in TASKS.values():
            assert config.episode_duration == 7200


class TestItineraryGeneration:
    def test_deterministic_with_seed(self):
        itins1 = generate_itineraries(TASKS["easy"], seed=42)
        itins2 = generate_itineraries(TASKS["easy"], seed=42)
        assert len(itins1) == len(itins2)
        for a, b in zip(itins1[:100], itins2[:100]):
            assert a.start_time == b.start_time
            assert a.origin == b.origin
            assert a.destination == b.destination
            assert a.category == b.category

    def test_different_seeds_differ(self):
        itins1 = generate_itineraries(TASKS["easy"], seed=42)
        itins2 = generate_itineraries(TASKS["easy"], seed=99)
        # Very unlikely to be identical
        origins1 = [i.origin for i in itins1[:50]]
        origins2 = [i.origin for i in itins2[:50]]
        assert origins1 != origins2

    def test_easy_generates_reasonable_count(self):
        itins = generate_itineraries(TASKS["easy"], seed=42)
        # 0.5/s * 7200s = ~3600 expected
        assert 2000 < len(itins) < 6000

    def test_hard_generates_more_than_easy(self):
        easy = generate_itineraries(TASKS["easy"], seed=42)
        hard = generate_itineraries(TASKS["hard"], seed=42)
        assert len(hard) > len(easy) * 5


class TestGrader:
    def test_score_at_worst(self):
        score = compute_score("easy", 50_000.0)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_score_at_best(self):
        score = compute_score("easy", 5_000.0)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_score_in_between(self):
        score = compute_score("easy", 27_500.0)
        assert 0.4 < score < 0.6

    def test_score_below_best_clamped(self):
        score = compute_score("easy", 0.0)
        assert score == 1.0

    def test_score_above_worst_clamped(self):
        score = compute_score("easy", 100_000.0)
        assert score == 0.0

    def test_unknown_task(self):
        score = compute_score("nonexistent", 1000.0)
        assert score == 0.0


class TestSimulationIntegration:
    def test_easy_short_episode(self):
        """Run 100 steps of the easy task to verify everything wires together."""
        config = TASKS["easy"]
        net = RoadNetwork.grid(
            config.grid_rows, config.grid_cols,
            road_capacity=config.road_capacity,
            road_length=config.road_length,
        )
        itins = generate_itineraries(config, seed=42)
        sim = TrafficSimulator(net, itins)

        for _ in range(100):
            metrics = sim.tick([0] * config.num_intersections)

        assert sim.current_time == 100
        assert sim.total_spawned > 0
