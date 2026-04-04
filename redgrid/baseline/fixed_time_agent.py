"""Fixed-time baseline agent: alternates NS/EW every cycle_length seconds."""

from __future__ import annotations

from ..models import TrafficAction, TrafficObservation


class FixedTimeAgent:
    """Baseline agent that uses a fixed-cycle signal timing.

    All intersections switch between NS_GREEN and EW_GREEN
    on a fixed schedule (default: every 30 seconds).
    """

    def __init__(self, num_intersections: int, cycle_length: int = 30) -> None:
        self.num_intersections = num_intersections
        self.cycle_length = cycle_length

    def act(self, observation: TrafficObservation) -> TrafficAction:
        t = observation.current_time
        phase = 0 if (t // self.cycle_length) % 2 == 0 else 1
        return TrafficAction(phases=[phase] * self.num_intersections)
