"""Tests for vehicle types and itineraries."""

from vehicles import (
    Vehicle,
    Itinerary,
    VehicleCategory,
    VEHICLE_SIZE,
)


def test_vehicle_sizes():
    assert VEHICLE_SIZE[VehicleCategory.TWO_WHEELER] == 0.5
    assert VEHICLE_SIZE[VehicleCategory.THREE_WHEELER] == 0.75
    assert VEHICLE_SIZE[VehicleCategory.SMALL_CAR] == 1.0
    assert VEHICLE_SIZE[VehicleCategory.LARGE_VEHICLE] == 2.0


def test_vehicle_current_edge():
    itin = Itinerary(0, 0, 8, VehicleCategory.SMALL_CAR)
    v = Vehicle(id=0, itinerary=itin, path=[0, 1, 2, 5, 8])
    assert v.current_edge == (0, 1)
    assert v.current_node == 1
    assert v.previous_node == 0


def test_vehicle_advance():
    itin = Itinerary(0, 0, 2, VehicleCategory.SMALL_CAR)
    v = Vehicle(id=0, itinerary=itin, path=[0, 1, 2])
    v.current_edge_index = 1
    assert v.current_edge == (1, 2)
    assert v.current_node == 2


def test_vehicle_size_property():
    itin = Itinerary(0, 0, 1, VehicleCategory.LARGE_VEHICLE)
    v = Vehicle(id=0, itinerary=itin, path=[0, 1])
    assert v.size == 2.0
