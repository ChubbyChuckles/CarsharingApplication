from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import pytest
from PyQt6.QtWidgets import QApplication

from src.rideshare_app import AnalyticsTab, DatabaseManager


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def db_manager(tmp_path: Path) -> DatabaseManager:
    return DatabaseManager(tmp_path / "analytics.db")


def _seed_sample_data(db: DatabaseManager) -> Tuple[int, int, int]:
    alice = db.add_team_member("Alice", True)
    ben = db.add_team_member("Ben", True)
    chloe = db.add_team_member("Chloe", True)

    ride_one = db.record_ride(
        start_address="Clubhouse",
        destination_address="Tournament",
        distance_km=12.5,
        driver_ids=[alice],
        passenger_ids=[ben, chloe],
        paying_passenger_ids=[ben, chloe],
        flat_fee=6.0,
        fee_per_km=0.45,
        total_cost=30.0,
        cost_per_passenger=15.0,
    )
    _set_ride_timestamp(db, ride_one, "2025-08-18T10:30:00+00:00")

    ride_two = db.record_ride(
        start_address="Training Hall",
        destination_address="League Night",
        distance_km=18.0,
        driver_ids=[ben],
        passenger_ids=[alice],
        paying_passenger_ids=[alice],
        flat_fee=8.0,
        fee_per_km=0.50,
        total_cost=20.0,
        cost_per_passenger=20.0,
    )
    _set_ride_timestamp(db, ride_two, "2025-09-12T18:05:00+00:00")

    ride_three = db.record_ride(
        start_address="Clubhouse",
        destination_address="Practice",
        distance_km=9.0,
        driver_ids=[alice],
        passenger_ids=[ben],
        paying_passenger_ids=[ben],
        flat_fee=5.0,
        fee_per_km=0.40,
        total_cost=10.0,
        cost_per_passenger=10.0,
    )
    _set_ride_timestamp(db, ride_three, "2025-10-02T09:15:00+00:00")

    return alice, ben, chloe


def _set_ride_timestamp(db: DatabaseManager, ride_id: int, timestamp: str) -> None:
    with db._connect() as conn:  # type: ignore[attr-defined]
        conn.execute("UPDATE rides SET ride_datetime = ? WHERE id = ?", (timestamp, ride_id))
        conn.commit()


def test_member_cost_trends_grouped_by_month(db_manager: DatabaseManager) -> None:
    _seed_sample_data(db_manager)
    reference = datetime(2025, 10, 15, tzinfo=timezone.utc)

    trend = db_manager.fetch_member_cost_trends(months=3, reference=reference)

    assert trend["periods"] == ["2025-08", "2025-09", "2025-10"]

    series_lookup = {entry["member"]: entry["values"] for entry in trend["series"]}

    assert series_lookup["Alice"] == [0.0, 20.0, 0.0]
    assert series_lookup["Ben"] == [15.0, 0.0, 10.0]
    assert series_lookup["Chloe"] == [15.0, 0.0, 0.0]


def test_ride_frequency_matrix_counts(db_manager: DatabaseManager) -> None:
    _seed_sample_data(db_manager)

    frequency = db_manager.fetch_ride_frequency()

    assert frequency["weekday_labels"] == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    assert frequency["hour_labels"][0] == "00"
    matrix = frequency["matrix"]
    # Monday 10:00 (ride one)
    assert matrix[0][10] == 1
    # Friday 18:00 (ride two)
    assert matrix[4][18] == 1
    # Thursday 09:00 (ride three)
    assert matrix[3][9] == 1
    assert frequency["total_rides"] == 3


def test_analytics_tab_refresh_toggles_views(
    qapp, db_manager: DatabaseManager
) -> None:  # noqa: ARG001
    tab = AnalyticsTab(db_manager)
    tab.refresh()

    assert tab.cost_stack.currentIndex() == 1
    assert tab.heatmap_stack.currentIndex() == 1

    _seed_sample_data(db_manager)
    tab.refresh()

    assert tab.cost_stack.currentIndex() == 0
    assert tab.heatmap_stack.currentIndex() == 0
    assert tab._latest_frequency_total == 3  # type: ignore[attr-defined]

    tab.deleteLater()
