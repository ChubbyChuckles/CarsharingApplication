from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import pytest

from PyQt6.QtWidgets import QApplication

from src.rideshare_app import DatabaseManager, RideHistoryTab


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def db_manager(tmp_path: Path) -> DatabaseManager:
    return DatabaseManager(tmp_path / "rideshare.db")


def _create_members(db: DatabaseManager) -> Tuple[int, int, int, int]:
    alice = db.add_team_member("Alice", True)
    ben = db.add_team_member("Ben", True)
    chloe = db.add_team_member("Chloe", True)
    dave = db.add_team_member("Dave", False)
    return alice, ben, chloe, dave


def test_snapshot_metrics_reflect_live_ledger_state(
    db_manager: DatabaseManager, qapp
):  # noqa: ARG001
    alice, ben, chloe, dave = _create_members(db_manager)

    db_manager.record_ride(
        start_address="Clubhouse",
        destination_address="Away Venue",
        distance_km=12.5,
        driver_ids=[alice],
        passenger_ids=[chloe, dave],
        paying_passenger_ids=[chloe, dave],
        flat_fee=6.0,
        fee_per_km=0.45,
        total_cost=36.0,
        cost_per_passenger=18.0,
        ride_datetime=datetime(2025, 7, 15, 17, 45, tzinfo=timezone.utc),
    )

    db_manager.record_ride(
        start_address="Training Hall",
        destination_address="Tournament",
        distance_km=18.0,
        driver_ids=[ben],
        passenger_ids=[chloe],
        paying_passenger_ids=[chloe],
        flat_fee=8.0,
        fee_per_km=0.55,
        total_cost=24.0,
        cost_per_passenger=24.0,
        ride_datetime=datetime(2025, 9, 2, 8, 15, tzinfo=timezone.utc),
    )

    tab = RideHistoryTab(db_manager)
    tab.refresh()

    outstanding_value, outstanding_detail = tab._summary_labels["outstanding"]
    assert outstanding_value.text() == "€60.00"
    assert "3 open balances" in outstanding_detail.text()
    assert "4 teammates" in outstanding_detail.text()

    largest_value, largest_detail = tab._summary_labels["largest"]
    assert largest_value.text() == "€24.00"
    assert "Chloe" in largest_detail.text()
    assert "Ben" in largest_detail.text()

    latest_value, latest_detail = tab._summary_labels["latest"]
    assert latest_value.text() == "€24.00"
    detail_text = latest_detail.text()
    assert "Drivers:" in detail_text and "Ben" in detail_text
    assert "Passengers:" in detail_text and "Chloe" in detail_text

    tab.deleteLater()
