import pytest
from pathlib import Path
from typing import List
from PyQt6.QtCore import QThreadPool
from PyQt6.QtWidgets import QApplication

from src.rideshare_app import (
    DatabaseManager,
    DistanceLookupResult,
    RideSetupTab,
    TeamManagementTab,
    TeamMember,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class _StubMapsHandler:
    enabled = True

    @staticmethod
    def distance_km(_start: str, _destination: str) -> DistanceLookupResult:
        return DistanceLookupResult(distance_km=4.2, from_cache=False, attempts=1)

    @staticmethod
    def autocomplete(_query: str) -> List[str]:
        return []


@pytest.fixture
def db_manager(tmp_path: Path) -> DatabaseManager:
    return DatabaseManager(tmp_path / "rideshare.db")


def test_team_management_status_shows_inline_errors(
    db_manager: DatabaseManager, qapp
):  # noqa: ARG001
    tab = TeamManagementTab(db_manager)
    tab.name_input.clear()

    tab._on_add_member()

    assert tab.status_label.messages
    assert tab.status_label.severity == "warning"
    assert "name" in " ".join(tab.status_label.messages).lower()
    assert tab.name_input.property("validationState") == "error"

    tab.name_input.setText("Alice")
    tab._on_add_member()

    assert tab.status_label.severity == "success"
    assert "added" in " ".join(tab.status_label.messages).lower()
    assert tab.name_input.property("validationState") == ""

    tab.deleteLater()


def test_ride_setup_validation_collects_errors_and_success(
    db_manager: DatabaseManager, qapp
):  # noqa: ARG001
    maps_handler = _StubMapsHandler()
    tab = RideSetupTab(db_manager, maps_handler, QThreadPool())

    members = [
        TeamMember(member_id=1, name="Alice", is_core=True),
        TeamMember(member_id=2, name="Bob", is_core=True),
    ]
    tab.set_team_members(members)

    tab._on_calculate_clicked()

    assert tab.validation_banner.messages
    assert tab.validation_banner.severity == "warning"
    assert "start" in " ".join(tab.validation_banner.messages).lower()

    tab.start_input.setText("Start Street")
    tab.dest_input.setText("Destination Avenue")
    tab.driver_list.item(0).setSelected(True)
    tab.passenger_list.item(1).setSelected(True)

    form_state, errors = tab._collect_form_state()
    assert not errors
    assert form_state is not None

    tab._on_distance_ready(
        DistanceLookupResult(distance_km=5.0, from_cache=False, attempts=1),
        flat_fee=form_state["flat_fee"],
        per_km_fee=form_state["per_km_fee"],
        passenger_ids=form_state["passenger_ids"],
        core_passenger_ids=form_state["core_passenger_ids"],
        driver_ids=form_state["driver_ids"],
    )

    assert tab.save_button.isEnabled()
    assert tab.validation_banner.severity == "success"
    assert "calculation" in " ".join(tab.validation_banner.messages).lower()

    tab.deleteLater()


def test_team_management_shows_participation_counts(
    db_manager: DatabaseManager, qapp
):  # noqa: ARG001
    driver_id = db_manager.add_team_member("Driver Dave", True)
    passenger_id = db_manager.add_team_member("Passenger Pam", False)

    db_manager.record_ride(
        start_address="Arena",
        destination_address="Club",
        distance_km=12.5,
        driver_ids=[driver_id],
        passenger_ids=[passenger_id],
        paying_passenger_ids=[passenger_id],
        flat_fee=5.0,
        fee_per_km=0.5,
        total_cost=11.25,
        cost_per_passenger=11.25,
    )

    tab = TeamManagementTab(db_manager)
    tab.refresh_members()

    participation: dict[str, tuple[int, int]] = {}
    for row in range(tab.table.rowCount()):
        name_item = tab.table.item(row, 0)
        driver_item = tab.table.item(row, 2)
        passenger_item = tab.table.item(row, 3)
        if not name_item or not driver_item or not passenger_item:
            continue
        participation[name_item.text()] = (
            int(driver_item.text()),
            int(passenger_item.text()),
        )

    assert participation.get("Driver Dave") == (1, 0)
    assert participation.get("Passenger Pam") == (0, 1)

    tab.deleteLater()
