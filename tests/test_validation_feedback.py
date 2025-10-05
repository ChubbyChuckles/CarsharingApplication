import pytest
from pathlib import Path
from typing import List
from PyQt6.QtCore import QThreadPool
from PyQt6.QtWidgets import QApplication

from src.rideshare_app import (
    DatabaseManager,
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
    def distance_km(_start: str, _destination: str) -> float:
        return 4.2

    @staticmethod
    def autocomplete(_query: str) -> List[str]:
        return []


@pytest.fixture
def db_manager(tmp_path: Path) -> DatabaseManager:
    return DatabaseManager(tmp_path / "rideshare.db")


def test_team_management_banner_shows_inline_errors(
    db_manager: DatabaseManager, qapp
):  # noqa: ARG001
    tab = TeamManagementTab(db_manager)
    tab.name_input.clear()

    tab._on_add_member()

    assert tab.feedback_banner.messages
    assert tab.feedback_banner.severity == "warning"
    assert "name" in " ".join(tab.feedback_banner.messages).lower()
    assert tab.name_input.property("validationState") == "error"

    tab.name_input.setText("Alice")
    tab._on_add_member()

    assert tab.feedback_banner.severity == "success"
    assert "added" in " ".join(tab.feedback_banner.messages).lower()
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
        distance_km=5.0,
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
