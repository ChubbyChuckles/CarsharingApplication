import pytest
from pathlib import Path
from typing import List, Tuple

from PyQt6.QtCore import QThreadPool
from PyQt6.QtWidgets import QApplication, QGridLayout

from src.rideshare_app import (
    DatabaseManager,
    DistanceLookupResult,
    RideHistoryTab,
    RideSetupTab,
    TeamMember,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def db_manager(tmp_path: Path) -> DatabaseManager:
    return DatabaseManager(tmp_path / "rideshare.db")


class _StubMapsHandler:
    enabled = True

    @staticmethod
    def distance_km(_start: str, _destination: str) -> DistanceLookupResult:
        return DistanceLookupResult(distance_km=4.2, from_cache=False, attempts=1)

    @staticmethod
    def autocomplete(_query: str) -> List[str]:
        return []


def _widget_position(layout: QGridLayout, widget) -> Tuple[int, int, int, int]:
    for index in range(layout.count()):
        item = layout.itemAt(index)
        if item is not None and item.widget() is widget:
            return layout.getItemPosition(index)
    raise AssertionError("Widget not found in layout")


def test_ride_setup_tab_switches_layouts(db_manager: DatabaseManager, qapp):  # noqa: ARG001
    maps_handler = _StubMapsHandler()
    tab = RideSetupTab(db_manager, maps_handler, QThreadPool())

    members = [
        TeamMember(member_id=1, name="Alice", is_core=True),
        TeamMember(member_id=2, name="Bob", is_core=True),
    ]
    tab.set_team_members(members)

    tab._apply_responsive_layout(720)
    assert tab._layout_mode == "stacked"

    assert _widget_position(tab._content_layout, tab.address_section)[:2] == (0, 0)
    assert _widget_position(tab._content_layout, tab.team_section)[:2] == (1, 0)
    assert _widget_position(tab._content_layout, tab.fees_section)[:2] == (2, 0)
    assert _widget_position(tab._content_layout, tab.summary_section)[:2] == (3, 0)

    tab._apply_responsive_layout(1360)
    assert tab._layout_mode == "wide"

    assert _widget_position(tab._content_layout, tab.address_section)[:2] == (0, 0)
    assert _widget_position(tab._content_layout, tab.team_section)[:2] == (0, 1)
    assert _widget_position(tab._content_layout, tab.fees_section)[:2] == (1, 0)
    summary_position = _widget_position(tab._content_layout, tab.summary_section)
    assert summary_position[:2] == (2, 0)
    assert summary_position[2:] == (1, 2)  # spans two columns in wide mode

    tab.deleteLater()


def test_ride_history_tab_responsive_layout(db_manager: DatabaseManager, qapp):  # noqa: ARG001
    tab = RideHistoryTab(db_manager)

    tab._apply_responsive_layout(900)
    assert tab._layout_mode == "stacked"

    assert _widget_position(tab._content_layout, tab.snapshot_section)[:2] == (0, 0)
    assert _widget_position(tab._content_layout, tab.rides_section)[:2] == (1, 0)
    assert _widget_position(tab._content_layout, tab.ledger_section)[:2] == (2, 0)

    snapshot_cards_layout = tab._summary_cards_layout
    for row, card in enumerate(tab._summary_cards):
        assert snapshot_cards_layout.itemAtPosition(row, 0).widget() is card

    tab._apply_responsive_layout(1400)
    assert tab._layout_mode == "wide"

    assert _widget_position(tab._content_layout, tab.snapshot_section)[:2] == (0, 0)
    assert _widget_position(tab._content_layout, tab.ledger_section)[:2] == (1, 0)
    assert _widget_position(tab._content_layout, tab.rides_section)[:2] == (0, 1)

    for column, card in enumerate(tab._summary_cards):
        assert snapshot_cards_layout.itemAtPosition(0, column).widget() is card

    tab.deleteLater()
