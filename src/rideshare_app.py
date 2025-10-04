"""PyQt6 ride-sharing manager for table tennis team.

This module builds a fully featured GUI that helps the team manage shared rides, split
costs, and keep a persistent ledger. It demonstrates:

* Google Maps Places Autocomplete and Distance Matrix API usage via ``googlemaps``.
* SQLite-backed persistence for team members, ride history, and outstanding balances.
* A polished PyQt6 interface styled with an external QSS theme.

API key setup
------------
Store your Google Maps API key in an environment variable named ``GOOGLE_MAPS_API_KEY``.
For local development you can create a ``.env`` file next to this script containing::

    GOOGLE_MAPS_API_KEY=your-secret-key

The application loads this variable on start-up. The key must have access to the
"Places API" and the "Distance Matrix API".
"""

from __future__ import annotations

import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

import googlemaps
from dotenv import load_dotenv
from googlemaps.exceptions import ApiError, TransportError
from PyQt6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, pyqtSignal, QStringListModel
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


DATABASE_FILE = Path(__file__).resolve().parent / "rideshare.db"
STYLE_FILE = Path(__file__).resolve().parent / "resources" / "style.qss"


class GoogleMapsError(RuntimeError):
    """Domain-specific error raised for Google Maps integration problems."""


class WorkerSignals(QObject):
    """Signals available from a running background worker."""

    finished = pyqtSignal(object)
    error = pyqtSignal(str)


class Worker(QRunnable):
    """Utility runnable that executes callables on the thread pool."""

    def __init__(self, fn: Callable, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self) -> None:  # pragma: no cover - executed in a worker thread
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            self.signals.error.emit(str(exc))
        else:
            self.signals.finished.emit(result)


class GoogleMapsHandler:
    """Wrapper around the ``googlemaps`` client with helpful defaults."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise GoogleMapsError(
                "Google Maps API key missing. Set the GOOGLE_MAPS_API_KEY environment variable."
            )
        try:
            self.client = googlemaps.Client(key=api_key)
        except (ApiError, TransportError) as exc:  # pragma: no cover - network issues
            raise GoogleMapsError(f"Unable to initialise Google Maps client: {exc}") from exc

    def autocomplete(self, query: str) -> List[str]:
        """Return address suggestions for the provided query string."""
        if not query.strip():
            return []
        try:
            predictions = self.client.places_autocomplete(
                input_text=query,
                types="geocode",
                language="en",
            )
        except (ApiError, TransportError) as exc:  # pragma: no cover - network issues
            raise GoogleMapsError(f"Autocomplete request failed: {exc}") from exc
        return [item.get("description", "") for item in predictions]

    def distance_km(self, origin: str, destination: str) -> float:
        """Return the driving distance between two addresses in kilometres."""
        try:
            matrix = self.client.distance_matrix(
                origins=[origin],
                destinations=[destination],
                mode="driving",
                units="metric",
            )
        except (ApiError, TransportError) as exc:  # pragma: no cover - network issues
            raise GoogleMapsError(f"Distance Matrix request failed: {exc}") from exc

        status = matrix.get("status")
        if status != "OK":
            raise GoogleMapsError(f"Distance Matrix response status not OK: {status}")

        rows = matrix.get("rows", [])
        if not rows:
            raise GoogleMapsError("Distance Matrix returned no rows.")

        elements = rows[0].get("elements", [])
        if not elements:
            raise GoogleMapsError("Distance Matrix returned no elements.")

        element = elements[0]
        if element.get("status") != "OK":
            raise GoogleMapsError(f"Distance Matrix element status not OK: {element.get('status')}")

        distance_meters = element.get("distance", {}).get("value")
        if distance_meters is None:
            raise GoogleMapsError("Distance value missing in Distance Matrix response.")
        return round(distance_meters / 1000.0, 2)


class DatabaseManager:
    """Manage all SQLite operations for the application."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _ensure_schema(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS team_members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            is_core INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS rides (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_address TEXT NOT NULL,
            destination_address TEXT NOT NULL,
            distance_km REAL NOT NULL,
            driver_id INTEGER NOT NULL,
            flat_fee REAL NOT NULL,
            fee_per_km REAL NOT NULL,
            total_cost REAL NOT NULL,
            cost_per_passenger REAL NOT NULL,
            ride_datetime TEXT NOT NULL,
            FOREIGN KEY(driver_id) REFERENCES team_members(id) ON DELETE RESTRICT
        );

        CREATE TABLE IF NOT EXISTS ride_passengers (
            ride_id INTEGER NOT NULL,
            passenger_id INTEGER NOT NULL,
            PRIMARY KEY (ride_id, passenger_id),
            FOREIGN KEY(ride_id) REFERENCES rides(id) ON DELETE CASCADE,
            FOREIGN KEY(passenger_id) REFERENCES team_members(id) ON DELETE RESTRICT
        );

        CREATE TABLE IF NOT EXISTS ledger_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ride_id INTEGER NOT NULL,
            driver_id INTEGER NOT NULL,
            passenger_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            FOREIGN KEY(ride_id) REFERENCES rides(id) ON DELETE CASCADE,
            FOREIGN KEY(driver_id) REFERENCES team_members(id) ON DELETE CASCADE,
            FOREIGN KEY(passenger_id) REFERENCES team_members(id) ON DELETE CASCADE
        );
        """
        with self._connect() as conn:
            conn.executescript(schema)

            # Apply lightweight migration for existing databases that pre-date the
            # ``is_core`` flag. ``ALTER TABLE`` will raise an operational error if the
            # column already exists, so we only run it when required.
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(team_members)")}
            if "is_core" not in columns:
                conn.execute(
                    "ALTER TABLE team_members ADD COLUMN is_core INTEGER NOT NULL DEFAULT 1"
                )
            conn.commit()

    # Team management -----------------------------------------------------
    def add_team_member(self, name: str, is_core: bool) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO team_members (name, is_core) VALUES (?, ?)",
                (name.strip(), int(is_core)),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def update_team_member(self, member_id: int, new_name: str, is_core: bool) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE team_members SET name = ?, is_core = ? WHERE id = ?",
                (new_name.strip(), int(is_core), member_id),
            )
            conn.commit()

    def delete_team_member(self, member_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM team_members WHERE id = ?", (member_id,))
            conn.commit()

    def fetch_team_members(self) -> List[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, is_core FROM team_members ORDER BY name COLLATE NOCASE"
            ).fetchall()
        return [dict(row) for row in rows]

    # Ride storage --------------------------------------------------------
    def record_ride(
        self,
        start_address: str,
        destination_address: str,
        distance_km: float,
        driver_id: int,
        passenger_ids: Iterable[int],
        paying_passenger_ids: Iterable[int],
        flat_fee: float,
        fee_per_km: float,
        total_cost: float,
        cost_per_passenger: float,
    ) -> int:
        passenger_ids = list(passenger_ids)
        paying_passenger_ids = list(paying_passenger_ids)
        timestamp = datetime.utcnow().isoformat(timespec="seconds")
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO rides (
                    start_address,
                    destination_address,
                    distance_km,
                    driver_id,
                    flat_fee,
                    fee_per_km,
                    total_cost,
                    cost_per_passenger,
                    ride_datetime
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    start_address,
                    destination_address,
                    distance_km,
                    driver_id,
                    flat_fee,
                    fee_per_km,
                    total_cost,
                    cost_per_passenger,
                    timestamp,
                ),
            )
            ride_id = int(cursor.lastrowid)

            for passenger_id in passenger_ids:
                conn.execute(
                    "INSERT INTO ride_passengers (ride_id, passenger_id) VALUES (?, ?)",
                    (ride_id, passenger_id),
                )
            for passenger_id in paying_passenger_ids:
                conn.execute(
                    """
                    INSERT INTO ledger_entries (ride_id, driver_id, passenger_id, amount)
                    VALUES (?, ?, ?, ?)
                    """,
                    (ride_id, driver_id, passenger_id, cost_per_passenger),
                )
            conn.commit()
        return ride_id

    def fetch_rides_with_passengers(self) -> List[dict[str, Any]]:
        with self._connect() as conn:
            ride_rows = conn.execute(
                """
                SELECT r.*, d.name AS driver_name
                FROM rides r
                JOIN team_members d ON r.driver_id = d.id
                ORDER BY datetime(r.ride_datetime) DESC
                """
            ).fetchall()
            rides: List[dict[str, Any]] = []
            for row in ride_rows:
                passenger_rows = conn.execute(
                    """
                    SELECT tm.name, tm.is_core
                    FROM ride_passengers rp
                    JOIN team_members tm ON rp.passenger_id = tm.id
                    WHERE rp.ride_id = ?
                    ORDER BY tm.name COLLATE NOCASE
                    """,
                    (row["id"],),
                ).fetchall()
                rides.append(
                    {
                        "id": row["id"],
                        "start_address": row["start_address"],
                        "destination_address": row["destination_address"],
                        "distance_km": row["distance_km"],
                        "flat_fee": row["flat_fee"],
                        "fee_per_km": row["fee_per_km"],
                        "total_cost": row["total_cost"],
                        "cost_per_passenger": row["cost_per_passenger"],
                        "ride_datetime": row["ride_datetime"],
                        "passengers": [
                            f"{p['name']} ({'Core' if p['is_core'] else 'Reserve'})"
                            for p in passenger_rows
                        ],
                        "driver_name": row["driver_name"],
                    }
                )
        return rides

    def fetch_ledger_summary(self) -> List[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    tm_passenger.name AS passenger_name,
                    tm_driver.name AS driver_name,
                    SUM(le.amount) AS amount_owed
                FROM ledger_entries le
                JOIN team_members tm_driver ON le.driver_id = tm_driver.id
                JOIN team_members tm_passenger ON le.passenger_id = tm_passenger.id
                GROUP BY le.driver_id, le.passenger_id
                ORDER BY driver_name COLLATE NOCASE, passenger_name COLLATE NOCASE
                """
            ).fetchall()
        return [dict(row) for row in rows]


@dataclass
class TeamMember:
    """Lightweight helper used by widgets to track team member metadata."""

    member_id: int
    name: str
    is_core: bool


class AddressLineEdit(QLineEdit):
    """Line edit that wires into Google Maps autocomplete suggestions."""

    api_error = pyqtSignal(str)

    def __init__(
        self,
        maps_handler: GoogleMapsHandler,
        thread_pool: QThreadPool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.maps_handler = maps_handler
        self.thread_pool = thread_pool
        self.setPlaceholderText("Type an address...")
        self._last_query = ""

        self._timer = QTimer(self)
        self._timer.setInterval(450)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._fetch_suggestions)

        self.textChanged.connect(self._on_text_changed)

        self._model = QStringListModel(self)
        self._completer = QCompleter(self)
        self._completer.setModel(self._model)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.setCompleter(self._completer)

    def _on_text_changed(self, text: str) -> None:
        if len(text.strip()) < 3:
            self._model.setStringList([])
            return
        self._timer.start()

    def _fetch_suggestions(self) -> None:
        query = self.text().strip()
        if not query or query == self._last_query:
            return
        self._last_query = query
        worker = Worker(self.maps_handler.autocomplete, query)
        worker.signals.finished.connect(self._update_suggestions)
        worker.signals.error.connect(self.api_error.emit)
        self.thread_pool.start(worker)

    def _update_suggestions(self, suggestions: list[str]) -> None:
        self._model.setStringList(suggestions)


class TeamManagementTab(QWidget):
    """Widget responsible for CRUD operations on team members."""

    members_changed = pyqtSignal(list)

    def __init__(self, db_manager: DatabaseManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.db_manager = db_manager
        self._selected_member: Optional[TeamMember] = None

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Team member name")

        self.member_type_combo = QComboBox()
        self.member_type_combo.addItem("Core Member", True)
        self.member_type_combo.addItem("Reserve Player", False)
        self._set_type_combo(True)

        self.add_button = QPushButton("Save Team Member")
        self.update_button = QPushButton("Update Selected")
        self.delete_button = QPushButton("Delete Selected")

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Name", "Type"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)

        self._build_layout()
        self._wire_signals()
        self.refresh_members()

    def _build_layout(self) -> None:
        form_group = QGroupBox("Add or Edit Team Members")
        form_layout = QVBoxLayout(form_group)
        form_layout.setSpacing(14)

        name_row = QFormLayout()
        name_row.addRow("Name", self.name_input)
        name_row.addRow("Role", self.member_type_combo)
        form_layout.addLayout(name_row)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(12)
        buttons_layout.addWidget(self.add_button)
        buttons_layout.addWidget(self.update_button)
        buttons_layout.addWidget(self.delete_button)
        buttons_layout.addStretch(1)
        form_layout.addLayout(buttons_layout)

        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.addWidget(form_group)
        layout.addWidget(self.table)

    def _wire_signals(self) -> None:
        self.add_button.clicked.connect(self._on_add_member)
        self.update_button.clicked.connect(self._on_update_member)
        self.delete_button.clicked.connect(self._on_delete_member)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)

    def refresh_members(self) -> None:
        members = [
            TeamMember(member_id=row["id"], name=row["name"], is_core=bool(row["is_core"]))
            for row in self.db_manager.fetch_team_members()
        ]
        self.table.setRowCount(len(members))
        for row_index, member in enumerate(members):
            name_item = QTableWidgetItem(member.name)
            name_item.setData(Qt.ItemDataRole.UserRole, member.member_id)
            name_item.setData(Qt.ItemDataRole.UserRole + 1, member.is_core)
            name_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)

            type_label = "Core" if member.is_core else "Reserve"
            type_item = QTableWidgetItem(type_label)
            type_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)

            self.table.setItem(row_index, 0, name_item)
            self.table.setItem(row_index, 1, type_item)
        self.table.resizeRowsToContents()
        self.members_changed.emit(members)

    # Button handlers -----------------------------------------------------
    def _validate_name(self) -> Optional[str]:
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid Input", "Please enter a non-empty name.")
            return None
        return name

    def _on_add_member(self) -> None:
        name = self._validate_name()
        if not name:
            return
        try:
            self.db_manager.add_team_member(name, self._selected_type())
        except sqlite3.IntegrityError:
            QMessageBox.warning(
                self,
                "Duplicate Member",
                "A team member with that name already exists.",
            )
            return
        self.name_input.clear()
        self._set_type_combo(True)
        self._selected_member = None
        self.refresh_members()
        self.table.clearSelection()

    def _on_update_member(self) -> None:
        name = self._validate_name()
        if not name or self._selected_member is None:
            QMessageBox.information(
                self,
                "Select Member",
                "Choose a team member from the table to update their name.",
            )
            return
        try:
            self.db_manager.update_team_member(
                self._selected_member.member_id, name, self._selected_type()
            )
        except sqlite3.IntegrityError:
            QMessageBox.warning(
                self,
                "Duplicate Member",
                "A team member with that name already exists.",
            )
            return
        self.name_input.clear()
        self._set_type_combo(True)
        self._selected_member = None
        self.refresh_members()
        self.table.clearSelection()

    def _on_delete_member(self) -> None:
        if self._selected_member is None:
            QMessageBox.information(
                self,
                "Select Member",
                "Choose a team member from the table to delete.",
            )
            return
        confirm = QMessageBox.question(
            self,
            "Delete Member?",
            f"Are you sure you want to remove {self._selected_member.name}?",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            self.db_manager.delete_team_member(self._selected_member.member_id)
        except sqlite3.IntegrityError:
            QMessageBox.critical(
                self,
                "Cannot Delete",
                "This member is associated with rides and cannot be removed.",
            )
            return
        self._selected_member = None
        self.name_input.clear()
        self._set_type_combo(True)
        self.refresh_members()
        self.table.clearSelection()

    def _on_table_selection_changed(self) -> None:
        items = self.table.selectedItems()
        if not items:
            self._selected_member = None
            self._set_type_combo(True)
            return
        item = items[0]
        member_id = int(item.data(Qt.ItemDataRole.UserRole))
        name = item.text()
        is_core = bool(item.data(Qt.ItemDataRole.UserRole + 1))
        self._selected_member = TeamMember(member_id, name, is_core)
        self.name_input.setText(name)
        self._set_type_combo(is_core)

    def _selected_type(self) -> bool:
        data = self.member_type_combo.currentData(Qt.ItemDataRole.UserRole)
        if data is None:
            return True
        return bool(data)

    def _set_type_combo(self, is_core: bool) -> None:
        for index in range(self.member_type_combo.count()):
            if bool(self.member_type_combo.itemData(index, Qt.ItemDataRole.UserRole)) == is_core:
                self.member_type_combo.setCurrentIndex(index)
                break


class RideSetupTab(QWidget):
    """Configure and persist new rides, including cost calculations."""

    ride_saved = pyqtSignal()

    def __init__(
        self,
        db_manager: DatabaseManager,
        maps_handler: GoogleMapsHandler,
        thread_pool: QThreadPool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.db_manager = db_manager
        self.maps_handler = maps_handler
        self.thread_pool = thread_pool
        self.members: list[TeamMember] = []
        self.member_lookup: dict[int, TeamMember] = {}

        self.start_input = AddressLineEdit(self.maps_handler, self.thread_pool)
        self.dest_input = AddressLineEdit(self.maps_handler, self.thread_pool)
        self.start_input.api_error.connect(self._on_api_error)
        self.dest_input.api_error.connect(self._on_api_error)

        self.driver_combo = QComboBox()
        self.passenger_list = QListWidget()
        self.passenger_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)

        self.flat_fee_input = QDoubleSpinBox()
        self.flat_fee_input.setRange(0, 10000)
        self.flat_fee_input.setPrefix("€ ")
        self.flat_fee_input.setDecimals(2)
        self.flat_fee_input.setSingleStep(0.5)
        self.flat_fee_input.setValue(5.0)

        self.per_km_input = QDoubleSpinBox()
        self.per_km_input.setRange(0, 500)
        self.per_km_input.setPrefix("€ ")
        self.per_km_input.setSuffix(" / km")
        self.per_km_input.setDecimals(2)
        self.per_km_input.setSingleStep(0.1)
        self.per_km_input.setValue(0.5)

        self.calculate_button = QPushButton("Calculate Ride Cost")
        self.save_button = QPushButton("Save Ride")
        self.save_button.setEnabled(False)

        self.distance_value = QLabel("—")
        self.total_cost_value = QLabel("—")
        self.cost_per_passenger_value = QLabel("—")

        self._current_distance: Optional[float] = None
        self._current_total_cost: Optional[float] = None
        self._current_cost_per_passenger: Optional[float] = None
        self._current_paying_passenger_ids: list[int] = []
        self._current_all_passenger_ids: list[int] = []

        self._build_layout()
        self._wire_signals()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(24)

        addresses_group = QGroupBox("Addresses")
        addr_layout = QFormLayout(addresses_group)
        addr_layout.addRow("Start Address", self.start_input)
        addr_layout.addRow("Destination", self.dest_input)
        layout.addWidget(addresses_group)

        team_group = QGroupBox("Team Selection")
        team_layout = QFormLayout(team_group)
        team_layout.addRow("Driver", self.driver_combo)
        team_layout.addRow("Passengers", self.passenger_list)
        layout.addWidget(team_group)

        fees_group = QGroupBox("Fees")
        fees_layout = QFormLayout(fees_group)
        fees_layout.addRow("Flat Fee", self.flat_fee_input)
        fees_layout.addRow("Fee per km", self.per_km_input)
        layout.addWidget(fees_group)

        results_group = QGroupBox("Results")
        results_layout = QFormLayout(results_group)
        results_layout.addRow("Distance (km)", self.distance_value)
        results_layout.addRow("Total Cost", self.total_cost_value)
        results_layout.addRow("Cost per Passenger", self.cost_per_passenger_value)
        layout.addWidget(results_group)

        layout.addWidget(self.calculate_button)
        layout.addWidget(self.save_button)
        layout.addStretch(1)

    def _wire_signals(self) -> None:
        self.calculate_button.clicked.connect(self._on_calculate_clicked)
        self.save_button.clicked.connect(self._on_save_clicked)
        self.driver_combo.currentIndexChanged.connect(self._remove_driver_from_passengers)
        self.passenger_list.itemSelectionChanged.connect(self._invalidate_calculation)
        self.flat_fee_input.valueChanged.connect(self._invalidate_calculation)
        self.per_km_input.valueChanged.connect(self._invalidate_calculation)
        self.start_input.textChanged.connect(self._invalidate_calculation)
        self.dest_input.textChanged.connect(self._invalidate_calculation)

    # Public API ----------------------------------------------------------
    def set_team_members(self, members: list[TeamMember]) -> None:
        self.members = members
        self.member_lookup = {member.member_id: member for member in members}
        self.driver_combo.clear()
        for member in members:
            self.driver_combo.addItem(self._format_member(member), member.member_id)

        self.passenger_list.clear()
        for member in members:
            item = QListWidgetItem(self._format_member(member))
            item.setData(Qt.ItemDataRole.UserRole, member.member_id)
            self.passenger_list.addItem(item)
        self._remove_driver_from_passengers()
        self._invalidate_calculation()

    # Helper properties ---------------------------------------------------
    def _selected_driver_id(self) -> Optional[int]:
        index = self.driver_combo.currentIndex()
        if index < 0:
            return None
        return int(self.driver_combo.currentData(Qt.ItemDataRole.UserRole))

    def _selected_passenger_ids(self) -> List[int]:
        return [
            int(item.data(Qt.ItemDataRole.UserRole)) for item in self.passenger_list.selectedItems()
        ]

    def _format_member(self, member: TeamMember) -> str:
        role = "Core" if member.is_core else "Reserve"
        return f"{member.name} ({role})"

    def _is_core_member(self, member_id: int) -> bool:
        member = self.member_lookup.get(member_id)
        return bool(member and member.is_core)

    def _remove_driver_from_passengers(self) -> None:
        driver_id = self._selected_driver_id()
        if driver_id is None:
            return
        for index in range(self.passenger_list.count()):
            item = self.passenger_list.item(index)
            if int(item.data(Qt.ItemDataRole.UserRole)) == driver_id:
                item.setSelected(False)
        self._invalidate_calculation()

    # Event handlers ------------------------------------------------------
    def _on_api_error(self, message: str) -> None:
        QMessageBox.critical(self, "Google Maps Error", message)

    def _on_calculate_clicked(self) -> None:
        start_address = self.start_input.text().strip()
        destination_address = self.dest_input.text().strip()
        driver_id = self._selected_driver_id()
        passenger_ids = self._selected_passenger_ids()
        core_passenger_ids = [pid for pid in passenger_ids if self._is_core_member(pid)]
        flat_fee = float(self.flat_fee_input.value())
        per_km_fee = float(self.per_km_input.value())

        if not start_address or not destination_address:
            QMessageBox.warning(self, "Missing Addresses", "Please enter both addresses.")
            return
        if driver_id is None:
            QMessageBox.warning(self, "Driver Missing", "Please select a driver.")
            return
        if not passenger_ids:
            QMessageBox.warning(self, "Passengers Missing", "Select at least one passenger.")
            return
        if driver_id in passenger_ids:
            QMessageBox.warning(
                self,
                "Invalid Selection",
                "The driver cannot also be listed as a passenger.",
            )
            return
        if flat_fee <= 0 or per_km_fee <= 0:
            QMessageBox.warning(
                self,
                "Invalid Fees",
                "Both the flat fee and per-kilometre fee must be positive values.",
            )
            return
        if not core_passenger_ids:
            QMessageBox.warning(
                self,
                "No Core Members",
                "Select at least one core team member as a passenger to share the ride cost.",
            )
            return

        self.calculate_button.setEnabled(False)
        worker = Worker(self.maps_handler.distance_km, start_address, destination_address)
        worker.signals.finished.connect(
            lambda distance: self._on_distance_ready(
                distance,
                flat_fee,
                per_km_fee,
                passenger_ids,
                core_passenger_ids,
            )
        )
        worker.signals.error.connect(self._on_calculate_error)
        self.thread_pool.start(worker)

    def _on_distance_ready(
        self,
        distance_km: float,
        flat_fee: float,
        per_km_fee: float,
        passenger_ids: list[int],
        core_passenger_ids: list[int],
    ) -> None:
        self.calculate_button.setEnabled(True)
        round_trip_distance = round(distance_km * 2, 2)
        self._current_distance = round_trip_distance
        total_cost = round(flat_fee + round_trip_distance * per_km_fee, 2)
        cost_per_passenger = round(total_cost / len(core_passenger_ids), 2)
        self._current_total_cost = total_cost
        self._current_cost_per_passenger = cost_per_passenger
        self._current_paying_passenger_ids = core_passenger_ids
        self._current_all_passenger_ids = passenger_ids

        self.distance_value.setText(f"{round_trip_distance:.2f} km")
        self.total_cost_value.setText(f"€{total_cost:.2f}")
        self.cost_per_passenger_value.setText(
            f"€{cost_per_passenger:.2f} per core member ({len(core_passenger_ids)} total)"
        )
        self.save_button.setEnabled(True)

    def _on_calculate_error(self, message: str) -> None:
        self.calculate_button.setEnabled(True)
        QMessageBox.critical(self, "Calculation Error", message)

    def _on_save_clicked(self) -> None:
        if self._current_distance is None or self._current_total_cost is None:
            QMessageBox.information(
                self,
                "Calculate First",
                "Please calculate the ride cost before saving.",
            )
            return
        start_address = self.start_input.text().strip()
        destination_address = self.dest_input.text().strip()
        driver_id = self._selected_driver_id()
        passenger_ids = self._selected_passenger_ids()
        core_passenger_ids = [pid for pid in passenger_ids if self._is_core_member(pid)]

        if driver_id is None or not passenger_ids:
            QMessageBox.warning(
                self,
                "Invalid Selection",
                "Please ensure a driver and at least one passenger are selected.",
            )
            return
        if not core_passenger_ids:
            QMessageBox.warning(
                self,
                "No Core Members",
                "Select at least one core team member as a passenger to share the ride cost.",
            )
            return

        flat_fee = float(self.flat_fee_input.value())
        per_km_fee = float(self.per_km_input.value())

        total_cost = self._current_total_cost
        if total_cost is None:
            QMessageBox.information(
                self,
                "Calculate First",
                "Please calculate the ride cost before saving.",
            )
            return

        cost_per_passenger = round(total_cost / len(core_passenger_ids), 2)
        self._current_cost_per_passenger = cost_per_passenger

        try:
            self.db_manager.record_ride(
                start_address=start_address,
                destination_address=destination_address,
                distance_km=self._current_distance,
                driver_id=driver_id,
                passenger_ids=passenger_ids,
                paying_passenger_ids=core_passenger_ids,
                flat_fee=flat_fee,
                fee_per_km=per_km_fee,
                total_cost=total_cost,
                cost_per_passenger=cost_per_passenger,
            )
        except sqlite3.DatabaseError as exc:
            QMessageBox.critical(
                self,
                "Database Error",
                f"Failed to save ride: {exc}",
            )
            return

        QMessageBox.information(self, "Ride Saved", "The ride has been stored successfully.")
        self._reset_form()
        self.ride_saved.emit()

    def _reset_form(self) -> None:
        self.start_input.clear()
        self.dest_input.clear()
        self.passenger_list.clearSelection()
        self.flat_fee_input.setValue(5.0)
        self.per_km_input.setValue(0.5)
        self.distance_value.setText("—")
        self.total_cost_value.setText("—")
        self.cost_per_passenger_value.setText("—")
        self.save_button.setEnabled(False)
        self._current_distance = None
        self._current_total_cost = None
        self._current_cost_per_passenger = None
        self._current_paying_passenger_ids = []
        self._current_all_passenger_ids = []

    def _invalidate_calculation(self) -> None:
        self.save_button.setEnabled(False)
        self._current_distance = None
        self._current_total_cost = None
        self._current_cost_per_passenger = None
        self._current_paying_passenger_ids = []
        self._current_all_passenger_ids = []
        self.distance_value.setText("—")
        self.total_cost_value.setText("—")
        self.cost_per_passenger_value.setText("—")


class RideHistoryTab(QWidget):
    """Display past rides and current ledger balances."""

    def __init__(self, db_manager: DatabaseManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.db_manager = db_manager

        self.rides_table = QTableWidget(0, 9)
        self.rides_table.setHorizontalHeaderLabels(
            [
                "Date",
                "Driver",
                "Passengers",
                "From",
                "To",
                "Distance (km)",
                "Flat Fee",
                "Fee/km",
                "Total",
            ]
        )
        self.rides_table.verticalHeader().setVisible(False)
        self.rides_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.rides_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        header = self.rides_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        self.ledger_table = QTableWidget(0, 3)
        self.ledger_table.setHorizontalHeaderLabels(["Passenger", "Driver", "Amount Owed"])
        self.ledger_table.verticalHeader().setVisible(False)
        self.ledger_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.ledger_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        ledger_header = self.ledger_table.horizontalHeader()
        ledger_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        layout = QVBoxLayout(self)
        layout.setSpacing(24)
        layout.addWidget(QLabel("Ride History"))
        layout.addWidget(self.rides_table)
        layout.addWidget(QLabel("Ledger"))
        layout.addWidget(self.ledger_table)
        layout.addStretch(1)

    def refresh(self) -> None:
        rides = self.db_manager.fetch_rides_with_passengers()
        self.rides_table.setRowCount(len(rides))
        for row_idx, ride in enumerate(rides):
            self._set_table_item(
                self.rides_table, row_idx, 0, self._format_datetime(ride["ride_datetime"])
            )
            self._set_table_item(self.rides_table, row_idx, 1, ride["driver_name"])
            self._set_table_item(self.rides_table, row_idx, 2, ", ".join(ride["passengers"]))
            self._set_table_item(self.rides_table, row_idx, 3, ride["start_address"])
            self._set_table_item(self.rides_table, row_idx, 4, ride["destination_address"])
            self._set_table_item(self.rides_table, row_idx, 5, f"{ride['distance_km']:.2f}")
            self._set_table_item(self.rides_table, row_idx, 6, f"€{ride['flat_fee']:.2f}")
            self._set_table_item(self.rides_table, row_idx, 7, f"€{ride['fee_per_km']:.2f}")
            self._set_table_item(self.rides_table, row_idx, 8, f"€{ride['total_cost']:.2f}")
        self.rides_table.resizeRowsToContents()

        ledger_entries = self.db_manager.fetch_ledger_summary()
        self.ledger_table.setRowCount(len(ledger_entries))
        for row_idx, entry in enumerate(ledger_entries):
            self._set_table_item(self.ledger_table, row_idx, 0, entry["passenger_name"])
            self._set_table_item(self.ledger_table, row_idx, 1, entry["driver_name"])
            self._set_table_item(self.ledger_table, row_idx, 2, f"€{entry['amount_owed']:.2f}")
        self.ledger_table.resizeRowsToContents()

    @staticmethod
    def _set_table_item(table: QTableWidget, row: int, column: int, text: str) -> None:
        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        table.setItem(row, column, item)

    @staticmethod
    def _format_datetime(timestamp: str) -> str:
        try:
            dt = datetime.fromisoformat(timestamp)
        except ValueError:
            return timestamp
        return dt.strftime("%Y-%m-%d %H:%M")


class RideShareApp(QMainWindow):
    """Main window that orchestrates the individual tabs."""

    def __init__(self, db_manager: DatabaseManager, maps_handler: GoogleMapsHandler) -> None:
        super().__init__()
        self.db_manager = db_manager
        self.maps_handler = maps_handler
        self.thread_pool = QThreadPool()

        self.setWindowTitle("Table Tennis RideShare Manager")
        self.resize(1100, 740)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.team_tab = TeamManagementTab(self.db_manager)
        self.ride_tab = RideSetupTab(self.db_manager, self.maps_handler, self.thread_pool)
        self.history_tab = RideHistoryTab(self.db_manager)

        self.tabs.addTab(self.team_tab, "Team Management")
        self.tabs.addTab(self.ride_tab, "Ride Setup")
        self.tabs.addTab(self.history_tab, "Ride History & Ledger")

        self.team_tab.members_changed.connect(self._on_members_changed)
        self.ride_tab.ride_saved.connect(self.history_tab.refresh)

        self._initialise_state()

    def _initialise_state(self) -> None:
        members = [
            TeamMember(
                member_id=row["id"],
                name=row["name"],
                is_core=bool(row["is_core"]),
            )
            for row in self.db_manager.fetch_team_members()
        ]
        self.ride_tab.set_team_members(members)
        self.history_tab.refresh()

    def _on_members_changed(self, members: list[TeamMember]) -> None:
        self.ride_tab.set_team_members(members)


def load_stylesheet() -> str:
    if STYLE_FILE.exists():
        return STYLE_FILE.read_text(encoding="utf-8")
    return ""


def bootstrap_app() -> int:
    """Configure the QApplication and start the GUI loop.

    Returns the exit code produced by ``QApplication.exec``.
    Raises ``GoogleMapsError`` if configuration is invalid before the GUI starts.
    """

    load_dotenv()
    api_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    if not api_key:
        raise GoogleMapsError(
            "Google Maps API key missing. Set the GOOGLE_MAPS_API_KEY environment variable."
        )

    db_manager = DatabaseManager(DATABASE_FILE)
    maps_handler = GoogleMapsHandler(api_key)

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    stylesheet = load_stylesheet()
    if stylesheet:
        app.setStyleSheet(stylesheet)

    window = RideShareApp(db_manager, maps_handler)
    window.show()
    return app.exec()


if __name__ == "__main__":
    try:
        sys.exit(bootstrap_app())
    except GoogleMapsError as exc:
        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "Google Maps Configuration", str(exc))
        sys.exit(1)
