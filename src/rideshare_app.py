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

import json
import os
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

import googlemaps
from dotenv import load_dotenv
from googlemaps.exceptions import ApiError, TransportError
from PyQt6.QtCore import (
    QObject,
    QRunnable,
    QThreadPool,
    QTimer,
    Qt,
    pyqtSignal,
    QStringListModel,
    QEvent,
    QPoint,
)
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QFileDialog,
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
    QSizeGrip,
    QSizePolicy,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .utils.pdf_exporter import export_ledger_pdf


APP_BUNDLE_ROOT = Path(__file__).resolve().parent
STYLE_FILE = APP_BUNDLE_ROOT / "resources" / "style.qss"
_BUNDLED_DATABASE_FILE = APP_BUNDLE_ROOT / "rideshare.db"
_BUNDLED_SETTINGS_FILE = APP_BUNDLE_ROOT / "config" / "settings.json"


def _resolve_data_directory() -> Path:
    if sys.platform == "win32":
        base = Path(os.getenv("APPDATA", Path.home() / "AppData/Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library/Application Support"
    else:
        base = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local/share"))
    data_dir = base / "TableTennisRideShare"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


APP_DATA_DIR = _resolve_data_directory()
DATABASE_FILE = APP_DATA_DIR / "rideshare.db"
SETTINGS_FILE = APP_DATA_DIR / "settings.json"


def _prime_persistent_files() -> None:
    if not SETTINGS_FILE.exists():
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        if _BUNDLED_SETTINGS_FILE.exists():
            shutil.copy2(_BUNDLED_SETTINGS_FILE, SETTINGS_FILE)
    if not DATABASE_FILE.exists():
        DATABASE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if _BUNDLED_DATABASE_FILE.exists():
            shutil.copy2(_BUNDLED_DATABASE_FILE, DATABASE_FILE)
        else:
            with sqlite3.connect(DATABASE_FILE) as connection:
                connection.execute("PRAGMA journal_mode = WAL")


_prime_persistent_files()


def split_amount_evenly(amount: float, parts: int) -> list[float]:
    """Split *amount* into *parts* nearly equal two-decimal floats.

    The result sums to the original amount (rounded to cents).
    """

    if parts <= 0:
        raise ValueError("parts must be a positive integer")

    cents = int(round(amount * 100))
    base = cents // parts
    remainder = cents % parts

    shares = []
    for index in range(parts):
        share_cents = base + (1 if index < remainder else 0)
        shares.append(round(share_cents / 100.0, 2))
    return shares


class GoogleMapsError(RuntimeError):
    """Domain-specific error raised for Google Maps integration problems."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


class SettingsManager:
    """Load and persist lightweight JSON application settings."""

    DEFAULTS: dict[str, Any] = {
        "default_home_address": "Franckestraße 15, Leipzig-Ost, Germany",
        "default_flat_fee": 5.0,
        "default_fee_per_km": 0.5,
        "window_size": {"width": 1100, "height": 740},
        "team_table_column_widths": [420, 140],
    }

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = json.loads(json.dumps(self.DEFAULTS))  # deep copy
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.save()
            return
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self.save()
            return
        if isinstance(loaded, dict):
            self.data = _deep_merge(self.data, loaded)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def update(self, updates: dict[str, Any]) -> None:
        self.data = _deep_merge(self.data, updates)
        self.save()


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
        self.enabled = bool(api_key)
        self.client = None
        if not self.enabled:
            return
        try:
            self.client = googlemaps.Client(key=api_key)
        except (ApiError, TransportError) as exc:  # pragma: no cover - network issues
            raise GoogleMapsError(f"Unable to initialise Google Maps client: {exc}") from exc

    def autocomplete(self, query: str) -> List[str]:
        """Return address suggestions for the provided query string."""
        if not self.enabled or self.client is None:
            return []
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
        if not self.enabled:
            raise GoogleMapsError(
                "Google Maps API key is not configured. Distance lookup is disabled."
            )
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

        CREATE TABLE IF NOT EXISTS ride_drivers (
            ride_id INTEGER NOT NULL,
            driver_id INTEGER NOT NULL,
            PRIMARY KEY (ride_id, driver_id),
            FOREIGN KEY(ride_id) REFERENCES rides(id) ON DELETE CASCADE,
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

            conn.execute(
                """
                INSERT OR IGNORE INTO ride_drivers (ride_id, driver_id)
                SELECT id, driver_id FROM rides
                """
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
        driver_ids: Iterable[int],
        passenger_ids: Iterable[int],
        paying_passenger_ids: Iterable[int],
        flat_fee: float,
        fee_per_km: float,
        total_cost: float,
        cost_per_passenger: float,
    ) -> int:
        driver_ids = [int(driver_id) for driver_id in driver_ids]
        if not driver_ids:
            raise ValueError("At least one driver must be supplied for a ride.")

        passenger_ids = [int(pid) for pid in passenger_ids]
        paying_passenger_ids = [int(pid) for pid in paying_passenger_ids]
        timestamp = datetime.utcnow().isoformat(timespec="seconds")
        primary_driver_id = driver_ids[0]
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
                    primary_driver_id,
                    flat_fee,
                    fee_per_km,
                    total_cost,
                    cost_per_passenger,
                    timestamp,
                ),
            )
            ride_id = int(cursor.lastrowid)

            for driver_id in driver_ids:
                conn.execute(
                    "INSERT INTO ride_drivers (ride_id, driver_id) VALUES (?, ?)",
                    (ride_id, driver_id),
                )
            for passenger_id in passenger_ids:
                conn.execute(
                    "INSERT INTO ride_passengers (ride_id, passenger_id) VALUES (?, ?)",
                    (ride_id, passenger_id),
                )
            per_driver_amounts = split_amount_evenly(cost_per_passenger, len(driver_ids))
            for passenger_id in paying_passenger_ids:
                for index, driver_id in enumerate(driver_ids):
                    conn.execute(
                        """
                        INSERT INTO ledger_entries (ride_id, driver_id, passenger_id, amount)
                        VALUES (?, ?, ?, ?)
                        """,
                        (ride_id, driver_id, passenger_id, per_driver_amounts[index]),
                    )
            conn.commit()
        return ride_id

    def delete_ride(self, ride_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM rides WHERE id = ?", (ride_id,))
            conn.commit()

    def fetch_rides_with_passengers(self, limit: Optional[int] = None) -> List[dict[str, Any]]:
        with self._connect() as conn:
            query = """
                SELECT r.id,
                       r.start_address,
                       r.destination_address,
                       r.distance_km,
                       r.flat_fee,
                       r.fee_per_km,
                       r.total_cost,
                       r.cost_per_passenger,
                       r.ride_datetime
                FROM rides r
                ORDER BY datetime(r.ride_datetime) DESC
            """
            params: tuple[Any, ...] = ()
            if limit is not None:
                query += " LIMIT ?"
                params = (limit,)
            ride_rows = conn.execute(query, params).fetchall()
            rides: List[dict[str, Any]] = []
            for row in ride_rows:
                driver_rows = conn.execute(
                    """
                    SELECT tm.name, tm.is_core
                    FROM ride_drivers rd
                    JOIN team_members tm ON rd.driver_id = tm.id
                    WHERE rd.ride_id = ?
                    ORDER BY tm.name COLLATE NOCASE
                    """,
                    (row["id"],),
                ).fetchall()
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
                        "drivers": [
                            f"{d['name']} ({'Core' if d['is_core'] else 'Reserve'})"
                            for d in driver_rows
                        ],
                        "passengers": [
                            f"{p['name']} ({'Core' if p['is_core'] else 'Reserve'})"
                            for p in passenger_rows
                        ],
                    }
                )
        return rides

    def fetch_recent_addresses(self, limit: int = 10) -> dict[str, List[str]]:
        with self._connect() as conn:
            start_rows = conn.execute(
                "SELECT start_address, ride_datetime FROM rides ORDER BY datetime(ride_datetime) DESC"
            ).fetchall()
            dest_rows = conn.execute(
                "SELECT destination_address, ride_datetime FROM rides ORDER BY datetime(ride_datetime) DESC"
            ).fetchall()

        def _dedupe(rows: Iterable[sqlite3.Row]) -> List[str]:
            seen: List[str] = []
            for row in rows:
                address = str(row[0])
                if address and address not in seen:
                    seen.append(address)
                if len(seen) >= limit:
                    break
            return seen

        return {
            "start": _dedupe(start_rows),
            "destination": _dedupe(dest_rows),
        }

    def fetch_ledger_summary(self) -> List[dict[str, Any]]:
        with self._connect() as conn:
            balances = conn.execute(
                """
                SELECT driver_id, passenger_id, SUM(amount) AS total_amount
                FROM ledger_entries
                GROUP BY driver_id, passenger_id
                """
            ).fetchall()
            members = conn.execute("SELECT id, name FROM team_members").fetchall()

        name_lookup = {row["id"]: row["name"] for row in members}
        raw_balances: dict[tuple[int, int], float] = {}
        for row in balances:
            key = (int(row["passenger_id"]), int(row["driver_id"]))
            raw_balances[key] = round(float(row["total_amount"] or 0.0), 2)

        processed: set[tuple[int, int]] = set()
        results: list[dict[str, Any]] = []
        for (passenger_id, driver_id), amount in raw_balances.items():
            if (passenger_id, driver_id) in processed:
                continue

            opposite_amount = raw_balances.get((driver_id, passenger_id), 0.0)
            net = round(amount - opposite_amount, 2)

            processed.add((passenger_id, driver_id))
            processed.add((driver_id, passenger_id))

            if abs(net) < 0.01:
                continue

            if net > 0:
                owes_id, owed_id, value = passenger_id, driver_id, net
            else:
                owes_id, owed_id, value = driver_id, passenger_id, abs(net)

            results.append(
                {
                    "owes_name": name_lookup.get(owes_id, "Unknown"),
                    "owed_name": name_lookup.get(owed_id, "Unknown"),
                    "amount": round(value, 2),
                }
            )

        results.sort(key=lambda item: (-item["amount"], item["owes_name"], item["owed_name"]))
        return results


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
        popup = self._completer.popup()
        popup.setStyleSheet(
            "QListView { background-color: #1a2739; color: #edf2fb; } "
            "QListView::item:selected { background-color: #35c4c7; color: #ffffff; }"
        )

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
        self._column_widths: Optional[list[int]] = None

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Team member name")

        self.member_type_combo = QComboBox()
        self.member_type_combo.addItem("Core Member", True)
        self.member_type_combo.addItem("Reserve Player", False)
        self.member_type_combo.setStyleSheet(
            """
            QComboBox {
                background-color: #1f2a36;
                color: #f4f6fa;
                border: 1px solid #31445a;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QComboBox::drop-down {
                background-color: #2b3b4d;
                border-left: 1px solid #31445a;
                width: 22px;
            }
            QComboBox::down-arrow {
                image: none;
            }
            QComboBox QAbstractItemView {
                background-color: #141c27;
                color: #f4f6fa;
                selection-background-color: #35c4c7;
                selection-color: #0b1118;
            }
            """
        )
        self._set_type_combo(True)

        self.add_button = QPushButton("Save Team Member")
        self.update_button = QPushButton("Update Selected")
        self.delete_button = QPushButton("Delete Selected")

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Name", "Type"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.setMinimumSectionSize(120)
        header.sectionResized.connect(self._on_column_resized)
        self.table.setColumnWidth(1, 160)

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
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(form_group)
        layout.addWidget(self.table, 1)

    def _wire_signals(self) -> None:
        self.add_button.clicked.connect(self._on_add_member)
        self.update_button.clicked.connect(self._on_update_member)
        self.delete_button.clicked.connect(self._on_delete_member)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)

    def apply_settings(self, settings: dict[str, Any]) -> None:
        widths = settings.get("team_table_column_widths")
        if isinstance(widths, list) and widths:
            coerced: list[int] = []
            for value in widths:
                try:
                    sanitized = max(80, int(value))
                except (TypeError, ValueError):
                    return
                index = len(coerced)
                if index == 1:
                    sanitized = min(sanitized, 360)
                coerced.append(sanitized)
            if len(coerced) > self.table.columnCount():
                coerced = coerced[: self.table.columnCount()]
            self._column_widths = coerced
            self._restore_column_widths()

    def get_column_widths(self) -> list[int]:
        return [self.table.columnWidth(i) for i in range(self.table.columnCount())]

    def _restore_column_widths(self) -> None:
        if not self._column_widths:
            return
        header = self.table.horizontalHeader()
        header.blockSignals(True)
        for index, width in enumerate(self._column_widths):
            if index == 0:
                continue
            if index < self.table.columnCount():
                self.table.setColumnWidth(index, width)
        header.blockSignals(False)

    def _on_column_resized(self, index: int, _old: int, new: int) -> None:
        if index == 0:
            return
        widths = self._column_widths or [
            self.table.columnWidth(i) for i in range(self.table.columnCount())
        ]
        if index >= len(widths):
            widths = [self.table.columnWidth(i) for i in range(self.table.columnCount())]
        widths[index] = max(80, new)
        if index == 1:
            widths[index] = min(widths[index], 360)
        self._column_widths = widths

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

            accent_color = QColor("#35c4c7") if member.is_core else QColor("#b3bed4")
            background_color = QColor("#103b43") if member.is_core else QColor("#232f44")
            name_item.setForeground(accent_color)
            type_item.setForeground(accent_color)
            name_item.setBackground(background_color)
            type_item.setBackground(background_color)

            self.table.setItem(row_index, 0, name_item)
            self.table.setItem(row_index, 1, type_item)
        self._restore_column_widths()
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
        self._default_home_address = ""
        self._default_flat_fee = 5.0
        self._default_fee_per_km = 0.5
        self._recent_start_addresses: list[str] = []
        self._recent_destination_addresses: list[str] = []

        self.start_input = AddressLineEdit(self.maps_handler, self.thread_pool)
        self.dest_input = AddressLineEdit(self.maps_handler, self.thread_pool)
        self.start_input.api_error.connect(self._on_api_error)
        self.dest_input.api_error.connect(self._on_api_error)

        self.start_history_combo = QComboBox()
        self.start_history_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.dest_history_combo = QComboBox()
        self.dest_history_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._style_history_combo(self.start_history_combo)
        self._style_history_combo(self.dest_history_combo)

        self.driver_list = QListWidget()
        self.driver_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
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
        self.per_km_input.setSingleStep(0.05)
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
        self._current_driver_ids: list[int] = []

        self._build_layout()
        self._wire_signals()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(24)

        addresses_group = QGroupBox("Addresses")
        addr_layout = QFormLayout(addresses_group)

        start_row_widget = QWidget()
        start_row_layout = QHBoxLayout(start_row_widget)
        start_row_layout.setContentsMargins(0, 0, 0, 0)
        start_row_layout.setSpacing(8)
        start_row_layout.addWidget(self.start_input)
        self.start_history_combo.setMinimumWidth(200)
        self.start_history_combo.setToolTip("Select a start address from previous rides")
        start_row_layout.addWidget(self.start_history_combo)
        addr_layout.addRow("Start Address", start_row_widget)

        dest_row_widget = QWidget()
        dest_row_layout = QHBoxLayout(dest_row_widget)
        dest_row_layout.setContentsMargins(0, 0, 0, 0)
        dest_row_layout.setSpacing(8)
        dest_row_layout.addWidget(self.dest_input)
        self.dest_history_combo.setMinimumWidth(200)
        self.dest_history_combo.setToolTip("Select a destination from previous rides")
        dest_row_layout.addWidget(self.dest_history_combo)
        addr_layout.addRow("Destination", dest_row_widget)
        layout.addWidget(addresses_group)

        team_group = QGroupBox("Team Selection")
        team_layout = QFormLayout(team_group)
        team_layout.addRow("Drivers", self.driver_list)
        team_layout.addRow("Passengers", self.passenger_list)
        layout.addWidget(team_group)

        fees_group = QGroupBox("Fees")
        fees_layout = QFormLayout(fees_group)
        fees_layout.addRow("Flat Fee (per Driver)", self.flat_fee_input)
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
        self.driver_list.itemSelectionChanged.connect(self._on_driver_selection_changed)
        self.passenger_list.itemSelectionChanged.connect(self._invalidate_calculation)
        self.flat_fee_input.valueChanged.connect(self._invalidate_calculation)
        self.per_km_input.valueChanged.connect(self._invalidate_calculation)
        self.start_input.textChanged.connect(self._invalidate_calculation)
        self.dest_input.textChanged.connect(self._invalidate_calculation)
        self.start_history_combo.currentIndexChanged.connect(self._on_start_history_selected)
        self.dest_history_combo.currentIndexChanged.connect(self._on_dest_history_selected)

    # Public API ----------------------------------------------------------
    def set_team_members(self, members: list[TeamMember]) -> None:
        self.members = members
        self.member_lookup = {member.member_id: member for member in members}
        self.driver_list.clear()
        self.passenger_list.clear()
        for member in members:
            driver_item = QListWidgetItem(self._format_member(member))
            driver_item.setData(Qt.ItemDataRole.UserRole, member.member_id)
            self.driver_list.addItem(driver_item)

            passenger_item = QListWidgetItem(self._format_member(member))
            passenger_item.setData(Qt.ItemDataRole.UserRole, member.member_id)
            self.passenger_list.addItem(passenger_item)
        self.driver_list.clearSelection()
        self.passenger_list.clearSelection()
        self._sync_driver_passenger_selection()
        self._invalidate_calculation()

    def apply_settings(self, settings: dict[str, Any]) -> None:
        self._default_home_address = str(
            settings.get("default_home_address", self._default_home_address)
        )
        self._default_flat_fee = float(settings.get("default_flat_fee", self._default_flat_fee))
        self._default_fee_per_km = float(
            settings.get("default_fee_per_km", self._default_fee_per_km)
        )

        self.flat_fee_input.setValue(self._default_flat_fee)
        self.per_km_input.setValue(self._default_fee_per_km)
        if self._default_home_address:
            self.start_input.setText(self._default_home_address)
        else:
            self.start_input.clear()
        self.dest_input.clear()

    def set_recent_addresses(
        self, start_addresses: list[str], destination_addresses: list[str]
    ) -> None:
        self._recent_start_addresses = start_addresses
        self._recent_destination_addresses = destination_addresses
        self._populate_history_combo(self.start_history_combo, start_addresses)
        self._populate_history_combo(self.dest_history_combo, destination_addresses)

    def _populate_history_combo(self, combo: QComboBox, addresses: list[str]) -> None:
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Select previous address…", None)
        for address in addresses:
            combo.addItem(address, address)
        combo.setCurrentIndex(0)
        combo.setEnabled(bool(addresses))
        combo.blockSignals(False)

    def _style_history_combo(self, combo: QComboBox) -> None:
        combo.setStyleSheet(
            """
            QComboBox {
                background-color: #1f2a36;
                color: #f4f6fa;
                border: 1px solid #31445a;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QComboBox::drop-down {
                background-color: #2b3b4d;
                border-left: 1px solid #31445a;
                width: 22px;
            }
            QComboBox::down-arrow {
                image: none;
            }
            """
        )
        view = combo.view()
        if view is not None:
            view.setStyleSheet(
                """
                QAbstractItemView {
                    background-color: #141c27;
                    color: #f4f6fa;
                    selection-background-color: #35c4c7;
                    selection-color: #0b1118;
                    padding: 4px;
                }
                """
            )
        combo.setMaxVisibleItems(8)
        combo.setMinimumContentsLength(1)

    def _on_start_history_selected(self, index: int) -> None:
        address = self.start_history_combo.itemData(index)
        if address:
            self.start_input.setText(str(address))
            self._invalidate_calculation()
        self._reset_history_selection(self.start_history_combo)

    def _on_dest_history_selected(self, index: int) -> None:
        address = self.dest_history_combo.itemData(index)
        if address:
            self.dest_input.setText(str(address))
            self._invalidate_calculation()
        self._reset_history_selection(self.dest_history_combo)

    @staticmethod
    def _reset_history_selection(combo: QComboBox) -> None:
        combo.blockSignals(True)
        combo.setCurrentIndex(0)
        combo.blockSignals(False)

    # Helper properties ---------------------------------------------------
    def _selected_driver_ids(self) -> List[int]:
        return [
            int(item.data(Qt.ItemDataRole.UserRole)) for item in self.driver_list.selectedItems()
        ]

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

    def _sync_driver_passenger_selection(self) -> None:
        driver_ids = {
            int(item.data(Qt.ItemDataRole.UserRole)) for item in self.driver_list.selectedItems()
        }
        if not driver_ids:
            return
        for index in range(self.passenger_list.count()):
            item = self.passenger_list.item(index)
            if int(item.data(Qt.ItemDataRole.UserRole)) in driver_ids:
                item.setSelected(False)

    def _on_driver_selection_changed(self) -> None:
        self._sync_driver_passenger_selection()
        self._invalidate_calculation()

    # Event handlers ------------------------------------------------------
    def _on_api_error(self, message: str) -> None:
        QMessageBox.critical(self, "Google Maps Error", message)

    def _on_calculate_clicked(self) -> None:
        start_address = self.start_input.text().strip()
        destination_address = self.dest_input.text().strip()
        driver_ids = self._selected_driver_ids()
        passenger_ids = self._selected_passenger_ids()
        core_passenger_ids = [
            pid for pid in passenger_ids if self._is_core_member(pid) and pid not in driver_ids
        ]
        flat_fee = float(self.flat_fee_input.value())
        per_km_fee = float(self.per_km_input.value())

        if not start_address or not destination_address:
            QMessageBox.warning(self, "Missing Addresses", "Please enter both addresses.")
            return
        if not driver_ids:
            QMessageBox.warning(self, "Driver Missing", "Please select at least one driver.")
            return
        if not passenger_ids:
            QMessageBox.warning(self, "Passengers Missing", "Select at least one passenger.")
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
            lambda distance, driver_ids=driver_ids, passenger_ids=passenger_ids, core_passenger_ids=core_passenger_ids: self._on_distance_ready(
                distance,
                flat_fee,
                per_km_fee,
                passenger_ids,
                core_passenger_ids,
                driver_ids,
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
        driver_ids: list[int],
    ) -> None:
        self.calculate_button.setEnabled(True)
        round_trip_distance = round(distance_km * 2, 2)
        driver_count = max(len(driver_ids), 1)
        driver_flat_fee_total = round(flat_fee * driver_count, 2)
        distance_cost = round(round_trip_distance * per_km_fee, 2)
        self._current_distance = round_trip_distance
        total_cost = round(driver_flat_fee_total + distance_cost, 2)
        cost_per_passenger = round(total_cost / len(core_passenger_ids), 2)
        self._current_total_cost = total_cost
        self._current_cost_per_passenger = cost_per_passenger
        self._current_paying_passenger_ids = core_passenger_ids
        self._current_all_passenger_ids = passenger_ids
        self._current_driver_ids = driver_ids

        self.distance_value.setText(f"{round_trip_distance:.2f} km")
        driver_label = "driver" if driver_count == 1 else "drivers"
        self.total_cost_value.setText(
            f"€{total_cost:.2f} (flat: €{flat_fee:.2f} × {driver_count}, distance: €{distance_cost:.2f})"
        )
        core_count = len(core_passenger_ids)
        self.cost_per_passenger_value.setText(
            f"€{cost_per_passenger:.2f} per core member ({core_count} total, {driver_count} {driver_label})"
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
        driver_ids = self._selected_driver_ids()
        passenger_ids = self._selected_passenger_ids()
        core_passenger_ids = [
            pid for pid in passenger_ids if self._is_core_member(pid) and pid not in driver_ids
        ]

        if not driver_ids or not passenger_ids:
            QMessageBox.warning(
                self,
                "Invalid Selection",
                "Please ensure at least one driver and one passenger are selected.",
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
                driver_ids=driver_ids,
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

        self._reset_form()
        self.ride_saved.emit()

    def _reset_form(self) -> None:
        if self._default_home_address:
            self.start_input.setText(self._default_home_address)
        else:
            self.start_input.clear()
        self.dest_input.clear()
        self.driver_list.clearSelection()
        self.passenger_list.clearSelection()
        self.start_history_combo.setCurrentIndex(0)
        self.dest_history_combo.setCurrentIndex(0)
        self.flat_fee_input.setValue(self._default_flat_fee)
        self.per_km_input.setValue(self._default_fee_per_km)
        self.distance_value.setText("—")
        self.total_cost_value.setText("—")
        self.cost_per_passenger_value.setText("—")
        self.save_button.setEnabled(False)
        self._current_distance = None
        self._current_total_cost = None
        self._current_cost_per_passenger = None
        self._current_paying_passenger_ids = []
        self._current_all_passenger_ids = []
        self._current_driver_ids = []

    def _invalidate_calculation(self) -> None:
        self.save_button.setEnabled(False)
        self._current_distance = None
        self._current_total_cost = None
        self._current_cost_per_passenger = None
        self._current_paying_passenger_ids = []
        self._current_all_passenger_ids = []
        self._current_driver_ids = []
        self.distance_value.setText("—")
        self.total_cost_value.setText("—")
        self.cost_per_passenger_value.setText("—")


class RideHistoryTab(QWidget):
    """Display past rides and current ledger balances."""

    ride_deleted = pyqtSignal()

    def __init__(self, db_manager: DatabaseManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.db_manager = db_manager
        self._ledger_entries: list[dict[str, Any]] = []

        self.rides_table = QTableWidget(0, 9)
        self.rides_table.setHorizontalHeaderLabels(
            [
                "Date",
                "Drivers",
                "Passengers",
                "From",
                "To",
                "Distance (km)",
                "Flat Fee (per Driver)",
                "Fee/km",
                "Total",
            ]
        )
        self.rides_table.verticalHeader().setVisible(False)
        self.rides_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.rides_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.rides_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.rides_table.itemSelectionChanged.connect(self._update_delete_button_state)
        header = self.rides_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        rides_vheader = self.rides_table.verticalHeader()
        rides_vheader.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        rides_vheader.setDefaultSectionSize(40)
        self.rides_table.setMinimumHeight(
            rides_vheader.defaultSectionSize() * 3 + header.height() + 36
        )

        self.ledger_table = QTableWidget(0, 3)
        self.ledger_table.setHorizontalHeaderLabels(["Pays", "Receives", "Net Amount"])
        self.ledger_table.verticalHeader().setVisible(False)
        self.ledger_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.ledger_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        ledger_header = self.ledger_table.horizontalHeader()
        ledger_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        ledger_vheader = self.ledger_table.verticalHeader()
        ledger_vheader.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        ledger_vheader.setDefaultSectionSize(28)
        self.ledger_table.setWordWrap(False)
        self.ledger_table.setAlternatingRowColors(True)
        self.ledger_table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.ledger_table.setMinimumHeight(
            ledger_vheader.defaultSectionSize() * 14 + ledger_header.height() + 24
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(24)
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Ride History (Last 3)"))
        header_layout.addStretch()
        self.delete_button = QPushButton("Delete Selected Ride")
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(self._on_delete_clicked)
        header_layout.addWidget(self.delete_button)
        layout.addLayout(header_layout)
        layout.addWidget(self.rides_table)
        layout.addWidget(QLabel("Net Ledger (All Rides)"))
        layout.addWidget(self.ledger_table)
        button_bar = QHBoxLayout()
        button_bar.addStretch()
        self.export_button = QPushButton("Export Ledger to PDF")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export_ledger)
        button_bar.addWidget(self.export_button)
        layout.addLayout(button_bar)
        layout.addStretch(1)

    def refresh(self) -> None:
        rides = self.db_manager.fetch_rides_with_passengers(limit=3)
        self.rides_table.setRowCount(len(rides))
        for row_idx, ride in enumerate(rides):
            self._set_table_item(
                self.rides_table,
                row_idx,
                0,
                self._format_datetime(ride["ride_datetime"]),
                user_data=ride["id"],
            )
            drivers_text = ", ".join(ride["drivers"]) if ride["drivers"] else "—"
            passengers_text = ", ".join(ride["passengers"]) if ride["passengers"] else "—"
            self._set_table_item(self.rides_table, row_idx, 1, drivers_text)
            self._set_table_item(self.rides_table, row_idx, 2, passengers_text)
            self._set_table_item(self.rides_table, row_idx, 3, ride["start_address"])
            self._set_table_item(self.rides_table, row_idx, 4, ride["destination_address"])
            self._set_table_item(self.rides_table, row_idx, 5, f"{ride['distance_km']:.2f}")
            self._set_table_item(self.rides_table, row_idx, 6, f"€{ride['flat_fee']:.2f}")
            self._set_table_item(self.rides_table, row_idx, 7, f"€{ride['fee_per_km']:.2f}")
            self._set_table_item(self.rides_table, row_idx, 8, f"€{ride['total_cost']:.2f}")
        self.rides_table.resizeRowsToContents()
        self.rides_table.clearSelection()
        self._update_delete_button_state()

        ledger_entries = self.db_manager.fetch_ledger_summary()
        self._ledger_entries = ledger_entries
        self.export_button.setEnabled(bool(ledger_entries))
        self.ledger_table.setRowCount(len(ledger_entries))
        for row_idx, entry in enumerate(ledger_entries):
            self._set_table_item(self.ledger_table, row_idx, 0, entry["owes_name"])
            self._set_table_item(self.ledger_table, row_idx, 1, entry["owed_name"])
            self._set_table_item(self.ledger_table, row_idx, 2, f"€{entry['amount']:.2f}")
        self.ledger_table.resizeRowsToContents()

    def _on_export_ledger(self) -> None:
        if not self._ledger_entries:
            QMessageBox.information(
                self,
                "Nothing to Export",
                "There are no ledger entries to export right now.",
            )
            return

        default_name = f"rideshare_ledger_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export Ledger to PDF",
            str(Path.home() / default_name),
            "PDF Files (*.pdf)",
        )
        if not file_name:
            return

        try:
            result_path = export_ledger_pdf(file_name, self._ledger_entries)
        except Exception as exc:  # pylint: disable=broad-except
            QMessageBox.critical(
                self,
                "Export Failed",
                f"The ledger could not be exported.\n\nDetails: {exc}",
            )
            return

        QMessageBox.information(
            self,
            "Export Complete",
            f"Ledger exported successfully to:\n{result_path}",
        )

    def _selected_ride_id(self) -> Optional[int]:
        row = self.rides_table.currentRow()
        if row < 0:
            return None
        item = self.rides_table.item(row, 0)
        if item is None:
            return None
        ride_id = item.data(Qt.ItemDataRole.UserRole)
        if ride_id is None:
            return None
        return int(ride_id)

    def _update_delete_button_state(self) -> None:
        self.delete_button.setEnabled(self._selected_ride_id() is not None)

    def _on_delete_clicked(self) -> None:
        ride_id = self._selected_ride_id()
        if ride_id is None:
            return
        try:
            self.db_manager.delete_ride(ride_id)
        except sqlite3.DatabaseError as exc:
            QMessageBox.critical(self, "Database Error", f"Failed to delete ride: {exc}")
            return
        self.refresh()
        self.ride_deleted.emit()

    @staticmethod
    def _set_table_item(
        table: QTableWidget,
        row: int,
        column: int,
        text: str,
        user_data: Any | None = None,
    ) -> None:
        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        if user_data is not None:
            item.setData(Qt.ItemDataRole.UserRole, user_data)
        table.setItem(row, column, item)

    @staticmethod
    def _format_datetime(timestamp: str) -> str:
        try:
            dt = datetime.fromisoformat(timestamp)
        except ValueError:
            return timestamp
        return dt.strftime("%Y-%m-%d %H:%M")


class WindowTitleBar(QWidget):
    """Custom dark-themed window chrome with caption controls."""

    def __init__(self, window: QMainWindow) -> None:
        super().__init__(window)
        self._window = window
        self.setObjectName("WindowTitleBar")
        self.setFixedHeight(46)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 6, 14, 6)
        layout.setSpacing(10)

        self.title_label = QLabel(window.windowTitle())
        self.title_label.setObjectName("WindowTitleBarLabel")
        self.title_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.title_label)
        layout.addStretch(1)

        self.min_button = self._make_button("–", "Minimize", "ChromeMinButton")
        self.max_button = self._make_button("⬜", "Maximize", "ChromeMaxButton")
        self.close_button = self._make_button("×", "Close", "ChromeCloseButton")

        layout.addWidget(self.min_button)
        layout.addWidget(self.max_button)
        layout.addWidget(self.close_button)

        self.min_button.clicked.connect(self._window.showMinimized)
        self.max_button.clicked.connect(self._window.toggle_max_restore)
        self.close_button.clicked.connect(self._window.close)

        self.setStyleSheet(
            """
            QWidget#WindowTitleBar {
                background-color: #0f1623;
                border-bottom: 1px solid #1d2736;
            }
            QLabel#WindowTitleBarLabel {
                color: #f2f6ff;
                font-weight: 600;
                letter-spacing: 0.2px;
            }
            QPushButton[chrome="true"] {
                background-color: transparent;
                color: #dee7ff;
                border: none;
                border-radius: 4px;
                min-width: 36px;
                min-height: 28px;
            }
            QPushButton[chrome="true"]:hover {
                background-color: #203045;
            }
            QPushButton#ChromeCloseButton[chrome="true"]:hover {
                background-color: #d64545;
                color: #ffffff;
            }
            """
        )

    def _make_button(self, text: str, tooltip: str, object_name: str) -> QPushButton:
        button = QPushButton(text, self)
        button.setObjectName(object_name)
        button.setProperty("chrome", True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setToolTip(tooltip)
        return button

    def set_title(self, title: str) -> None:
        self.title_label.setText(title)

    def update_for_window_state(self, maximized: bool) -> None:
        if maximized:
            self.max_button.setText("❐")
            self.max_button.setToolTip("Restore Down")
        else:
            self.max_button.setText("⬜")
            self.max_button.setToolTip("Maximize")

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._window.end_drag()
            self._window.toggle_max_restore()
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._window.begin_drag(event.globalPosition().toPoint())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._window.drag_to(event.globalPosition().toPoint())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._window.end_drag()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class RideShareApp(QMainWindow):
    """Main window that orchestrates the individual tabs."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        maps_handler: GoogleMapsHandler,
        settings_manager: SettingsManager,
    ) -> None:
        super().__init__()
        self.db_manager = db_manager
        self.maps_handler = maps_handler
        self.settings_manager = settings_manager
        self.thread_pool = QThreadPool()
        self._is_dragging = False
        self._drag_offset = QPoint()

        if sys.platform == "win32":
            self.setWindowFlags(
                Qt.WindowType.Window
                | Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowSystemMenuHint
                | Qt.WindowType.WindowMinimizeButtonHint
                | Qt.WindowType.WindowMaximizeButtonHint
                | Qt.WindowType.WindowCloseButtonHint
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        else:
            self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)

        self.setWindowTitle("Table Tennis RideShare Manager")
        window_size = self.settings_manager.data.get("window_size", {})
        self.resize(
            int(window_size.get("width", 1100)),
            int(window_size.get("height", 740)),
        )

        self._chrome_body = QWidget(self)
        self._chrome_body.setObjectName("ChromeBody")
        chrome_layout = QVBoxLayout(self._chrome_body)
        chrome_layout.setContentsMargins(0, 0, 0, 0)
        chrome_layout.setSpacing(0)

        self.title_bar = WindowTitleBar(self)
        chrome_layout.addWidget(self.title_bar)

        self.tabs = QTabWidget()
        chrome_layout.addWidget(self.tabs, 1)

        grip_container = QWidget(self._chrome_body)
        grip_container.setFixedHeight(24)
        grip_layout = QHBoxLayout(grip_container)
        grip_layout.setContentsMargins(0, 0, 10, 10)
        grip_layout.setSpacing(0)
        grip_layout.addStretch(1)
        self._size_grip = QSizeGrip(grip_container)
        self._size_grip.setFixedSize(16, 16)
        grip_layout.addWidget(
            self._size_grip,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
        )
        chrome_layout.addWidget(grip_container, 0)

        self.setCentralWidget(self._chrome_body)
        self.windowTitleChanged.connect(self.title_bar.set_title)
        self.title_bar.update_for_window_state(self.isMaximized())
        self._size_grip.setVisible(not self.isMaximized())

        self.team_tab = TeamManagementTab(self.db_manager)
        self.team_tab.apply_settings(self.settings_manager.data)
        self.ride_tab = RideSetupTab(self.db_manager, self.maps_handler, self.thread_pool)
        self.ride_tab.apply_settings(self.settings_manager.data)
        self.history_tab = RideHistoryTab(self.db_manager)

        self.tabs.addTab(self.team_tab, "Team Management")
        self.tabs.addTab(self.ride_tab, "Ride Setup")
        self.tabs.addTab(self.history_tab, "Ride History & Ledger")

        self.team_tab.members_changed.connect(self._on_members_changed)
        self.ride_tab.ride_saved.connect(self._on_ride_saved)
        self.history_tab.ride_deleted.connect(self._on_ride_deleted)

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
        self._refresh_recent_addresses()

    def _on_members_changed(self, members: list[TeamMember]) -> None:
        self.ride_tab.set_team_members(members)
        self.history_tab.refresh()
        self._refresh_recent_addresses()

    def _refresh_recent_addresses(self) -> None:
        addresses = self.db_manager.fetch_recent_addresses()
        self.ride_tab.set_recent_addresses(
            addresses.get("start", []), addresses.get("destination", [])
        )

    def _on_ride_saved(self) -> None:
        self.history_tab.refresh()
        self._refresh_recent_addresses()

    def _on_ride_deleted(self) -> None:
        self._refresh_recent_addresses()

    def toggle_max_restore(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()
        self.title_bar.update_for_window_state(self.isMaximized())
        self._size_grip.setVisible(not self.isMaximized())

    def changeEvent(self, event: QEvent) -> None:  # type: ignore[override]
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            self.title_bar.update_for_window_state(self.isMaximized())
            self._size_grip.setVisible(not self.isMaximized())

    def begin_drag(self, global_pos: QPoint) -> None:
        if self.isMaximized():
            return
        self._is_dragging = True
        self._drag_offset = global_pos - self.frameGeometry().topLeft()

    def drag_to(self, global_pos: QPoint) -> None:
        if self._is_dragging and not self.isMaximized():
            self.move(global_pos - self._drag_offset)

    def end_drag(self) -> None:
        self._is_dragging = False

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.settings_manager.update(
            {
                "default_flat_fee": float(self.ride_tab.flat_fee_input.value()),
                "default_fee_per_km": float(self.ride_tab.per_km_input.value()),
                "window_size": {"width": self.width(), "height": self.height()},
                "team_table_column_widths": self.team_tab.get_column_widths(),
            }
        )
        super().closeEvent(event)
        self.history_tab.refresh()


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
    db_manager = DatabaseManager(DATABASE_FILE)
    maps_handler = GoogleMapsHandler(api_key)
    settings_manager = SettingsManager(SETTINGS_FILE)

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    stylesheet = load_stylesheet()
    if stylesheet:
        app.setStyleSheet(stylesheet)

    window = RideShareApp(db_manager, maps_handler, settings_manager)
    window.show()
    if not maps_handler.enabled:
        QTimer.singleShot(
            0,
            lambda: QMessageBox.warning(
                window,
                "Google Maps Disabled",
                (
                    "The Google Maps API key wasn't found. Autocomplete and distance lookup "
                    "are disabled until you add one to the environment or your .env file."
                ),
            ),
        )
    return app.exec()


if __name__ == "__main__":
    try:
        sys.exit(bootstrap_app())
    except GoogleMapsError as exc:
        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "Google Maps Configuration", str(exc))
        sys.exit(1)
