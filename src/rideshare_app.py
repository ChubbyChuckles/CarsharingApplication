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
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence

import numpy as np
import googlemaps
import pyqtgraph as pg
from dotenv import load_dotenv
from googlemaps.exceptions import ApiError, TransportError
from PyQt6.QtCore import (
    QDateTime,
    QObject,
    QRunnable,
    QThreadPool,
    QTimer,
    Qt,
    pyqtSignal,
    QStringListModel,
    QEvent,
    QPoint,
    QRegularExpression,
    QRectF,
)
from PyQt6.QtGui import (
    QColor,
    QFont,
    QRegularExpressionValidator,
    QValidator,
    QResizeEvent,
    QShortcut,
    QBrush,
    QKeySequence,
)
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QComboBox,
    QDateTimeEdit,
    QCompleter,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
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
    QSplitter,
    QSizeGrip,
    QSizePolicy,
    QStackedLayout,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .utils.onboarding import OnboardingAborted, maybe_run_onboarding
from .utils.pdf_exporter import export_ledger_pdf

pg.setConfigOptions(antialias=True, background=None, foreground="#f4f6fa")


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
        "google_maps_api_key": "",
        "onboarding": {"completed": False, "completed_at": None},
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


@dataclass(frozen=True)
class DistanceLookupResult:
    """Structured result for Google Maps distance lookups."""

    distance_km: float
    from_cache: bool
    attempts: int
    cached_at: Optional[datetime] = None
    message: str = ""


class GoogleMapsHandler:
    """Wrapper around the ``googlemaps`` client with helpful defaults."""

    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 0.75

    def __init__(self, api_key: str, db_manager: Optional["DatabaseManager"] = None) -> None:
        self.enabled = bool(api_key)
        self.client = None
        self._db_manager = db_manager
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

    def distance_km(self, origin: str, destination: str) -> DistanceLookupResult:
        """Return the driving distance between two addresses in kilometres.

        Includes retry logic and falls back to cached distances when the API is
        unavailable. The distance returned is one-way; callers should determine
        round-trip totals separately.
        """

        origin_key = origin.strip()
        destination_key = destination.strip()
        if not origin_key or not destination_key:
            raise GoogleMapsError("Both origin and destination addresses must be provided.")

        # Attempt immediate cache when Maps access is unavailable.
        if not self.enabled or self.client is None:
            cached = self._fetch_cached_distance(origin_key, destination_key)
            if cached is not None:
                return cached
            raise GoogleMapsError(
                "Google Maps API key is not configured. Distance lookup is disabled."
            )

        last_error: Optional[GoogleMapsError] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                distance_km = self._request_distance(origin_key, destination_key)
            except GoogleMapsError as exc:  # pragma: no cover - exercised in tests
                last_error = exc
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY_SECONDS)
                continue
            else:
                self._store_distance(origin_key, destination_key, distance_km)
                return DistanceLookupResult(
                    distance_km=distance_km,
                    from_cache=False,
                    attempts=attempt,
                )

        cached_result = self._fetch_cached_distance(origin_key, destination_key)
        if cached_result is not None:
            fallback_reason = (
                f"Falling back to cached route because the live lookup failed: {last_error}."
                if last_error
                else "Using cached route distance due to Google Maps connectivity issues."
            )
            combined_message = " ".join(
                part.strip() for part in (cached_result.message, fallback_reason) if part
            ).strip()
            return DistanceLookupResult(
                distance_km=cached_result.distance_km,
                from_cache=True,
                attempts=max(self.MAX_RETRIES, cached_result.attempts),
                cached_at=cached_result.cached_at,
                message=combined_message,
            )

        if last_error is not None:
            raise last_error
        raise GoogleMapsError("Distance lookup failed for an unknown reason.")

    def _request_distance(self, origin: str, destination: str) -> float:
        if self.client is None:
            raise GoogleMapsError("Google Maps client is not available.")
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

    def _store_distance(self, origin: str, destination: str, distance_km: float) -> None:
        if self._db_manager is None:
            return
        self._db_manager.upsert_route_cache(origin, destination, distance_km)
        # Store the reverse direction as a helpful shortcut for future lookups.
        self._db_manager.upsert_route_cache(destination, origin, distance_km)

    def _fetch_cached_distance(
        self, origin: str, destination: str
    ) -> Optional[DistanceLookupResult]:
        if self._db_manager is None:
            return None
        record = self._db_manager.fetch_route_cache(origin, destination)
        reversed_match = False
        if record is None:
            record = self._db_manager.fetch_route_cache(destination, origin)
            reversed_match = record is not None
        if record is None:
            return None

        cached_at: Optional[datetime] = None
        updated_at = record.get("updated_at")
        if updated_at:
            try:
                cached_at = datetime.fromisoformat(str(updated_at))
            except ValueError:  # pragma: no cover - defensive
                cached_at = None

        message_suffix = " (reverse order)" if reversed_match else ""
        message = "Using cached route distance" + message_suffix + "."
        return DistanceLookupResult(
            distance_km=float(record["distance_km"]),
            from_cache=True,
            attempts=0,
            cached_at=cached_at,
            message=message,
        )


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

        CREATE TABLE IF NOT EXISTS route_cache (
            origin TEXT NOT NULL,
            destination TEXT NOT NULL,
            distance_km REAL NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (origin, destination)
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
        ride_datetime: datetime | None = None,
    ) -> int:
        driver_ids = [int(driver_id) for driver_id in driver_ids]
        if not driver_ids:
            raise ValueError("At least one driver must be supplied for a ride.")

        passenger_ids = [int(pid) for pid in passenger_ids]
        paying_passenger_ids = [int(pid) for pid in paying_passenger_ids]
        recorded_at = ride_datetime or datetime.now(timezone.utc)
        if recorded_at.tzinfo is None:
            recorded_at = recorded_at.replace(tzinfo=timezone.utc)
        else:
            recorded_at = recorded_at.astimezone(timezone.utc)
        timestamp = recorded_at.isoformat(timespec="seconds")
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
                ORDER BY datetime(r.ride_datetime) DESC, r.id DESC
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

    def fetch_member_cost_trends(
        self,
        months: int = 6,
        reference: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Return total passenger spend per month for the requested window.

        The result is a dictionary with ``periods`` (e.g. ``["2025-01", ...]``) and ``series``
        (each entry contains ``member`` and ``values`` for each period).
        """

        if months <= 0:
            months = 6
        ref = reference or datetime.now(timezone.utc)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        else:
            ref = ref.astimezone(timezone.utc)

        periods: list[str] = []
        year = ref.year
        month = ref.month
        for _ in range(months):
            periods.append(f"{year:04d}-{month:02d}")
            month -= 1
            if month <= 0:
                month = 12
                year -= 1
        periods.reverse()
        period_lookup = set(periods)

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    le.passenger_id AS passenger_id,
                    tm.name AS passenger_name,
                    le.amount AS amount,
                    r.ride_datetime AS ride_datetime
                FROM ledger_entries le
                JOIN rides r ON le.ride_id = r.id
                JOIN team_members tm ON le.passenger_id = tm.id
                """
            ).fetchall()

        if not rows:
            return {"periods": periods, "series": []}

        aggregates: dict[str, dict[str, float]] = {}
        for row in rows:
            ride_timestamp = str(row["ride_datetime"]) if "ride_datetime" in row.keys() else ""
            try:
                ride_dt = datetime.fromisoformat(ride_timestamp)
            except ValueError:
                continue
            if ride_dt.tzinfo is None:
                ride_dt = ride_dt.replace(tzinfo=timezone.utc)
            else:
                ride_dt = ride_dt.astimezone(timezone.utc)
            period = f"{ride_dt.year:04d}-{ride_dt.month:02d}"
            if period not in period_lookup:
                continue

            name = str(row["passenger_name"]) if "passenger_name" in row.keys() else ""
            if not name:
                continue
            amount = float(row["amount"] or 0.0)
            member_data = aggregates.setdefault(name, {})
            member_data[period] = round(member_data.get(period, 0.0) + amount, 2)

        if not aggregates:
            return {"periods": periods, "series": []}

        series: list[dict[str, Any]] = []
        for name in sorted(aggregates.keys()):
            member_data = aggregates[name]
            values = [round(member_data.get(period, 0.0), 2) for period in periods]
            if any(value > 0 for value in values):
                series.append({"member": name, "values": values})

        return {"periods": periods, "series": series}

    def fetch_ride_frequency(self) -> dict[str, Any]:
        """Return ride counts grouped by weekday and hour (UTC)."""

        weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        hour_labels = [f"{hour:02d}" for hour in range(24)]
        matrix: list[list[int]] = [[0 for _ in range(24)] for _ in range(7)]

        with self._connect() as conn:
            rows = conn.execute("SELECT ride_datetime FROM rides").fetchall()

        total_rides = 0
        for row in rows:
            ride_timestamp = str(row["ride_datetime"]) if "ride_datetime" in row.keys() else ""
            try:
                ride_dt = datetime.fromisoformat(ride_timestamp)
            except ValueError:
                continue
            if ride_dt.tzinfo is None:
                ride_dt = ride_dt.replace(tzinfo=timezone.utc)
            else:
                ride_dt = ride_dt.astimezone(timezone.utc)

            weekday = ride_dt.weekday()
            hour = ride_dt.hour
            if 0 <= weekday < 7 and 0 <= hour < 24:
                matrix[weekday][hour] += 1
                total_rides += 1

        return {
            "matrix": matrix,
            "weekday_labels": weekday_labels,
            "hour_labels": hour_labels,
            "total_rides": total_rides,
        }

    def upsert_route_cache(self, origin: str, destination: str, distance_km: float) -> None:
        origin_key = origin.strip()
        destination_key = destination.strip()
        if not origin_key or not destination_key:
            return
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO route_cache (origin, destination, distance_km, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(origin, destination)
                DO UPDATE SET distance_km=excluded.distance_km, updated_at=excluded.updated_at
                """,
                (origin_key, destination_key, float(distance_km), timestamp),
            )
            conn.commit()

    def fetch_route_cache(self, origin: str, destination: str) -> Optional[dict[str, Any]]:
        origin_key = origin.strip()
        destination_key = destination.strip()
        if not origin_key or not destination_key:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT distance_km, updated_at FROM route_cache WHERE origin = ? AND destination = ?",
                (origin_key, destination_key),
            ).fetchone()
        if row is None:
            return None
        return {
            "distance_km": float(row["distance_km"] or 0.0),
            "updated_at": str(row["updated_at"]),
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

    def fetch_ledger_details(self) -> List[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    le.passenger_id,
                    passenger.name AS passenger_name,
                    le.driver_id,
                    driver.name AS driver_name,
                    le.amount,
                    r.id AS ride_id,
                    r.ride_datetime,
                    r.start_address,
                    r.destination_address,
                    r.distance_km,
                    r.total_cost,
                    r.flat_fee,
                    r.fee_per_km,
                    r.cost_per_passenger
                FROM ledger_entries le
                JOIN team_members passenger ON passenger.id = le.passenger_id
                JOIN team_members driver ON driver.id = le.driver_id
                JOIN rides r ON r.id = le.ride_id
                ORDER BY datetime(r.ride_datetime) ASC, passenger.name COLLATE NOCASE, driver.name COLLATE NOCASE
                """
            ).fetchall()

        details: List[dict[str, Any]] = []
        for row in rows:
            details.append(
                {
                    "passenger_id": int(row["passenger_id"]),
                    "passenger_name": str(row["passenger_name"]),
                    "driver_id": int(row["driver_id"]),
                    "driver_name": str(row["driver_name"]),
                    "amount": float(row["amount"] or 0.0),
                    "ride_id": int(row["ride_id"]),
                    "ride_datetime": str(row["ride_datetime"]),
                    "start_address": str(row["start_address"]),
                    "destination_address": str(row["destination_address"]),
                    "distance_km": float(row["distance_km"] or 0.0),
                    "total_cost": float(row["total_cost"] or 0.0),
                    "flat_fee": float(row["flat_fee"] or 0.0),
                    "fee_per_km": float(row["fee_per_km"] or 0.0),
                    "cost_per_passenger": float(row["cost_per_passenger"] or 0.0),
                }
            )
        return details


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


class CollapsibleSection(QWidget):
    """Reusable section widget with an expandable body for progressive disclosure."""

    toggled = pyqtSignal(bool)

    def __init__(
        self,
        title: str,
        *,
        description: str | None = None,
        expanded: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._expanded = expanded

        self._toggle = QToolButton(self)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._toggle.setText(title)
        self._toggle.setProperty("collapsibleHeader", True)
        self._toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._toggle.toggled.connect(self._on_toggled)

        self._description_label: QLabel | None = None
        if description:
            self._description_label = QLabel(description, self)
            self._description_label.setWordWrap(True)
            self._description_label.setProperty("role", "hint")

        self._content_frame = QFrame(self)
        self._content_frame.setFrameShape(QFrame.Shape.NoFrame)
        self._content_layout = QVBoxLayout(self._content_frame)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(16)
        self._content_frame.setVisible(expanded)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._toggle)
        if self._description_label is not None:
            layout.addWidget(self._description_label)
        layout.addWidget(self._content_frame)

    def add_content_widget(self, widget: QWidget) -> None:
        self._content_layout.addWidget(widget)

    def add_layout(self, layout: QVBoxLayout | QHBoxLayout | QGridLayout | QFormLayout) -> None:
        self._content_layout.addLayout(layout)

    def set_expanded(self, expanded: bool) -> None:
        if self._toggle.isChecked() != expanded:
            self._toggle.setChecked(expanded)
        else:
            self._apply_visibility(expanded)

    def is_expanded(self) -> bool:
        return self._toggle.isChecked()

    def _on_toggled(self, checked: bool) -> None:
        self._apply_visibility(checked)
        self.toggled.emit(checked)

    def _apply_visibility(self, expanded: bool) -> None:
        self._expanded = expanded
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._content_frame.setVisible(expanded)


class InlineFeedbackBanner(QFrame):
    """Compact inline alert widget for contextual validation feedback."""

    _ICONS: dict[str, str] = {
        "info": "ℹ",
        "success": "✔",
        "warning": "⚠",
        "error": "⛔",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("InlineFeedbackBanner")
        self.setProperty("severity", "info")
        self.setVisible(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._messages: list[str] = []
        self._severity = "info"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 14, 12)
        layout.setSpacing(12)

        self._icon_label = QLabel(self._ICONS["info"], self)
        self._icon_label.setObjectName("InlineFeedbackIcon")
        layout.addWidget(self._icon_label, 0, Qt.AlignmentFlag.AlignTop)

        self._message_label = QLabel("", self)
        self._message_label.setObjectName("InlineFeedbackMessage")
        self._message_label.setWordWrap(True)
        self._message_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self._message_label, 1)

        self._close_button = QPushButton("×", self)
        self._close_button.setObjectName("InlineFeedbackCloseButton")
        self._close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._close_button.setFlat(True)
        self._close_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._close_button.clicked.connect(self.clear)
        layout.addWidget(self._close_button, 0, Qt.AlignmentFlag.AlignTop)

    def show_messages(self, messages: Sequence[str], *, severity: str = "info") -> None:
        cleaned = [line.strip() for line in messages if line and line.strip()]
        if not cleaned:
            self.clear()
            return
        self._messages = cleaned
        self._severity = severity
        self.setProperty("severity", severity)
        icon = self._ICONS.get(severity, self._ICONS["info"])
        self._icon_label.setText(icon)

        if len(cleaned) == 1:
            body = cleaned[0]
        else:
            bullet_items = "".join(f"<li>{msg}</li>" for msg in cleaned)
            body = f"<ul>{bullet_items}</ul>"
        self._message_label.setText(body)
        self._refresh_style()
        self.setVisible(True)

    def show_message(self, message: str, *, severity: str = "info") -> None:
        self.show_messages([message], severity=severity)

    def clear(self) -> None:
        self._messages = []
        self._message_label.clear()
        self._severity = "info"
        self.setProperty("severity", "info")
        self._refresh_style()
        self.setVisible(False)

    @property
    def messages(self) -> list[str]:
        return list(self._messages)

    @property
    def severity(self) -> str:
        return self._severity

    def _refresh_style(self) -> None:
        style = self.style()
        style.unpolish(self)
        style.polish(self)
        self.update()


def _refresh_widget_style(widget: QWidget) -> None:
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


@dataclass
class NotificationEntry:
    entry_id: int
    created_at: datetime
    severity: str
    title: str
    message: str


class NotificationCenter(QWidget):
    """Display and persist recent application notifications."""

    unread_changed = pyqtSignal(int)

    _SEVERITY_COLORS: dict[str, str] = {
        "info": "#61bdf2",
        "success": "#46c38d",
        "warning": "#f0c674",
        "error": "#ff8080",
    }
    _SEVERITY_LABELS: dict[str, str] = {
        "info": "Info",
        "success": "Success",
        "warning": "Warning",
        "error": "Error",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("NotificationCenter")
        self._entries: list[NotificationEntry] = []
        self._next_id = count(1)
        self._unread = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        header_layout = QHBoxLayout()
        header_layout.setSpacing(10)
        self._header_label = QLabel("Activity Stream (0)")
        self._header_label.setProperty("role", "sectionLabel")
        header_layout.addWidget(self._header_label)
        header_layout.addStretch(1)
        self._unread_badge = QLabel("")
        self._unread_badge.setProperty("role", "hint")
        header_layout.addWidget(self._unread_badge)
        self._clear_button = QPushButton("Clear log")
        self._clear_button.setEnabled(False)
        self._clear_button.clicked.connect(self._clear_entries)
        header_layout.addWidget(self._clear_button)
        layout.addLayout(header_layout)

        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(8)
        filter_label = QLabel("Filter")
        filter_label.setProperty("role", "hint")
        filter_layout.addWidget(filter_label)
        self._filter_combo = QComboBox()
        self._filter_combo.addItem("All events", None)
        self._filter_combo.addItem("Info", "info")
        self._filter_combo.addItem("Success", "success")
        self._filter_combo.addItem("Warning", "warning")
        self._filter_combo.addItem("Error", "error")
        self._filter_combo.currentIndexChanged.connect(self._refresh_entries)
        filter_layout.addWidget(self._filter_combo)
        filter_layout.addStretch(1)
        layout.addLayout(filter_layout)

        self._list = QListWidget()
        self._list.setObjectName("NotificationList")
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._list, 2)

        self._empty_label = QLabel("No notifications yet. Activity will appear here.")
        self._empty_label.setProperty("role", "hint")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setWordWrap(True)
        layout.addWidget(self._empty_label)

        self._detail = QTextEdit()
        self._detail.setReadOnly(True)
        self._detail.setMinimumHeight(140)
        self._detail.setObjectName("NotificationDetail")
        layout.addWidget(self._detail, 1)

        self._update_placeholders(0)

    # Public API ------------------------------------------------------
    def add_entry(self, severity: str, title: str, message: str) -> None:
        entry = NotificationEntry(next(self._next_id), datetime.now(), severity, title, message)
        self._entries.insert(0, entry)
        self._unread += 1
        self.unread_changed.emit(self._unread)
        self._refresh_entries(preserve_selection=False)

    def mark_all_read(self) -> None:
        if self._unread:
            self._unread = 0
            self.unread_changed.emit(0)
            self._update_unread_badge()

    # Internal helpers ------------------------------------------------
    def _clear_entries(self) -> None:
        self._entries.clear()
        self._list.clear()
        self._detail.clear()
        self._unread = 0
        self.unread_changed.emit(0)
        self._update_placeholders(0)

    def _update_placeholders(self, filtered_count: int | None) -> None:
        if filtered_count is None:
            filtered_count = len(self._filtered_entries())
        has_results = filtered_count > 0
        self._empty_label.setVisible(not has_results)
        if not has_results:
            active_filter = self._filter_combo.currentData()
            if active_filter is None:
                self._empty_label.setText("No notifications yet. Activity will appear here.")
            else:
                label = self._SEVERITY_LABELS.get(active_filter, "events")
                self._empty_label.setText(f"No {label.lower()} entries match this filter just yet.")
        self._detail.setVisible(has_results)
        self._clear_button.setEnabled(bool(self._entries))
        self._header_label.setText(f"Activity Stream ({len(self._entries)})")
        self._update_unread_badge()

    def _update_unread_badge(self) -> None:
        if self._unread:
            self._unread_badge.setText(f"Unread: {self._unread}")
        else:
            self._unread_badge.clear()

    def _filtered_entries(self) -> list[NotificationEntry]:
        severity = self._filter_combo.currentData()
        if severity is None:
            return list(self._entries)
        return [entry for entry in self._entries if entry.severity == severity]

    def _refresh_entries(self, preserve_selection: bool = True) -> None:
        filtered = self._filtered_entries()
        selected_id: Optional[int] = None
        if preserve_selection:
            current_item = self._list.currentItem()
            if current_item is not None:
                selected_id = current_item.data(Qt.ItemDataRole.UserRole)

        self._list.blockSignals(True)
        self._list.clear()
        for entry in filtered:
            display_time = entry.created_at.strftime("%H:%M:%S")
            severity_label = self._SEVERITY_LABELS.get(entry.severity, entry.severity.title())
            item = QListWidgetItem(f"[{display_time}] {severity_label} · {entry.title}")
            item.setData(Qt.ItemDataRole.UserRole, entry.entry_id)
            item.setData(Qt.ItemDataRole.UserRole + 1, entry)
            color_hex = self._SEVERITY_COLORS.get(entry.severity)
            if color_hex:
                item.setForeground(QBrush(QColor(color_hex)))
            self._list.addItem(item)
        self._list.blockSignals(False)

        if filtered:
            row_to_select = 0
            if selected_id is not None:
                for index in range(self._list.count()):
                    item = self._list.item(index)
                    if item.data(Qt.ItemDataRole.UserRole) == selected_id:
                        row_to_select = index
                        break
            self._list.setCurrentRow(row_to_select)
        else:
            self._detail.clear()

        self._update_placeholders(len(filtered))

    def _on_selection_changed(self) -> None:
        item = self._list.currentItem()
        if item is None:
            self._detail.clear()
            return
        entry = item.data(Qt.ItemDataRole.UserRole + 1)
        if not isinstance(entry, NotificationEntry):
            self._detail.clear()
            return
        timestamp = entry.created_at.strftime("%Y-%m-%d %H:%M:%S")
        severity_label = self._SEVERITY_LABELS.get(entry.severity, entry.severity.title())
        detail = f"{severity_label} · {timestamp}\n" f"Source: {entry.title}\n\n" f"{entry.message}"
        self._detail.setPlainText(detail)


class TeamManagementTab(QWidget):
    """Widget responsible for CRUD operations on team members."""

    members_changed = pyqtSignal(list)
    activity_event = pyqtSignal(str, str, str)

    def __init__(self, db_manager: DatabaseManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.db_manager = db_manager
        self._selected_member: Optional[TeamMember] = None
        self._column_widths: Optional[list[int]] = None

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Team member name")
        self._name_validator = QRegularExpressionValidator(
            QRegularExpression(r"^(?=.{2,60}$)[A-Za-zÀ-ÖØ-öø-ÿ0-9 ,.'-]+$"),
            self,
        )
        self.name_input.setValidator(self._name_validator)
        self.name_input.textChanged.connect(self._on_name_changed)

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

        self.feedback_banner = InlineFeedbackBanner(self)

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
        layout.addWidget(self.feedback_banner)
        layout.addWidget(form_group)
        layout.addWidget(self.table, 1)

    def _wire_signals(self) -> None:
        self.add_button.clicked.connect(self._on_add_member)
        self.update_button.clicked.connect(self._on_update_member)
        self.delete_button.clicked.connect(self._on_delete_member)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)

    def _emit_activity(self, severity: str, title: str, message: str) -> None:
        self.activity_event.emit(severity, title, message)

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
        raw_text = self.name_input.text()
        state, *_ = self._name_validator.validate(raw_text, 0)
        if state != QValidator.State.Acceptable:
            message = (
                "Enter a name between 2 and 60 characters using letters, spaces, "
                "apostrophes, or hyphens."
            )
            self._mark_invalid(self.name_input, message)
            self.feedback_banner.show_message(message, severity="warning")
            self._emit_activity("warning", "Roster validation", message)
            return None
        name = raw_text.strip()
        self._clear_invalid(self.name_input)
        return name

    def _on_add_member(self) -> None:
        name = self._validate_name()
        if not name:
            return
        is_core = self._selected_type()
        try:
            self.db_manager.add_team_member(name, is_core)
        except sqlite3.IntegrityError:
            message = f"{name} is already on the roster. Try another name."
            self._mark_invalid(self.name_input, message)
            self.feedback_banner.show_message(message, severity="error")
            self._emit_activity("error", "Roster update failed", message)
            return
        self.name_input.clear()
        self._set_type_combo(True)
        self._selected_member = None
        self.refresh_members()
        self.table.clearSelection()
        self.feedback_banner.show_message(f"Added {name} to the roster.", severity="success")
        role_label = "core" if is_core else "reserve"
        self._emit_activity("success", "Roster updated", f"Added {name} as a {role_label} member.")

    def _on_update_member(self) -> None:
        name = self._validate_name()
        if not name:
            return
        if self._selected_member is None:
            message = "Select a team member in the table before updating."
            self._mark_invalid(self.table, message)
            self.feedback_banner.show_message(message, severity="warning")
            self._emit_activity("warning", "Roster update", "Update cancelled: no member selected.")
            return
        is_core = self._selected_type()
        try:
            self.db_manager.update_team_member(self._selected_member.member_id, name, is_core)
        except sqlite3.IntegrityError:
            message = f"{name} is already on the roster. Try another name."
            self._mark_invalid(self.name_input, message)
            self.feedback_banner.show_message(message, severity="error")
            self._emit_activity("error", "Roster update failed", message)
            return
        self.name_input.clear()
        self._set_type_combo(True)
        self._selected_member = None
        self.refresh_members()
        self.table.clearSelection()
        self._clear_invalid(self.table)
        self.feedback_banner.show_message("Team member updated.", severity="success")
        role_label = "core" if is_core else "reserve"
        self._emit_activity(
            "success", "Roster updated", f"Updated details for {name} ({role_label})."
        )

    def _on_delete_member(self) -> None:
        if self._selected_member is None:
            message = "Select a team member in the table before deleting."
            self._mark_invalid(self.table, message)
            self.feedback_banner.show_message(message, severity="warning")
            self._emit_activity(
                "warning", "Roster update", "Deletion cancelled: no member selected."
            )
            return
        try:
            self.db_manager.delete_team_member(self._selected_member.member_id)
        except sqlite3.IntegrityError:
            message = (
                f"{self._selected_member.name} is linked to existing rides and cannot be removed."
            )
            self.feedback_banner.show_message(message, severity="error")
            self._emit_activity("error", "Roster update failed", message)
            return
        deleted_name = self._selected_member.name
        self._selected_member = None
        self.name_input.clear()
        self._set_type_combo(True)
        self.refresh_members()
        self.table.clearSelection()
        self._clear_invalid(self.table)
        self.feedback_banner.show_message("Team member deleted.", severity="success")
        self._emit_activity("success", "Roster updated", f"Removed {deleted_name} from the roster.")

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
        self._clear_invalid(self.table)
        self.feedback_banner.clear()

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

    def _on_name_changed(self, text: str) -> None:
        state, *_ = self._name_validator.validate(text, 0)
        if state == QValidator.State.Acceptable:
            self._clear_invalid(self.name_input)
            if self.feedback_banner.severity in {"warning", "error"}:
                self.feedback_banner.clear()

    def _mark_invalid(self, widget: QWidget, message: str | None = None) -> None:
        widget.setProperty("validationState", "error")
        if message:
            widget.setToolTip(message)
        _refresh_widget_style(widget)

    def _clear_invalid(self, widget: QWidget) -> None:
        if widget.property("validationState"):
            widget.setProperty("validationState", "")
        widget.setToolTip("")
        _refresh_widget_style(widget)


class RideSetupTab(QWidget):
    """Configure and persist new rides, including cost calculations."""

    ride_saved = pyqtSignal()
    activity_event = pyqtSignal(str, str, str)

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
        address_pattern = QRegularExpression(r"^[^\n]{0,120}$")
        self._address_validator = QRegularExpressionValidator(address_pattern, self)
        self.start_input.setValidator(self._address_validator)
        self.dest_input.setValidator(self._address_validator)

        self.start_history_combo = QComboBox()
        self.start_history_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.dest_history_combo = QComboBox()
        self.dest_history_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._style_history_combo(self.start_history_combo)
        self._style_history_combo(self.dest_history_combo)
        self._constrain_history_combo(self.start_history_combo)
        self._constrain_history_combo(self.dest_history_combo)
        self._update_history_popup_width(self.start_history_combo)
        self._update_history_popup_width(self.dest_history_combo)

        self.ride_datetime_input = QDateTimeEdit()
        self.ride_datetime_input.setCalendarPopup(True)
        self.ride_datetime_input.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.ride_datetime_input.setDateTime(QDateTime.currentDateTime())
        self.ride_datetime_input.setMinimumWidth(0)
        self.ride_datetime_input.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.ride_datetime_input.setStyleSheet(
            """
            QDateTimeEdit {
                background-color: #1f2a36;
                color: #f4f6fa;
                border: 1px solid #31445a;
                border-radius: 6px;
                padding: 6px 10px;
            }
            QDateTimeEdit::drop-down {
                background-color: #2b3b4d;
                border-left: 1px solid #31445a;
                width: 24px;
            }
            QDateTimeEdit::down-arrow {
                image: none;
            }
            """
        )
        calendar = self.ride_datetime_input.calendarWidget()
        if calendar is not None:
            calendar.setFirstDayOfWeek(Qt.DayOfWeek.Monday)
            calendar.setStyleSheet(
                """
                QCalendarWidget {
                    background-color: #0f1724;
                    border: 1px solid #31445a;
                    color: #f4f6fa;
                }
                QCalendarWidget QWidget#qt_calendar_navigationbar {
                    background-color: #141f2e;
                    border-bottom: 1px solid #31445a;
                }
                QCalendarWidget QToolButton {
                    background-color: transparent;
                    color: #f4f6fa;
                    border: none;
                    padding: 4px 8px;
                }
                QCalendarWidget QToolButton:hover {
                    background-color: #1f2a36;
                }
                QCalendarWidget QSpinBox {
                    background-color: #1f2a36;
                    border: 1px solid #31445a;
                    color: #f4f6fa;
                }
                QCalendarWidget QAbstractItemView:enabled {
                    background-color: #101a2b;
                    color: #f4f6fa;
                    selection-background-color: #35c4c7;
                    selection-color: #041226;
                    gridline-color: #22324b;
                }
                QCalendarWidget QAbstractItemView:disabled {
                    color: rgba(244, 246, 250, 0.25);
                }
                """
            )

        self.driver_list = QListWidget()
        self.driver_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.driver_list.setMaximumWidth(260)
        self.passenger_list = QListWidget()
        self.passenger_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.passenger_list.setMaximumWidth(260)

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

        self._distance_detail_default = "Run a calculation to pull the latest Google Maps data."
        self._total_cost_detail_default = (
            "Flat fees plus distance are split across your driver team."
        )
        self._cost_per_passenger_detail_default = (
            "Core members divide the total once drivers are excluded."
        )

        self._current_distance: Optional[float] = None
        self._current_total_cost: Optional[float] = None
        self._current_cost_per_passenger: Optional[float] = None
        self._current_paying_passenger_ids: list[int] = []
        self._current_all_passenger_ids: list[int] = []
        self._current_driver_ids: list[int] = []

        self.validation_banner = InlineFeedbackBanner(self)
        self._layout_mode = "wide"

        self._build_layout()
        self._wire_signals()
        self._setup_live_validation()
        self._configure_focus_disclosure()
        self._apply_responsive_layout(self.width() or 1200)

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(18)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Create a shared ride in four quick steps")
        title.setProperty("role", "title")
        layout.addWidget(title)

        subtitle = QLabel(
            "Start with the route, confirm your drivers and passengers, adjust the tariffs, "
            "then review the split before saving."
        )
        subtitle.setWordWrap(True)
        subtitle.setProperty("role", "subtitle")
        layout.addWidget(subtitle)
        layout.addWidget(self.validation_banner)

        self._content_layout = QGridLayout()
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setHorizontalSpacing(20)
        self._content_layout.setVerticalSpacing(20)
        layout.addLayout(self._content_layout)

        self.address_section = CollapsibleSection(
            "Step 1 · Route details",
            description="Capture where the ride begins and ends.",
            expanded=True,
        )
        address_widget = QWidget()
        address_grid = QGridLayout(address_widget)
        address_grid.setContentsMargins(0, 0, 0, 0)
        address_grid.setHorizontalSpacing(12)
        address_grid.setVerticalSpacing(8)

        start_label = QLabel("Start address")
        start_label.setProperty("role", "sectionLabel")
        address_grid.addWidget(start_label, 0, 0)

        dest_label = QLabel("Destination")
        dest_label.setProperty("role", "sectionLabel")
        address_grid.addWidget(dest_label, 0, 1)

        self.start_history_combo.setToolTip("Select a start address from previous rides")
        start_row = QWidget()
        start_row_layout = QHBoxLayout(start_row)
        start_row_layout.setContentsMargins(0, 0, 0, 0)
        start_row_layout.setSpacing(8)
        start_row_layout.addWidget(self.start_input, 1)
        start_row_layout.addWidget(self.start_history_combo, 0)
        address_grid.addWidget(start_row, 1, 0)

        self.dest_history_combo.setToolTip("Select a destination from previous rides")
        dest_row = QWidget()
        dest_row_layout = QHBoxLayout(dest_row)
        dest_row_layout.setContentsMargins(0, 0, 0, 0)
        dest_row_layout.setSpacing(8)
        dest_row_layout.addWidget(self.dest_input, 1)
        dest_row_layout.addWidget(self.dest_history_combo, 0)
        address_grid.addWidget(dest_row, 1, 1)

        start_hint = QLabel("Tip: set a default home base in settings to auto-fill the start.")
        start_hint.setWordWrap(True)
        start_hint.setProperty("role", "hint")
        address_grid.addWidget(start_hint, 2, 0)

        dest_hint = QLabel("Maps suggestions appear once you type three characters.")
        dest_hint.setWordWrap(True)
        dest_hint.setProperty("role", "hint")
        address_grid.addWidget(dest_hint, 2, 1)

        ride_time_label = QLabel("Ride date & time")
        ride_time_label.setProperty("role", "sectionLabel")
        address_grid.addWidget(ride_time_label, 3, 0)

        address_grid.addWidget(self.ride_datetime_input, 4, 0)

        address_grid.setColumnStretch(0, 1)
        address_grid.setColumnStretch(1, 1)
        self.address_section.add_content_widget(address_widget)
        self._content_layout.addWidget(self.address_section, 0, 0)

        self.team_section = CollapsibleSection(
            "Step 2 · Team",
            description="Choose who drives and who shares the fare.",
            expanded=True,
        )
        team_widget = QWidget()
        team_layout = QVBoxLayout(team_widget)
        team_layout.setContentsMargins(0, 0, 0, 0)
        team_layout.setSpacing(12)

        lists_widget = QWidget()
        lists_layout = QHBoxLayout(lists_widget)
        lists_layout.setContentsMargins(0, 0, 0, 0)
        lists_layout.setSpacing(14)

        driver_column = QVBoxLayout()
        driver_label = QLabel("Drivers")
        driver_label.setProperty("role", "sectionLabel")
        driver_column.addWidget(driver_label)
        self.driver_list.setMinimumHeight(200)
        driver_column.addWidget(self.driver_list, 1)

        passenger_column = QVBoxLayout()
        passenger_label = QLabel("Passengers")
        passenger_label.setProperty("role", "sectionLabel")
        passenger_column.addWidget(passenger_label)
        self.passenger_list.setMinimumHeight(200)
        passenger_column.addWidget(self.passenger_list, 1)

        lists_layout.addLayout(driver_column, 1)
        lists_layout.addLayout(passenger_column, 1)
        team_layout.addWidget(lists_widget)

        team_hint = QLabel(
            "Core members pay automatically. Selected drivers are excluded from passenger costs."
        )
        team_hint.setWordWrap(True)
        team_hint.setProperty("role", "hint")
        team_layout.addWidget(team_hint)

        self.team_section.add_content_widget(team_widget)
        self._content_layout.addWidget(self.team_section, 0, 1, 2, 1)

        self.fees_section = CollapsibleSection(
            "Step 3 · Fees",
            description="Confirm tariffs for this ride — defaults come from settings.",
            expanded=True,
        )
        fees_widget = QWidget()
        fees_layout = QGridLayout(fees_widget)
        fees_layout.setContentsMargins(0, 0, 0, 0)
        fees_layout.setHorizontalSpacing(12)
        fees_layout.setVerticalSpacing(10)

        flat_label = QLabel("Flat fee per driver")
        flat_label.setProperty("role", "sectionLabel")
        fees_layout.addWidget(flat_label, 0, 0)
        fees_layout.addWidget(self.flat_fee_input, 1, 0)

        per_km_label = QLabel("Per kilometre")
        per_km_label.setProperty("role", "sectionLabel")
        fees_layout.addWidget(per_km_label, 0, 1)
        fees_layout.addWidget(self.per_km_input, 1, 1)

        fees_hint = QLabel("Include your shared maintenance or fuel costs here.")
        fees_hint.setWordWrap(True)
        fees_hint.setProperty("role", "hint")
        fees_layout.addWidget(fees_hint, 2, 0, 1, 2)

        fees_layout.setColumnStretch(0, 1)
        fees_layout.setColumnStretch(1, 1)
        self.fees_section.add_content_widget(fees_widget)
        self._content_layout.addWidget(self.fees_section, 1, 0)

        self.summary_section = CollapsibleSection(
            "Step 4 · Review & save",
            description="Calculate totals, then persist the ride to the ledger.",
            expanded=True,
        )

        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.setSpacing(14)

        cards_row = QHBoxLayout()
        cards_row.setContentsMargins(0, 0, 0, 0)
        cards_row.setSpacing(12)

        distance_card, self.distance_detail_label = self._create_metric_card(
            "Round trip distance",
            self.distance_value,
            self._distance_detail_default,
        )
        cards_row.addWidget(distance_card)

        total_card, self.total_cost_detail_label = self._create_metric_card(
            "Total ride cost",
            self.total_cost_value,
            self._total_cost_detail_default,
        )
        cards_row.addWidget(total_card)

        passenger_card, self.cost_per_passenger_detail_label = self._create_metric_card(
            "Cost per core passenger",
            self.cost_per_passenger_value,
            self._cost_per_passenger_detail_default,
        )
        cards_row.addWidget(passenger_card)

        summary_layout.addLayout(cards_row)

        actions_row = QHBoxLayout()
        actions_row.setSpacing(10)
        actions_row.addStretch()
        actions_row.addWidget(self.calculate_button)
        actions_row.addWidget(self.save_button)
        summary_layout.addLayout(actions_row)

        self.summary_section.add_content_widget(summary_widget)
        self._content_layout.addWidget(self.summary_section, 2, 0, 1, 2)

        self._content_layout.setColumnStretch(0, 5)
        self._content_layout.setColumnStretch(1, 3)
        self._content_layout.setRowStretch(0, 1)
        self._content_layout.setRowStretch(1, 0)
        self._content_layout.setRowStretch(2, 0)

        layout.addStretch(1)

    def _create_metric_card(
        self,
        title: str,
        value_label: QLabel,
        default_detail: str,
    ) -> tuple[QFrame, QLabel]:
        value_label.setProperty("role", "metricValue")
        value_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        card = QFrame()
        card.setProperty("card", True)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 14, 16, 14)
        card_layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setProperty("role", "metricTitle")
        card_layout.addWidget(title_label)

        card_layout.addWidget(value_label)

        detail_label = QLabel(default_detail)
        detail_label.setWordWrap(True)
        detail_label.setProperty("role", "metricDetail")
        card_layout.addWidget(detail_label)

        card_layout.addStretch(1)
        return card, detail_label

    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_responsive_layout(event.size().width())

    def _apply_responsive_layout(self, width: int) -> None:
        if width <= 0:
            return
        breakpoint = 980
        new_mode = "stacked" if width < breakpoint else "wide"
        if new_mode == self._layout_mode:
            return
        self._layout_mode = new_mode

        if new_mode == "stacked":
            self._content_layout.addWidget(self.address_section, 0, 0)
            self._content_layout.addWidget(self.team_section, 1, 0)
            self._content_layout.addWidget(self.fees_section, 2, 0)
            self._content_layout.addWidget(self.summary_section, 3, 0)

            self._content_layout.setColumnStretch(0, 1)
            self._content_layout.setColumnStretch(1, 0)
            self._content_layout.setRowStretch(0, 0)
            self._content_layout.setRowStretch(1, 1)
            self._content_layout.setRowStretch(2, 0)
            self._content_layout.setRowStretch(3, 0)
        else:
            self._content_layout.addWidget(self.address_section, 0, 0)
            self._content_layout.addWidget(self.team_section, 0, 1, 2, 1)
            self._content_layout.addWidget(self.fees_section, 1, 0)
            self._content_layout.addWidget(self.summary_section, 2, 0, 1, 2)

            self._content_layout.setColumnStretch(0, 5)
            self._content_layout.setColumnStretch(1, 3)
            self._content_layout.setRowStretch(0, 1)
            self._content_layout.setRowStretch(1, 0)
            self._content_layout.setRowStretch(2, 0)

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

    def _setup_live_validation(self) -> None:
        self.start_input.textChanged.connect(lambda: self._on_address_changed(self.start_input))
        self.dest_input.textChanged.connect(lambda: self._on_address_changed(self.dest_input))
        self.flat_fee_input.valueChanged.connect(
            lambda: self._on_numeric_changed(self.flat_fee_input)
        )
        self.per_km_input.valueChanged.connect(lambda: self._on_numeric_changed(self.per_km_input))
        self.driver_list.itemSelectionChanged.connect(self._clear_banner_if_error)
        self.passenger_list.itemSelectionChanged.connect(self._clear_banner_if_error)

    def _configure_focus_disclosure(self) -> None:
        self._section_focus_map: dict[QWidget, CollapsibleSection] = {
            self.start_input: self.address_section,
            self.dest_input: self.address_section,
            self.start_history_combo: self.address_section,
            self.dest_history_combo: self.address_section,
            self.ride_datetime_input: self.address_section,
            self.driver_list: self.team_section,
            self.passenger_list: self.team_section,
            self.flat_fee_input: self.fees_section,
            self.per_km_input: self.fees_section,
            self.calculate_button: self.summary_section,
            self.save_button: self.summary_section,
        }
        for widget in self._section_focus_map.keys():
            widget.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if event.type() == QEvent.Type.FocusIn:
            section = (
                self._section_focus_map.get(watched)
                if hasattr(self, "_section_focus_map")
                else None
            )
            if section is not None:
                section.set_expanded(True)
        return super().eventFilter(watched, event)

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
        self._constrain_history_combo(combo)
        self._update_history_popup_width(combo)

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

    def _constrain_history_combo(self, combo: QComboBox) -> None:
        metrics = combo.fontMetrics()
        default_text = "Select previous address…"
        width = metrics.horizontalAdvance(default_text) + 48
        policy = combo.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Policy.Fixed)
        combo.setSizePolicy(policy)
        combo.setFixedWidth(width)

    def _update_history_popup_width(self, combo: QComboBox) -> None:
        view = combo.view()
        if view is None:
            return
        metrics = view.fontMetrics()
        longest = max(
            (combo.itemText(index) for index in range(combo.count())), key=len, default=""
        )
        if not longest:
            longest = "Select previous address…"
        width = metrics.horizontalAdvance(longest) + 56
        view.setMinimumWidth(width)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

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
        for index in range(self.passenger_list.count()):
            item = self.passenger_list.item(index)
            member_id = int(item.data(Qt.ItemDataRole.UserRole))
            is_driver = member_id in driver_ids
            item.setHidden(is_driver)
            if is_driver and item.isSelected():
                item.setSelected(False)

    def _on_driver_selection_changed(self) -> None:
        self._sync_driver_passenger_selection()
        self._invalidate_calculation()

    def _collect_form_state(self) -> tuple[dict[str, Any] | None, list[str]]:
        errors: list[str] = []
        state: dict[str, Any] = {}

        start_address = self.start_input.text().strip()
        if len(start_address) < 3:
            message = "Enter a start address with at least three characters."
            errors.append(message)
            self._mark_invalid(self.start_input, message)
        else:
            self._clear_invalid(self.start_input)
            state["start_address"] = start_address

        destination_address = self.dest_input.text().strip()
        if len(destination_address) < 3:
            message = "Enter a destination with at least three characters."
            errors.append(message)
            self._mark_invalid(self.dest_input, message)
        else:
            self._clear_invalid(self.dest_input)
            state["destination_address"] = destination_address

        ride_datetime_qt = self.ride_datetime_input.dateTime()
        if not ride_datetime_qt.isValid():
            message = "Choose a valid ride date and time."
            errors.append(message)
            self._mark_invalid(self.ride_datetime_input, message)
        else:
            self._clear_invalid(self.ride_datetime_input)
            ride_datetime = datetime.fromtimestamp(
                ride_datetime_qt.toSecsSinceEpoch(), tz=timezone.utc
            )
            state["ride_datetime"] = ride_datetime

        driver_ids = self._selected_driver_ids()
        if not driver_ids:
            message = "Choose at least one driver."
            errors.append(message)
            self._mark_invalid(self.driver_list, message)
        else:
            self._clear_invalid(self.driver_list)
            state["driver_ids"] = driver_ids

        passenger_ids = self._selected_passenger_ids()
        if not passenger_ids:
            message = "Select at least one passenger."
            errors.append(message)
            self._mark_invalid(self.passenger_list, message)
        else:
            self._clear_invalid(self.passenger_list)
            state["passenger_ids"] = passenger_ids

        core_passenger_ids = [
            pid for pid in passenger_ids if self._is_core_member(pid) and pid not in driver_ids
        ]
        if passenger_ids and not core_passenger_ids:
            message = "Include at least one core team member in the passengers list."
            errors.append(message)
            self._mark_invalid(self.passenger_list, message)
        elif core_passenger_ids:
            state["core_passenger_ids"] = core_passenger_ids

        flat_fee = float(self.flat_fee_input.value())
        if flat_fee <= 0:
            message = "Flat fee must be greater than zero."
            errors.append(message)
            self._mark_invalid(self.flat_fee_input, message)
        else:
            self._clear_invalid(self.flat_fee_input)
            state["flat_fee"] = flat_fee

        per_km_fee = float(self.per_km_input.value())
        if per_km_fee <= 0:
            message = "Per-kilometre fee must be greater than zero."
            errors.append(message)
            self._mark_invalid(self.per_km_input, message)
        else:
            self._clear_invalid(self.per_km_input)
            state["per_km_fee"] = per_km_fee

        if errors:
            return None, errors
        return state, []

    def _emit_activity(self, severity: str, title: str, message: str) -> None:
        self.activity_event.emit(severity, title, message)

    def _mark_invalid(self, widget: QWidget, message: str | None = None) -> None:
        widget.setProperty("validationState", "error")
        if message:
            widget.setToolTip(message)
        _refresh_widget_style(widget)

    def _clear_invalid(self, widget: QWidget) -> None:
        if widget.property("validationState"):
            widget.setProperty("validationState", "")
        widget.setToolTip("")
        _refresh_widget_style(widget)

    def _clear_banner_if_error(self) -> None:
        if self.validation_banner.severity in {"warning", "error"}:
            self.validation_banner.clear()

    def _on_address_changed(self, widget: QLineEdit) -> None:
        state, *_ = self._address_validator.validate(widget.text(), 0)
        if state == QValidator.State.Acceptable:
            self._clear_invalid(widget)
            self._clear_banner_if_error()

    def _on_numeric_changed(self, widget: QDoubleSpinBox) -> None:
        if widget.value() > 0:
            self._clear_invalid(widget)
            self._clear_banner_if_error()

    # Event handlers ------------------------------------------------------
    def _on_api_error(self, message: str) -> None:
        detail = f"Google Maps error: {message}"
        self.validation_banner.show_message(detail, severity="error")
        self._emit_activity("error", "Google Maps autocomplete", detail)

    def _on_calculate_clicked(self) -> None:
        form_state, errors = self._collect_form_state()
        if errors:
            self.validation_banner.show_messages(errors, severity="warning")
            self._emit_activity(
                "warning",
                "Ride save",
                "Ride save blocked by validation errors in the form.",
            )
            return
        if form_state is None:
            return
        self.calculate_button.setEnabled(False)
        self.validation_banner.show_message(
            "Calculating the latest route distance…", severity="info"
        )
        self._emit_activity(
            "info",
            "Route calculation",
            f"Fetching distance for {form_state['start_address']} → {form_state['destination_address']}.",
        )
        worker = Worker(
            self.maps_handler.distance_km,
            form_state["start_address"],
            form_state["destination_address"],
        )
        worker.signals.finished.connect(
            lambda result, state=form_state: self._on_distance_ready(
                result,
                state["flat_fee"],
                state["per_km_fee"],
                state["passenger_ids"],
                state["core_passenger_ids"],
                state["driver_ids"],
            )
        )
        worker.signals.error.connect(self._on_calculate_error)
        self.thread_pool.start(worker)

    def _on_distance_ready(
        self,
        result: DistanceLookupResult,
        flat_fee: float,
        per_km_fee: float,
        passenger_ids: list[int],
        core_passenger_ids: list[int],
        driver_ids: list[int],
    ) -> None:
        self.calculate_button.setEnabled(True)
        distance_km = result.distance_km
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
        distance_detail = (
            f"{distance_km:.2f} km each way · {round_trip_distance:.2f} km round trip."
        )
        if result.from_cache:
            if result.cached_at is not None:
                cache_stamp = result.cached_at.strftime("%Y-%m-%d %H:%M UTC")
                distance_detail += f" Cached from {cache_stamp}."
            else:
                distance_detail += " Cached distance used."
        self.distance_detail_label.setText(distance_detail)
        self.total_cost_detail_label.setText(
            f"Flat: €{flat_fee:.2f} × {driver_count} + distance: €{distance_cost:.2f}."
        )
        self.cost_per_passenger_detail_label.setText(
            f"Split across {core_count} core passenger{'s' if core_count != 1 else ''}."
        )
        self.save_button.setEnabled(True)
        self.summary_section.set_expanded(True)
        banner_messages: list[str] = []
        if result.from_cache:
            cache_message = result.message or "Using cached route distance."
            banner_messages.append(cache_message)
        success_line = "Calculation updated. Review the totals, then save when you're ready."
        banner_messages.append(success_line)
        if len(banner_messages) == 1:
            self.validation_banner.show_message(banner_messages[0], severity="success")
        else:
            severity = "warning" if result.from_cache else "success"
            self.validation_banner.show_messages(banner_messages, severity=severity)

        activity_message = (
            f"Round trip {round_trip_distance:.2f} km · total cost €{total_cost:.2f} "
            f"for {len(core_passenger_ids)} core passenger(s)."
        )
        if result.from_cache:
            cache_activity = result.message or "Using cached route distance."
            self._emit_activity("warning", "Route calculation fallback", cache_activity)
            self._emit_activity("info", "Route calculation", activity_message)
        else:
            self._emit_activity("success", "Route calculation", activity_message)

    def _on_calculate_error(self, message: str) -> None:
        self.calculate_button.setEnabled(True)
        detail = f"Unable to calculate the route: {message}"
        self.validation_banner.show_message(detail, severity="error")
        self._emit_activity("error", "Route calculation", detail)

    def _on_save_clicked(self) -> None:
        form_state, errors = self._collect_form_state()
        if errors:
            self.validation_banner.show_messages(errors, severity="warning")
            return
        if form_state is None:
            return
        if self._current_distance is None or self._current_total_cost is None:
            self.validation_banner.show_message(
                "Run the cost calculation before saving so the totals are up to date.",
                severity="warning",
            )
            self.summary_section.set_expanded(True)
            self._emit_activity(
                "warning",
                "Ride save",
                "Save cancelled because the route calculation hasn't been run.",
            )
            return
        total_cost = self._current_total_cost
        if total_cost is None:
            self.validation_banner.show_message(
                "Run the cost calculation before saving so the totals are up to date.",
                severity="warning",
            )
            self.summary_section.set_expanded(True)
            self._emit_activity(
                "warning",
                "Ride save",
                "Save cancelled because route totals are unavailable.",
            )
            return

        core_passenger_ids = form_state["core_passenger_ids"]
        cost_per_passenger = round(total_cost / len(core_passenger_ids), 2)
        self._current_cost_per_passenger = cost_per_passenger

        try:
            self.db_manager.record_ride(
                start_address=form_state["start_address"],
                destination_address=form_state["destination_address"],
                distance_km=self._current_distance,
                driver_ids=form_state["driver_ids"],
                passenger_ids=form_state["passenger_ids"],
                paying_passenger_ids=core_passenger_ids,
                flat_fee=form_state["flat_fee"],
                fee_per_km=form_state["per_km_fee"],
                total_cost=total_cost,
                cost_per_passenger=cost_per_passenger,
                ride_datetime=form_state["ride_datetime"],
            )
        except sqlite3.DatabaseError as exc:
            detail = f"Failed to save the ride: {exc}"
            self.validation_banner.show_message(detail, severity="error")
            self._emit_activity("error", "Ride save", detail)
            return

        self._reset_form()
        self.validation_banner.show_message("Ride saved and ledger updated.", severity="success")
        self.ride_saved.emit()
        self._emit_activity(
            "success",
            "Ride saved",
            (
                f"Logged ride {form_state['start_address']} → {form_state['destination_address']} "
                f"for €{total_cost:.2f}."
            ),
        )

    def _reset_form(self) -> None:
        if self._default_home_address:
            self.start_input.setText(self._default_home_address)
        else:
            self.start_input.clear()
        self.dest_input.clear()
        self.ride_datetime_input.setDateTime(QDateTime.currentDateTime())
        self.driver_list.clearSelection()
        self.passenger_list.clearSelection()
        self.start_history_combo.setCurrentIndex(0)
        self.dest_history_combo.setCurrentIndex(0)
        self.flat_fee_input.setValue(self._default_flat_fee)
        self.per_km_input.setValue(self._default_fee_per_km)
        self.distance_value.setText("—")
        self.total_cost_value.setText("—")
        self.cost_per_passenger_value.setText("—")
        self.distance_detail_label.setText(self._distance_detail_default)
        self.total_cost_detail_label.setText(self._total_cost_detail_default)
        self.cost_per_passenger_detail_label.setText(self._cost_per_passenger_detail_default)
        self.save_button.setEnabled(False)
        self._current_distance = None
        self._current_total_cost = None
        self._current_cost_per_passenger = None
        self._current_paying_passenger_ids = []
        self._current_all_passenger_ids = []
        self._current_driver_ids = []
        for widget in (
            self.start_input,
            self.dest_input,
            self.driver_list,
            self.passenger_list,
            self.flat_fee_input,
            self.per_km_input,
        ):
            self._clear_invalid(widget)

    def _invalidate_calculation(self) -> None:
        was_ready = self._current_distance is not None or self.save_button.isEnabled()
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
        if was_ready:
            self.validation_banner.show_message(
                "Selections changed. Re-run the cost calculation to refresh totals.",
                severity="info",
            )
            self._emit_activity(
                "info",
                "Route calculation",
                "Ride inputs changed after a calculation. Totals cleared until you recalculate.",
            )


class RideHistoryTab(QWidget):
    """Display past rides and current ledger balances."""

    ride_deleted = pyqtSignal()
    activity_event = pyqtSignal(str, str, str)

    def __init__(self, db_manager: DatabaseManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.db_manager = db_manager
        self._ledger_entries: list[dict[str, Any]] = []
        self._summary_labels: dict[str, tuple[QLabel, QLabel]] = {}

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
        self.rides_table.setAlternatingRowColors(True)
        self.rides_table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
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
        self.ledger_table.setAlternatingRowColors(True)
        self.ledger_table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        ledger_header = self.ledger_table.horizontalHeader()
        ledger_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        ledger_vheader = self.ledger_table.verticalHeader()
        ledger_vheader.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        ledger_vheader.setDefaultSectionSize(28)
        self.ledger_table.setMinimumHeight(
            ledger_vheader.defaultSectionSize() * 10 + ledger_header.height() + 24
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(18)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Ledger overview")
        title.setProperty("role", "title")
        layout.addWidget(title)

        subtitle = QLabel(
            "Keep tabs on the latest rides and outstanding balances without scrolling through tables."
        )
        subtitle.setWordWrap(True)
        subtitle.setProperty("role", "subtitle")
        layout.addWidget(subtitle)

        self._content_layout = QGridLayout()
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setHorizontalSpacing(20)
        self._content_layout.setVerticalSpacing(20)
        layout.addLayout(self._content_layout)

        self.snapshot_section = CollapsibleSection(
            "Snapshot",
            description="High-level metrics for balances and recent rides.",
            expanded=True,
        )
        cards_container = QWidget()
        self._summary_cards_layout = QGridLayout(cards_container)
        self._summary_cards_layout.setContentsMargins(0, 0, 0, 0)
        self._summary_cards_layout.setHorizontalSpacing(12)
        self._summary_cards_layout.setVerticalSpacing(12)

        outstanding_card, outstanding_value, outstanding_detail = self._create_summary_card(
            "Outstanding balance",
            "€0.00",
            "All settled!",
        )
        self._summary_labels["outstanding"] = (outstanding_value, outstanding_detail)

        biggest_card, biggest_value, biggest_detail = self._create_summary_card(
            "Largest balance",
            "€0.00",
            "No balances yet.",
        )
        self._summary_labels["largest"] = (biggest_value, biggest_detail)

        latest_card, latest_value, latest_detail = self._create_summary_card(
            "Latest ride",
            "—",
            "No rides saved yet.",
        )
        self._summary_labels["latest"] = (latest_value, latest_detail)

        self._summary_cards = [outstanding_card, biggest_card, latest_card]
        for index, card in enumerate(self._summary_cards):
            self._summary_cards_layout.addWidget(card, 0, index)

        self.snapshot_section.add_content_widget(cards_container)
        self._content_layout.addWidget(self.snapshot_section, 0, 0)

        self.rides_section = CollapsibleSection(
            "Recent rides",
            description="A snapshot of the last three rides recorded.",
            expanded=True,
        )
        rides_content = QWidget()
        rides_content_layout = QVBoxLayout(rides_content)
        rides_content_layout.setContentsMargins(0, 0, 0, 0)
        rides_content_layout.setSpacing(12)

        rides_header = QHBoxLayout()
        rides_header.setSpacing(10)
        rides_heading = QLabel("Ride list")
        rides_heading.setProperty("role", "sectionLabel")
        rides_header.addWidget(rides_heading)
        rides_header.addStretch()
        self.delete_button = QPushButton("Delete selected ride")
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(self._on_delete_clicked)
        rides_header.addWidget(self.delete_button)
        rides_content_layout.addLayout(rides_header)
        rides_content_layout.addWidget(self.rides_table)
        rides_hint = QLabel("Select a row to enable deletion and inspect who travelled together.")
        rides_hint.setWordWrap(True)
        rides_hint.setProperty("role", "hint")
        rides_content_layout.addWidget(rides_hint)
        self.rides_section.add_content_widget(rides_content)
        self._content_layout.addWidget(self.rides_section, 0, 1, 2, 1)

        self.ledger_section = CollapsibleSection(
            "Ledger details",
            description="Expand for the full outstanding balance matrix.",
            expanded=False,
        )
        ledger_content = QWidget()
        ledger_layout = QVBoxLayout(ledger_content)
        ledger_layout.setContentsMargins(0, 0, 0, 0)
        ledger_layout.setSpacing(12)

        ledger_header_layout = QHBoxLayout()
        ledger_header_layout.setSpacing(10)
        ledger_heading = QLabel("Balances")
        ledger_heading.setProperty("role", "sectionLabel")
        ledger_header_layout.addWidget(ledger_heading)
        ledger_header_layout.addStretch()
        self.export_button = QPushButton("Export ledger to PDF")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export_ledger)
        ledger_header_layout.addWidget(self.export_button)
        ledger_layout.addLayout(ledger_header_layout)
        ledger_layout.addWidget(self.ledger_table)
        ledger_hint = QLabel("Export to share a polished PDF summary with the team.")
        ledger_hint.setWordWrap(True)
        ledger_hint.setProperty("role", "hint")
        ledger_layout.addWidget(ledger_hint)
        self.ledger_section.add_content_widget(ledger_content)
        self._content_layout.addWidget(self.ledger_section, 1, 0)

        layout.addStretch(1)
        self._layout_mode = "wide"
        self._arrange_summary_cards(self._layout_mode)
        self._apply_responsive_layout(self.width() or 1200)

        self.rides_table.itemSelectionChanged.connect(self._update_delete_button_state)

    def _emit_activity(self, severity: str, title: str, message: str) -> None:
        self.activity_event.emit(severity, title, message)

    def _create_summary_card(
        self,
        title: str,
        value_text: str,
        detail_text: str,
    ) -> tuple[QFrame, QLabel, QLabel]:
        card = QFrame()
        card.setProperty("card", True)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 14, 16, 14)
        card_layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setProperty("role", "metricTitle")
        card_layout.addWidget(title_label)

        value_label = QLabel(value_text)
        value_label.setProperty("role", "metricValue")
        card_layout.addWidget(value_label)

        detail_label = QLabel(detail_text)
        detail_label.setWordWrap(True)
        detail_label.setProperty("role", "metricDetail")
        card_layout.addWidget(detail_label)

        card_layout.addStretch(1)
        return card, value_label, detail_label

    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_responsive_layout(event.size().width())

    def _arrange_summary_cards(self, mode: str) -> None:
        for card in self._summary_cards:
            self._summary_cards_layout.removeWidget(card)

        if mode == "stacked":
            for row, card in enumerate(self._summary_cards):
                self._summary_cards_layout.addWidget(card, row, 0)
                self._summary_cards_layout.setRowStretch(row, 0)
            self._summary_cards_layout.setColumnStretch(0, 1)
            for column in range(1, 3):
                self._summary_cards_layout.setColumnStretch(column, 0)
            for column in range(1, len(self._summary_cards)):
                self._summary_cards_layout.setRowStretch(column, 0)
        else:
            for column, card in enumerate(self._summary_cards):
                self._summary_cards_layout.addWidget(card, 0, column)
            for column in range(len(self._summary_cards)):
                self._summary_cards_layout.setColumnStretch(column, 1)
            for row in range(len(self._summary_cards)):
                self._summary_cards_layout.setRowStretch(row, 0)

    def _apply_responsive_layout(self, width: int) -> None:
        if width <= 0:
            return

        new_mode = "stacked" if width < 1120 else "wide"
        if new_mode == self._layout_mode:
            return

        self._layout_mode = new_mode

        self._arrange_summary_cards(new_mode)

        if new_mode == "stacked":
            self._content_layout.addWidget(self.snapshot_section, 0, 0)
            self._content_layout.addWidget(self.rides_section, 1, 0)
            self._content_layout.addWidget(self.ledger_section, 2, 0)
            self._content_layout.setColumnStretch(0, 1)
            self._content_layout.setColumnStretch(1, 0)
            self._content_layout.setRowStretch(0, 0)
            self._content_layout.setRowStretch(1, 1)
            self._content_layout.setRowStretch(2, 0)
        else:
            self._content_layout.addWidget(self.snapshot_section, 0, 0)
            self._content_layout.addWidget(self.ledger_section, 1, 0)
            self._content_layout.addWidget(self.rides_section, 0, 1, 2, 1)
            self._content_layout.setColumnStretch(0, 3)
            self._content_layout.setColumnStretch(1, 5)
            self._content_layout.setRowStretch(0, 0)
            self._content_layout.setRowStretch(1, 1)
            self._content_layout.setRowStretch(2, 0)

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
        self._update_summary_cards(rides, ledger_entries)

    def _update_summary_cards(
        self,
        rides: list[dict[str, Any]],
        ledger_entries: list[dict[str, Any]],
    ) -> None:
        outstanding_value, outstanding_detail = self._summary_labels["outstanding"]
        total_outstanding = sum(entry["amount"] for entry in ledger_entries)
        outstanding_value.setText(f"€{total_outstanding:.2f}")
        if ledger_entries:
            balance_count = len(ledger_entries)
            member_names: set[str] = set()
            for entry in ledger_entries:
                member_names.add(entry["owes_name"])
                member_names.add(entry["owed_name"])
            balance_label = "balance" if balance_count == 1 else "balances"
            member_label = "teammate" if len(member_names) == 1 else "teammates"
            outstanding_detail.setText(
                f"{balance_count} open {balance_label} involving {len(member_names)} {member_label}."
            )
        else:
            outstanding_detail.setText("All settled!")

        largest_value, largest_detail = self._summary_labels["largest"]
        if ledger_entries:
            biggest = max(ledger_entries, key=lambda entry: entry["amount"])
            largest_value.setText(f"€{biggest['amount']:.2f}")
            largest_detail.setText(f"{biggest['owes_name']} owes {biggest['owed_name']}.")
        else:
            largest_value.setText("€0.00")
            largest_detail.setText("No balances yet.")

        latest_value, latest_detail = self._summary_labels["latest"]
        if rides:
            latest = rides[0]
            latest_value.setText(f"€{latest['total_cost']:.2f}")
            drivers = ", ".join(latest["drivers"]) if latest["drivers"] else "No drivers recorded"
            passengers = (
                ", ".join(latest["passengers"])
                if latest["passengers"]
                else "No passengers recorded"
            )
            timestamp = self._format_datetime(latest["ride_datetime"])
            latest_detail.setText(f"{timestamp}\nDrivers: {drivers}\nPassengers: {passengers}")
        else:
            latest_value.setText("—")
            latest_detail.setText("No rides saved yet.")

    def _on_export_ledger(self) -> None:
        if not self._ledger_entries:
            QMessageBox.information(
                self,
                "Nothing to Export",
                "There are no ledger entries to export right now.",
            )
            self._emit_activity(
                "warning",
                "Ledger export",
                "Export skipped because there are no ledger entries to include.",
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
            self._emit_activity(
                "info",
                "Ledger export",
                "Export cancelled before choosing a destination file.",
            )
            return

        ledger_details = self.db_manager.fetch_ledger_details()
        all_rides = self.db_manager.fetch_rides_with_passengers(limit=None)

        self._emit_activity(
            "info",
            "Ledger export",
            f"Building ledger PDF with {len(self._ledger_entries)} summary rows.",
        )

        try:
            result_path = export_ledger_pdf(
                file_name,
                self._ledger_entries,
                detailed_entries=ledger_details,
                rides=all_rides,
            )
        except Exception as exc:  # pylint: disable=broad-except
            QMessageBox.critical(
                self,
                "Export Failed",
                f"The ledger could not be exported.\n\nDetails: {exc}",
            )
            self._emit_activity("error", "Ledger export", f"Export failed: {exc}")
            return

        QMessageBox.information(
            self,
            "Export Complete",
            f"Ledger exported successfully to:\n{result_path}",
        )
        self._emit_activity(
            "success",
            "Ledger export",
            f"Ledger PDF saved to {result_path}",
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
        row = self.rides_table.currentRow()
        ride_id = self._selected_ride_id()
        if ride_id is None or row < 0:
            return
        try:
            self.db_manager.delete_ride(ride_id)
        except sqlite3.DatabaseError as exc:
            QMessageBox.critical(self, "Database Error", f"Failed to delete ride: {exc}")
            self._emit_activity(
                "error", "Ride deletion", f"Failed to delete ride #{ride_id}: {exc}"
            )
            return
        start_label = self.rides_table.item(row, 3)
        dest_label = self.rides_table.item(row, 4)
        summary = None
        if start_label and dest_label:
            summary = f"{start_label.text()} → {dest_label.text()}"
        self.refresh()
        self.ride_deleted.emit()
        details = summary if summary else f"ride #{ride_id}"
        self._emit_activity("success", "Ride deletion", f"Removed {details} from history.")

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


class AnalyticsTab(QWidget):
    """Visualise ride costs and activity using lightweight dashboards."""

    activity_event = pyqtSignal(str, str, str)

    def __init__(self, db_manager: DatabaseManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.db_manager = db_manager
        self._latest_periods: list[str] = []
        self._latest_series: list[dict[str, Any]] = []
        self._latest_frequency_total: int = 0

        self._view_background = QColor("#101a2b")
        self._axis_line_color = QColor("#2b3a52")
        self._axis_text_color = QColor("#dee7ff")
        self._grid_pen = pg.mkPen(QColor(35, 53, 74, 90), width=1)
        self._line_palette: list[QColor] = [
            QColor("#35c4c7"),
            QColor("#ff7b7b"),
            QColor("#ffd27d"),
            QColor("#4f70ff"),
            QColor("#9b59ff"),
        ]
        self._heatmap_cmap = pg.ColorMap(
            np.linspace(0.0, 1.0, 5),
            [
                (15, 22, 35),
                (26, 52, 78),
                (43, 105, 131),
                (53, 196, 199),
                (255, 210, 125),
            ],
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(18)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Analytics overview")
        title.setProperty("role", "title")
        layout.addWidget(title)

        subtitle = QLabel(
            "Spot spending patterns across recent months and identify the busiest travel windows."
        )
        subtitle.setWordWrap(True)
        subtitle.setProperty("role", "subtitle")
        layout.addWidget(subtitle)

        self.trend_section = CollapsibleSection(
            "Member cost trends",
            description="Compare how much each passenger has contributed over the selected window.",
            expanded=True,
        )
        trend_content = QWidget()
        trend_layout = QVBoxLayout(trend_content)
        trend_layout.setContentsMargins(0, 0, 0, 0)
        trend_layout.setSpacing(12)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(10)
        controls_row.addStretch(1)
        window_label = QLabel("Window")
        window_label.setProperty("role", "sectionLabel")
        controls_row.addWidget(window_label)
        self.period_combo = QComboBox()
        self.period_combo.addItem("Last 3 months", 3)
        self.period_combo.addItem("Last 6 months", 6)
        self.period_combo.addItem("Last 12 months", 12)
        controls_row.addWidget(self.period_combo)
        self.refresh_button = QPushButton("Refresh analytics")
        self.refresh_button.setCursor(Qt.CursorShape.PointingHandCursor)
        controls_row.addWidget(self.refresh_button)
        trend_layout.addLayout(controls_row)

        self.cost_plot = pg.PlotWidget()
        self.cost_plot.setBackground(self._view_background)
        self.cost_plot.setFrameShape(QFrame.Shape.NoFrame)
        self.cost_plot.setStyleSheet("border: 1px solid #1d2736;")
        self.cost_plot.setLabel("left", "Passenger spend (€)")
        self.cost_plot.setLabel("bottom", "Month")
        self.cost_plot.setMenuEnabled(False)
        self.cost_plot.setMinimumHeight(280)
        self.cost_plot.invertY(False)
        self.cost_legend = self.cost_plot.addLegend(offset=(10, 10))
        self._configure_cost_plot()

        self.cost_placeholder = QLabel(
            "Add rides with paying passengers to build monthly spend trends."
        )
        self.cost_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cost_placeholder.setProperty("role", "hint")

        self.cost_stack = QStackedLayout()
        self.cost_stack.addWidget(self.cost_plot)
        placeholder_container = QWidget()
        placeholder_container.setStyleSheet(
            "background-color: #101a2b; border: 1px dashed rgba(61, 85, 110, 0.35);"
        )
        placeholder_layout = QVBoxLayout(placeholder_container)
        placeholder_layout.setContentsMargins(0, 0, 0, 0)
        placeholder_layout.addStretch(1)
        placeholder_layout.addWidget(self.cost_placeholder, 0, Qt.AlignmentFlag.AlignCenter)
        placeholder_layout.addStretch(1)
        self.cost_stack.addWidget(placeholder_container)
        trend_layout.addLayout(self.cost_stack)

        trend_hint = QLabel(
            "Totals reflect passenger contributions recorded in the ledger. Drivers are excluded."
        )
        trend_hint.setWordWrap(True)
        trend_hint.setProperty("role", "hint")
        trend_layout.addWidget(trend_hint)

        self.trend_section.add_content_widget(trend_content)
        layout.addWidget(self.trend_section)

        self.heatmap_section = CollapsibleSection(
            "Ride frequency heatmap",
            description="Identify the weekdays and hours where rides occur most often.",
            expanded=True,
        )
        heat_content = QWidget()
        heat_layout = QVBoxLayout(heat_content)
        heat_layout.setContentsMargins(0, 0, 0, 0)
        heat_layout.setSpacing(12)

        self.heatmap_plot = pg.PlotWidget()
        self.heatmap_plot.setBackground(self._view_background)
        self.heatmap_plot.setFrameShape(QFrame.Shape.NoFrame)
        self.heatmap_plot.setStyleSheet("border: 1px solid #1d2736;")
        self.heatmap_plot.setMenuEnabled(False)
        self.heatmap_plot.setMouseEnabled(x=False, y=False)
        self.heatmap_plot.hideButtons()
        self.heatmap_plot.setLabel("bottom", "Hour (24h)")
        self.heatmap_plot.setLabel("left", "Weekday")
        self.heatmap_plot.showGrid(x=False, y=False)
        self.heatmap_plot.invertY(True)
        self.heatmap_plot.setMinimumHeight(320)
        self.heatmap_item = pg.ImageItem()
        self.heatmap_plot.addItem(self.heatmap_item)
        self._configure_heatmap_plot()

        self.heatmap_placeholder = QLabel(
            "Once rides are logged, their distribution appears here as a heatmap."
        )
        self.heatmap_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.heatmap_placeholder.setProperty("role", "hint")

        self.heatmap_stack = QStackedLayout()
        self.heatmap_stack.addWidget(self.heatmap_plot)
        heat_placeholder_container = QWidget()
        heat_placeholder_container.setStyleSheet(
            "background-color: #101a2b; border: 1px dashed rgba(61, 85, 110, 0.35);"
        )
        heat_placeholder_layout = QVBoxLayout(heat_placeholder_container)
        heat_placeholder_layout.setContentsMargins(0, 0, 0, 0)
        heat_placeholder_layout.addStretch(1)
        heat_placeholder_layout.addWidget(
            self.heatmap_placeholder,
            0,
            Qt.AlignmentFlag.AlignCenter,
        )
        heat_placeholder_layout.addStretch(1)
        self.heatmap_stack.addWidget(heat_placeholder_container)
        heat_layout.addLayout(self.heatmap_stack)

        heat_hint = QLabel("Hours use UTC based on saved ride timestamps.")
        heat_hint.setWordWrap(True)
        heat_hint.setProperty("role", "hint")
        heat_layout.addWidget(heat_hint)

        self.heatmap_section.add_content_widget(heat_content)
        layout.addWidget(self.heatmap_section)

        layout.addStretch(1)

        self.period_combo.currentIndexChanged.connect(self.refresh)
        self.refresh_button.clicked.connect(self.refresh)

        self.cost_stack.setCurrentIndex(1)
        self.heatmap_stack.setCurrentIndex(1)

    def refresh(self) -> None:
        months = int(self.period_combo.currentData(Qt.ItemDataRole.UserRole) or 6)

        trend_data = self.db_manager.fetch_member_cost_trends(months=months)
        periods = trend_data.get("periods", [])
        series = trend_data.get("series", [])
        self._latest_periods = periods
        self._latest_series = series

        self.cost_plot.clear()
        if self.cost_legend is not None:
            self.cost_legend.clear()

        messages: list[str] = []

        if series and periods:
            self.cost_stack.setCurrentIndex(0)
            x_values = list(range(len(periods)))
            bottom_axis = self.cost_plot.getAxis("bottom")
            bottom_axis.setTicks([[(index, label) for index, label in enumerate(periods)]])

            max_value = 0.0
            for index, entry in enumerate(series):
                values = entry.get("values", [])
                if len(values) != len(periods):
                    continue
                color = self._line_palette[index % len(self._line_palette)]
                pen = pg.mkPen(color=color, width=2)
                self.cost_plot.plot(
                    x_values,
                    values,
                    pen=pen,
                    name=entry.get("member", f"Member {index + 1}"),
                    symbol="o",
                    symbolSize=6,
                    symbolBrush=color,
                    symbolPen=pg.mkPen(color=color, width=1),
                )
                if values:
                    max_value = max(max_value, max(values))
            upper = max_value * 1.15 if max_value > 0 else 1.0
            self.cost_plot.setYRange(0, upper, padding=0.02)
            messages.append(
                f"cost trends for {len(series)} passenger{'s' if len(series) != 1 else ''}"
            )
        else:
            self.cost_stack.setCurrentIndex(1)
            if not series:
                self.cost_placeholder.setText(
                    "Add rides with core passengers to build monthly spend trends."
                )

        frequency = self.db_manager.fetch_ride_frequency()
        matrix = frequency.get("matrix", [])
        total_rides = int(frequency.get("total_rides", 0))
        self._latest_frequency_total = total_rides

        if total_rides > 0 and matrix:
            self.heatmap_stack.setCurrentIndex(0)
            arr = np.array(matrix, dtype=float)
            if arr.size == 0:
                arr = np.zeros((7, 24), dtype=float)
            max_count = float(arr.max()) if arr.size else 0.0
            if max_count <= 0:
                max_count = 1.0
            self.heatmap_item.setImage(arr, levels=(0.0, max_count))
            self.heatmap_item.setRect(QRectF(0, 0, arr.shape[1], arr.shape[0]))
            self.heatmap_item.setLookupTable(self._heatmap_cmap.getLookupTable(alpha=False))
            self.heatmap_item.setOpacity(0.92)

            hour_labels = frequency.get("hour_labels", [])
            weekday_labels = frequency.get("weekday_labels", [])
            bottom_ticks = [
                (hour + 0.5, label) for hour, label in enumerate(hour_labels) if hour % 2 == 0
            ]
            left_ticks = [(index + 0.5, label) for index, label in enumerate(weekday_labels)]
            self.heatmap_plot.getAxis("bottom").setTicks([bottom_ticks])
            self.heatmap_plot.getAxis("left").setTicks([left_ticks])
            self.heatmap_plot.setLimits(
                xMin=0,
                xMax=arr.shape[1],
                yMin=0,
                yMax=arr.shape[0],
            )
            messages.append(
                f"ride activity heatmap with {total_rides} entry{'ies' if total_rides != 1 else ''}"
            )
        else:
            self.heatmap_stack.setCurrentIndex(1)
            self.heatmap_placeholder.setText(
                "Once rides are logged, their distribution appears here as a heatmap."
            )

        if messages:
            summary = "; ".join(messages)
            self.activity_event.emit("info", "Analytics refreshed", summary.capitalize() + ".")

    def _configure_cost_plot(self) -> None:
        plot_item = self.cost_plot.getPlotItem()
        view_box = plot_item.getViewBox()
        view_box.setBackgroundColor(self._view_background)
        plot_item.showGrid(x=True, y=True, alpha=0.12)
        try:
            grid = plot_item.getGridItem()
            grid.setPen(self._grid_pen)
        except AttributeError:  # pragma: no cover - fallback if API absent
            pass
        plot_item.setMenuEnabled(False)
        try:
            self.cost_legend.setBrush(pg.mkBrush(QColor(15, 22, 35, 230)))
            self.cost_legend.setPen(pg.mkPen(QColor("#23324a")))
            self.cost_legend.setLabelTextColor(self._axis_text_color)
        except AttributeError:  # pragma: no cover - legend API differs across versions
            pass
        self._style_axis(plot_item, "left")
        self._style_axis(plot_item, "bottom")

    def _configure_heatmap_plot(self) -> None:
        plot_item = self.heatmap_plot.getPlotItem()
        view_box = plot_item.getViewBox()
        view_box.setBackgroundColor(self._view_background)
        plot_item.setMenuEnabled(False)
        plot_item.getAxis("right").setVisible(False)
        plot_item.getAxis("top").setVisible(False)
        self._style_axis(plot_item, "left")
        self._style_axis(plot_item, "bottom")

    def _style_axis(self, plot_item, axis: str) -> None:
        ax = plot_item.getAxis(axis)
        ax.setPen(pg.mkPen(self._axis_line_color, width=1))
        ax.setTextPen(pg.mkPen(self._axis_text_color))
        ax.setStyle(tickFont=QFont("Segoe UI", 9), tickTextOffset=6)


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

        self.notification_button = self._make_button(
            "🔔", "Activity center (Ctrl+Shift+N)", "ChromeNotifyButton"
        )
        self.notification_button.setCheckable(True)
        layout.addWidget(self.notification_button)

        self.min_button = self._make_button("–", "Minimize", "ChromeMinButton")
        self.max_button = self._make_button("⬜", "Maximize", "ChromeMaxButton")
        self.close_button = self._make_button("×", "Close", "ChromeCloseButton")

        layout.addWidget(self.min_button)
        layout.addWidget(self.max_button)
        layout.addWidget(self.close_button)

        self.min_button.clicked.connect(self._window.showMinimized)
        self.max_button.clicked.connect(self._window.toggle_max_restore)
        self.close_button.clicked.connect(self._window.close)
        self.set_notification_badge(0)

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
            QPushButton#ChromeNotifyButton[chrome="true"][panelVisible="true"] {
                background-color: #203045;
            }
            QPushButton#ChromeNotifyButton[chrome="true"][hasUnread="true"] {
                color: #ffd27d;
                font-weight: 600;
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

    def _refresh_button_style(self, button: QPushButton) -> None:
        style = button.style()
        style.unpolish(button)
        style.polish(button)
        button.update()

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

    def set_notification_badge(self, count: int) -> None:
        has_unread = count > 0
        text = "🔔" if not has_unread else f"🔔 {count}"
        self.notification_button.setText(text)
        self.notification_button.setProperty("hasUnread", has_unread)
        if has_unread:
            tooltip = (
                f"{count} unread alert{'s' if count != 1 else ''}. "
                "Toggle activity center (Ctrl+Shift+N)"
            )
        else:
            tooltip = "Activity center (Ctrl+Shift+N)"
        self.notification_button.setToolTip(tooltip)
        self._refresh_button_style(self.notification_button)

    def set_activity_panel_visible(self, visible: bool) -> None:
        self.notification_button.setChecked(visible)
        self.notification_button.setProperty("panelVisible", visible)
        self._refresh_button_style(self.notification_button)


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

        self._content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._content_splitter.setObjectName("ContentSplitter")
        self._content_splitter.setChildrenCollapsible(False)
        self._content_splitter.setHandleWidth(1)

        self.tabs = QTabWidget()
        self._content_splitter.addWidget(self.tabs)

        activity_panel = QWidget()
        activity_panel.setObjectName("ActivityPanel")
        activity_layout = QVBoxLayout(activity_panel)
        activity_layout.setContentsMargins(16, 16, 16, 16)
        activity_layout.setSpacing(12)

        activity_header = QHBoxLayout()
        activity_header.setSpacing(8)
        activity_label = QLabel("Activity & alerts")
        activity_label.setProperty("role", "sectionLabel")
        activity_header.addWidget(activity_label)
        activity_header.addStretch(1)
        activity_layout.addLayout(activity_header)

        sublabel = QLabel("System notifications, exports, and background tasks stay here.")
        sublabel.setWordWrap(True)
        sublabel.setProperty("role", "hint")
        activity_layout.addWidget(sublabel)

        self.notification_center = NotificationCenter()
        activity_layout.addWidget(self.notification_center, 1)

        self._content_splitter.addWidget(activity_panel)
        self._content_splitter.setStretchFactor(0, 4)
        self._content_splitter.setStretchFactor(1, 2)
        self._content_splitter.setSizes([780, 320])
        chrome_layout.addWidget(self._content_splitter, 1)

        self._notification_visible = True
        self._last_splitter_sizes = self._content_splitter.sizes()

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
        self.analytics_tab = AnalyticsTab(self.db_manager)

        self.tabs.addTab(self.team_tab, "Team Management")
        self.tabs.addTab(self.ride_tab, "Ride Setup")
        self.tabs.addTab(self.history_tab, "Ride History & Ledger")
        self.tabs.addTab(self.analytics_tab, "Analytics Dashboard")

        self.team_tab.members_changed.connect(self._on_members_changed)
        self.ride_tab.ride_saved.connect(self._on_ride_saved)
        self.history_tab.ride_deleted.connect(self._on_ride_deleted)
        self.team_tab.activity_event.connect(self._log_activity)
        self.ride_tab.activity_event.connect(self._log_activity)
        self.history_tab.activity_event.connect(self._log_activity)
        self.analytics_tab.activity_event.connect(self._log_activity)

        self.notification_center.unread_changed.connect(self._on_notification_unread_changed)
        self.title_bar.notification_button.clicked.connect(self._toggle_notification_panel)
        self.title_bar.set_activity_panel_visible(True)
        self._notification_shortcut = QShortcut(QKeySequence("Ctrl+Shift+N"), self)
        self._notification_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self._notification_shortcut.activated.connect(self._toggle_notification_panel)

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
        self.analytics_tab.refresh()
        self._refresh_recent_addresses()
        self._log_activity(
            "info",
            "Local data sync",
            (
                f"Loaded {len(members)} team member{'s' if len(members) != 1 else ''} and refreshed "
                "recent rides and addresses."
            ),
        )
        self.notification_center.mark_all_read()

    def _on_members_changed(self, members: list[TeamMember]) -> None:
        self.ride_tab.set_team_members(members)
        self.history_tab.refresh()
        self.analytics_tab.refresh()
        self._refresh_recent_addresses()

    def _refresh_recent_addresses(self) -> None:
        addresses = self.db_manager.fetch_recent_addresses()
        self.ride_tab.set_recent_addresses(
            addresses.get("start", []), addresses.get("destination", [])
        )

    def _log_activity(self, severity: str, title: str, message: str) -> None:
        if severity not in {"info", "success", "warning", "error"}:
            severity = "info"
        self.notification_center.add_entry(severity, title, message)
        if self._notification_visible:
            QTimer.singleShot(0, self.notification_center.mark_all_read)

    def _on_notification_unread_changed(self, count: int) -> None:
        self.title_bar.set_notification_badge(count)

    def _toggle_notification_panel(self) -> None:
        if self._notification_visible:
            self._last_splitter_sizes = self._content_splitter.sizes()
            self._content_splitter.setSizes([1, 0])
            self._notification_visible = False
        else:
            sizes = getattr(self, "_last_splitter_sizes", None)
            if not sizes or len(sizes) < 2 or sizes[1] <= 0:
                total_width = max(self.width(), 960)
                side_width = max(int(total_width * 0.28), 260)
                sizes = [max(total_width - side_width, 520), side_width]
            self._content_splitter.setSizes(sizes)
            self._last_splitter_sizes = self._content_splitter.sizes()
            self._notification_visible = True
            QTimer.singleShot(0, self.notification_center.mark_all_read)
        self.title_bar.set_activity_panel_visible(self._notification_visible)

    def _on_ride_saved(self) -> None:
        self.history_tab.refresh()
        self.analytics_tab.refresh()
        self._refresh_recent_addresses()
        self._log_activity(
            "info",
            "Background sync",
            "Ride history and address shortcuts refreshed after saving a ride.",
        )

    def _on_ride_deleted(self) -> None:
        self.analytics_tab.refresh()
        self._refresh_recent_addresses()
        self._log_activity(
            "info",
            "Background sync",
            "Recent address lists refreshed after deleting a ride.",
        )

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
    settings_manager = SettingsManager(SETTINGS_FILE)
    stored_key = str(settings_manager.data.get("google_maps_api_key", "")).strip()
    env_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    api_key = stored_key or env_key
    db_manager = DatabaseManager(DATABASE_FILE)

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    stylesheet = load_stylesheet()
    if stylesheet:
        app.setStyleSheet(stylesheet)

    try:
        api_key = maybe_run_onboarding(
            settings_data=settings_manager.data,
            settings_manager=settings_manager,
            db_manager=db_manager,
            api_key=api_key,
        )
    except OnboardingAborted:
        return 0

    os.environ["GOOGLE_MAPS_API_KEY"] = api_key
    maps_handler = GoogleMapsHandler(api_key, db_manager)

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
