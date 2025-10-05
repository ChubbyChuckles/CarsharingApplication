"""Build a Windows MSI installer for the Table Tennis RideShare Manager.

This script relies on ``cx_Freeze`` and packages the PyQt6 application together with
its bundled resources. Run it from the project root:

    python scripts/build_msi.py bdist_msi

The resulting ``.msi`` will be written to the ``dist`` directory.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Iterable, Tuple

try:
    from cx_Freeze import Executable, setup  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - optional build dependency
    raise SystemExit(
        "cx_Freeze is required to build the Windows installer. "
        "Install it via 'pip install cx_Freeze' or 'pip install .[windows-installer]'."
    ) from exc

APP_NAME = "Table Tennis RideShare Manager"
APP_VERSION = "0.1.0"
SUMMARY = "Coordinate table tennis team rides, costs, and ledgers."
UPGRADE_CODE = "{F1976CC0-A8E8-4B65-9E45-DA0E49ABF6DF}"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
RESOURCES_ROOT = SRC_ROOT / "resources"
BUNDLED_CONFIG = SRC_ROOT / "config" / "settings.json"
BUNDLED_DATABASE = SRC_ROOT / "rideshare.db"
BUILD_DIR = PROJECT_ROOT / "build" / "msi"


def _collect_include_files() -> Iterable[Tuple[str, str]]:
    files: list[Tuple[str, str]] = []
    if RESOURCES_ROOT.exists():
        files.append((str(RESOURCES_ROOT / "style.qss"), "resources/style.qss"))
    if BUNDLED_CONFIG.exists():
        files.append((str(BUNDLED_CONFIG), "config/settings.json"))
    if BUNDLED_DATABASE.exists():
        files.append((str(BUNDLED_DATABASE), "rideshare.db"))
    return files


MSI_OPTIONS = {
    "add_to_path": False,
    "initial_target_dir": r"[ProgramFilesFolder]\\TableTennisRideShare",
    "upgrade_code": UPGRADE_CODE,
    "summary_data": {
        "comments": SUMMARY,
        "author": "ChubbyChuckles",
    },
}

EXECUTABLES = [
    Executable(
        script=str(SRC_ROOT / "main.py"),
        base="Win32GUI",
        target_name="TableTennisRideShare.exe",
        shortcut_name="Table Tennis RideShare",
        shortcut_dir="DesktopFolder",
    )
]


def _prime_template_files() -> None:
    """Ensure template config/database files exist before the build runs."""

    if not BUNDLED_CONFIG.exists():
        BUNDLED_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        BUNDLED_CONFIG.write_text("{}", encoding="utf-8")

    if not BUNDLED_DATABASE.exists():
        BUNDLED_DATABASE.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(BUNDLED_DATABASE) as connection:
            connection.execute("PRAGMA journal_mode = WAL")


def main() -> None:
    _prime_template_files()
    os.chdir(PROJECT_ROOT)
    build_options = {
        "packages": ["src", "PyQt6", "googlemaps", "dotenv"],
        "includes": ["sqlite3"],
        "include_files": list(_collect_include_files()),
        "excludes": ["tkinter"],
        "include_msvcr": True,
        "build_exe": str(BUILD_DIR),
    }
    setup(
        name=APP_NAME,
        version=APP_VERSION,
        description=SUMMARY,
        executables=EXECUTABLES,
        options={
            "build_exe": build_options,
            "bdist_msi": MSI_OPTIONS,
        },
    )


if __name__ == "__main__":
    main()
