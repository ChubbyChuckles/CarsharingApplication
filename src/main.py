# src/main.py
"""Main entry point for the RideShare application."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMessageBox

if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    _rideshare = import_module("src.rideshare_app")
else:  # pragma: no cover - import path depends on runtime context
    _rideshare = import_module(".rideshare_app", package=__package__)

GoogleMapsError = _rideshare.GoogleMapsError
bootstrap_app = _rideshare.bootstrap_app


def main() -> None:
    """Launch the PyQt6 RideShare GUI."""
    try:
        exit_code = bootstrap_app()
    except GoogleMapsError as exc:
        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "Google Maps Configuration", str(exc))
        raise SystemExit(1) from exc
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
