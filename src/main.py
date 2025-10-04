# src/main.py
"""Main entry point for the RideShare application."""

import sys

from PyQt6.QtWidgets import QApplication, QMessageBox

from .rideshare_app import GoogleMapsError, bootstrap_app


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
