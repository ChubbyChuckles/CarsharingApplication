from typing import Dict

import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox

from src import rideshare_app
from src.main import main


def test_main_exits_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the application exits gracefully if the Google Maps key is absent."""

    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "")

    def fake_load_dotenv(*args, **kwargs):  # pragma: no cover - simple shim
        return True

    monkeypatch.setattr(rideshare_app, "load_dotenv", fake_load_dotenv)

    captured_warning: Dict[str, str] = {}

    def fake_warning(parent, title, text):
        captured_warning["title"] = title
        captured_warning["text"] = text
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(rideshare_app.QMessageBox, "warning", fake_warning)

    def fake_single_shot(_msec, callback):
        callback()

    monkeypatch.setattr(rideshare_app.QTimer, "singleShot", staticmethod(fake_single_shot))

    monkeypatch.setattr(rideshare_app.QApplication, "exec", staticmethod(lambda: 0))
    monkeypatch.setattr(
        rideshare_app.RideShareApp,
        "nativeEvent",
        lambda self, eventType, message: (False, 0),
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0
    assert captured_warning["title"] == "Google Maps Disabled"
    assert "API key" in captured_warning["text"]

    app = QApplication.instance()
    if app is not None:
        app.quit()
