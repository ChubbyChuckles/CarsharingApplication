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

    captured_message: Dict[str, str] = {}

    def fake_critical(parent, title, text):
        captured_message["title"] = title
        captured_message["text"] = text
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(rideshare_app.QMessageBox, "critical", fake_critical)

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    assert captured_message["title"] == "Google Maps Configuration"
    assert "Google Maps API key" in captured_message["text"]

    app = QApplication.instance()
    if app is not None:
        app.quit()
