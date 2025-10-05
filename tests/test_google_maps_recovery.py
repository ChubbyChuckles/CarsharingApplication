import pytest
from pathlib import Path
from typing import Optional

from src import rideshare_app
from src.rideshare_app import DatabaseManager, GoogleMapsError, GoogleMapsHandler


class _SuccessClient:
    def __init__(self, key: str) -> None:  # noqa: ARG002
        self.calls = 0

    def distance_matrix(self, origins, destinations, mode, units):  # noqa: D401, ARG002
        self.calls += 1
        return {
            "status": "OK",
            "rows": [
                {
                    "elements": [
                        {
                            "status": "OK",
                            "distance": {"value": 4200},
                        }
                    ]
                }
            ],
        }


class _FailClient:
    def __init__(self, key: Optional[str] = None) -> None:  # noqa: ARG002
        pass

    def distance_matrix(self, origins, destinations, mode, units):  # noqa: D401, ARG002
        raise rideshare_app.TransportError("network down")


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rideshare_app.time, "sleep", lambda _seconds: None)


@pytest.fixture
def db_manager(tmp_path: Path) -> DatabaseManager:
    return DatabaseManager(tmp_path / "rideshare.db")


def test_distance_lookup_caches_success(
    monkeypatch: pytest.MonkeyPatch, db_manager: DatabaseManager
) -> None:
    monkeypatch.setattr(rideshare_app.googlemaps, "Client", _SuccessClient)
    handler = GoogleMapsHandler("dummy-key", db_manager)

    result = handler.distance_km("Start", "End")

    assert result.distance_km == pytest.approx(4.2)
    assert result.from_cache is False
    cached = db_manager.fetch_route_cache("Start", "End")
    assert cached is not None
    reverse_cached = db_manager.fetch_route_cache("End", "Start")
    assert reverse_cached is not None


def test_distance_lookup_uses_cache_when_api_fails(
    monkeypatch: pytest.MonkeyPatch,
    db_manager: DatabaseManager,
) -> None:
    monkeypatch.setattr(rideshare_app.googlemaps, "Client", _SuccessClient)
    handler = GoogleMapsHandler("dummy-key", db_manager)
    assert handler.distance_km("Start", "End").distance_km == pytest.approx(4.2)

    handler.client = _FailClient()
    result = handler.distance_km("Start", "End")

    assert result.from_cache is True
    assert "cached" in result.message.lower()


def test_disabled_maps_returns_cached_distance(db_manager: DatabaseManager) -> None:
    db_manager.upsert_route_cache("Start", "End", 5.5)
    handler = GoogleMapsHandler("", db_manager)

    result = handler.distance_km("Start", "End")

    assert result.from_cache is True
    assert result.distance_km == pytest.approx(5.5)


def test_distance_lookup_raises_without_cache(
    monkeypatch: pytest.MonkeyPatch, db_manager: DatabaseManager
) -> None:
    monkeypatch.setattr(rideshare_app.googlemaps, "Client", _FailClient)
    handler = GoogleMapsHandler("dummy-key", db_manager)

    with pytest.raises(GoogleMapsError):
        handler.distance_km("Start", "End")
