import pytest
from pathlib import Path
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QApplication, QDialog, QFrame

from src.rideshare_app import DatabaseManager, SettingsManager
from src.utils import onboarding


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_onboarding_wizard_collects_data(
    qapp,
):  # noqa: ARG001 - fixture ensures QApplication exists
    wizard = onboarding.OnboardingWizard(
        existing_api_key="",
        default_flat_fee=5.0,
        default_fee_per_km=0.5,
        existing_members=[],
    )

    assert "QWizard::header" in wizard.styleSheet()
    assert wizard.palette().color(QPalette.ColorRole.Window).name() == "#0f1724"
    assert wizard.palette().color(QPalette.ColorRole.WindowText).name() == "#edf2fb"
    assert (
        wizard._intro_page.palette().color(QPalette.ColorRole.Window).name()  # type: ignore[attr-defined]
        == "#152133"
    )
    cards = wizard.findChildren(QFrame, "OnboardingCard")
    assert len(cards) >= 4

    wizard._api_page._api_input.setText("NEW-KEY-123")  # type: ignore[attr-defined]
    wizard._tariff_page._flat_fee_input.setValue(6.75)  # type: ignore[attr-defined]
    wizard._tariff_page._per_km_input.setValue(0.65)  # type: ignore[attr-defined]
    wizard._team_page._team_input.setPlainText("Alice\nBob\n")  # type: ignore[attr-defined]

    wizard.accept()
    data = wizard.result_data

    assert data.api_key == "NEW-KEY-123"
    assert data.default_flat_fee == pytest.approx(6.75)
    assert data.default_fee_per_km == pytest.approx(0.65)
    assert data.team_members == ["Alice", "Bob"]


def test_maybe_run_onboarding_updates_state(tmp_path: Path, qapp):  # noqa: ARG001
    settings_path = tmp_path / "settings.json"
    db_path = tmp_path / "rideshare.db"

    settings_manager = SettingsManager(settings_path)
    db_manager = DatabaseManager(db_path)

    class StubWizard:
        def __init__(self, **kwargs):
            self._result_data = onboarding.OnboardingData(
                api_key="STUB-KEY",
                default_flat_fee=7.0,
                default_fee_per_km=0.7,
                team_members=["Charlie", "Dana"],
            )

        def exec(self) -> QDialog.DialogCode:
            return QDialog.DialogCode.Accepted

        @property
        def result_data(self) -> onboarding.OnboardingData:
            return self._result_data

    new_key = onboarding.maybe_run_onboarding(
        settings_data=settings_manager.data,
        settings_manager=settings_manager,
        db_manager=db_manager,
        api_key="",
        wizard_cls=StubWizard,
    )

    assert new_key == "STUB-KEY"
    assert settings_manager.data["google_maps_api_key"] == "STUB-KEY"
    assert settings_manager.data["onboarding"]["completed"] is True

    members = {member["name"] for member in db_manager.fetch_team_members()}
    assert {"Charlie", "Dana"}.issubset(members)


def test_maybe_run_onboarding_aborted(tmp_path: Path, qapp):  # noqa: ARG001
    settings_manager = SettingsManager(tmp_path / "settings.json")
    db_manager = DatabaseManager(tmp_path / "rideshare.db")

    class CancelWizard:
        def __init__(self, **kwargs):
            self._result_data = onboarding.OnboardingData(
                api_key="",
                default_flat_fee=5.0,
                default_fee_per_km=0.5,
                team_members=[],
            )

        def exec(self) -> QDialog.DialogCode:
            return QDialog.DialogCode.Rejected

        @property
        def result_data(self) -> onboarding.OnboardingData:
            return self._result_data

    with pytest.raises(onboarding.OnboardingAborted):
        onboarding.maybe_run_onboarding(
            settings_data=settings_manager.data,
            settings_manager=settings_manager,
            db_manager=db_manager,
            api_key="",
            wizard_cls=CancelWizard,
        )
