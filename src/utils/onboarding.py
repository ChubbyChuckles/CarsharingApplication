"""Onboarding wizard for first-time setup of the RideShare application."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Sequence

import sqlite3

from PyQt6.QtCore import QPoint, Qt, QTimer
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QWidget,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)


@dataclass
class OnboardingData:
    """Collected values from the onboarding wizard."""

    api_key: str
    default_flat_fee: float
    default_fee_per_km: float
    team_members: List[str]


class _IntroPage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("Welcome to Table Tennis RideShare")
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 32, 24, 24)

        card = QFrame()
        card.setObjectName("OnboardingCard")
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(14)

        label = QLabel(
            "Before we dive in, let's capture a few essentials so rides, members, and\n"
            "maps data are ready to go. You can tweak everything later in Settings."
        )
        label.setWordWrap(True)
        card_layout.addWidget(label)

        layout.addWidget(card)
        layout.addStretch()
        self.setLayout(layout)


class _ApiKeyPage(QWizardPage):
    def __init__(self, initial_key: str) -> None:
        super().__init__()
        self.setTitle("Google Maps API key")
        self._api_input = QLineEdit(initial_key)
        self._api_input.setPlaceholderText("Paste your Google Maps API key (optional)")
        outer = QVBoxLayout()
        outer.setContentsMargins(24, 32, 24, 24)

        card = QFrame()
        card.setObjectName("OnboardingCard")
        layout = QFormLayout(card)
        layout.setHorizontalSpacing(18)
        layout.setVerticalSpacing(12)

        description = QLabel(
            "Autocomplete and distance calculations rely on a Google Maps API key.\n"
            "If you don't have one yet, continue and you can add it later from Settings."
        )
        description.setWordWrap(True)
        layout.addRow(description)
        layout.addRow("API key", self._api_input)
        outer.addWidget(card)
        outer.addStretch()
        self.setLayout(outer)

    @property
    def api_key(self) -> str:
        return self._api_input.text().strip()


class _TariffPage(QWizardPage):
    def __init__(self, default_flat_fee: float, default_fee_per_km: float) -> None:
        super().__init__()
        self.setTitle("Default ride costs")
        self._flat_fee_input = QDoubleSpinBox()
        self._flat_fee_input.setRange(0.0, 1_000.0)
        self._flat_fee_input.setPrefix("€ ")
        self._flat_fee_input.setDecimals(2)
        self._flat_fee_input.setValue(float(default_flat_fee))

        self._per_km_input = QDoubleSpinBox()
        self._per_km_input.setRange(0.0, 100.0)
        self._per_km_input.setPrefix("€ ")
        self._per_km_input.setDecimals(2)
        self._per_km_input.setValue(float(default_fee_per_km))

        outer = QVBoxLayout()
        outer.setContentsMargins(24, 32, 24, 24)

        card = QFrame()
        card.setObjectName("OnboardingCard")
        layout = QFormLayout(card)
        layout.setHorizontalSpacing(18)
        layout.setVerticalSpacing(12)

        description = QLabel(
            "Choose the baseline fees we should pre-fill every time you create a ride."
        )
        description.setWordWrap(True)
        layout.addRow(description)
        layout.addRow("Flat fee per vehicle", self._flat_fee_input)
        layout.addRow("Distance fee per km", self._per_km_input)
        outer.addWidget(card)
        outer.addStretch()
        self.setLayout(outer)

    @property
    def flat_fee(self) -> float:
        return float(self._flat_fee_input.value())

    @property
    def fee_per_km(self) -> float:
        return float(self._per_km_input.value())


class _TeamPage(QWizardPage):
    def __init__(self, existing_members: Sequence[str]) -> None:
        super().__init__()
        self.setTitle("Team roster")
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 32, 24, 24)
        layout.setSpacing(16)

        card = QFrame()
        card.setObjectName("OnboardingCard")
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(12)
        card_layout.setContentsMargins(0, 0, 0, 0)
        instructions = QLabel(
            "Enter the core players who regularly share rides. Add one name per line;\n"
            "you can always manage the roster from the Team tab later."
        )
        instructions.setWordWrap(True)
        card_layout.addWidget(instructions)

        self._team_input = QPlainTextEdit()
        if existing_members:
            self._team_input.setPlainText("\n".join(existing_members))
        else:
            self._team_input.setPlaceholderText("e.g.\nAlice Example\nBob Driver")
        card_layout.addWidget(self._team_input)
        layout.addWidget(card)
        layout.addStretch()
        self.setLayout(layout)

    @property
    def members(self) -> List[str]:
        lines = [line.strip() for line in self._team_input.toPlainText().splitlines()]
        return [line for line in lines if line]


class _OnboardingWizardPages(QWizard):
    """Wizard pages containing the onboarding flow."""

    def __init__(
        self,
        *,
        existing_api_key: str,
        default_flat_fee: float,
        default_fee_per_km: float,
        existing_members: Sequence[str],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("First-time setup")
        self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage, True)
        self.setModal(True)
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._intro_page = _IntroPage()
        self._api_page = _ApiKeyPage(existing_api_key)
        self._tariff_page = _TariffPage(default_flat_fee, default_fee_per_km)
        self._team_page = _TeamPage(existing_members)

        self.addPage(self._intro_page)
        self.addPage(self._api_page)
        self.addPage(self._tariff_page)
        self.addPage(self._team_page)

        self._result_data = OnboardingData(
            api_key=existing_api_key.strip(),
            default_flat_fee=float(default_flat_fee),
            default_fee_per_km=float(default_fee_per_km),
            team_members=list(existing_members),
        )
        self._apply_theme()
        QTimer.singleShot(0, self._refresh_theming_state)
        self.currentIdChanged.connect(lambda _id: self._refresh_theming_state())

    def accept(self) -> None:  # type: ignore[override]
        self._result_data = OnboardingData(
            api_key=self._api_page.api_key,
            default_flat_fee=self._tariff_page.flat_fee,
            default_fee_per_km=self._tariff_page.fee_per_km,
            team_members=self._team_page.members,
        )
        super().accept()

    @property
    def result_data(self) -> OnboardingData:
        return self._result_data

    def _apply_theme(self) -> None:
        background = QColor("#0f1724")
        page_background = QColor("#152133")
        text_color = QColor("#edf2fb")

        self._background_color = background
        self._page_background = page_background
        self._text_color = text_color

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, background)
        palette.setColor(QPalette.ColorRole.Base, page_background)
        palette.setColor(QPalette.ColorRole.Text, text_color)
        palette.setColor(QPalette.ColorRole.WindowText, text_color)
        palette.setColor(QPalette.ColorRole.Button, QColor("#1a2739"))
        palette.setColor(QPalette.ColorRole.ButtonText, text_color)
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#35c4c7"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#0f1724"))
        palette.setColor(QPalette.ColorRole.Light, QColor("#1d2736"))
        palette.setColor(QPalette.ColorRole.Midlight, QColor("#1a2435"))
        palette.setColor(QPalette.ColorRole.Mid, QColor("#142032"))
        palette.setColor(QPalette.ColorRole.Dark, QColor("#0a1320"))
        palette.setColor(QPalette.ColorRole.Shadow, QColor("#050a14"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#151f30"))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        self._apply_page_palettes()

        accent_start = "#1ba2a4"
        border_color = "#2b3b52"
        focus_color = "#35c4c7"

        self.setStyleSheet(
            f"""
            QWizard {{
                background-color: {background.name()};
                color: {text_color.name()};
                border: none;
            }}
            QWidget#OnboardingChrome {{
                background-color: {background.name()};
                border: 1px solid #1d2736;
                border-radius: 20px;
            }}
            QWizard QWidget#qt_wizard_titlebar,
            QWizard QFrame#qt_wizard_titlebar,
            QWizard QWidget#qt_wizard_header,
            QWizard QFrame#qt_wizard_header {{
                background-color: {background.name()};
                border: none;
                padding: 0px;
            }}
            QWizard::header {{
                background: transparent;
                border: none;
                margin: 0px;
                padding: 0px;
            }}
            QWizard QWidget#qt_wizard_header {{
                border-bottom: 1px solid #24334a;
            }}
            QWizard QWidget#qt_wizard_titlebar {{
                border-bottom: 1px solid #24334a;
            }}
            QWizard::separator {{
                background-color: #1d2736;
                height: 1px;
                margin: 0px;
            }}
            QWizard::page {{
                background-color: {page_background.name()};
                border: none;
            }}
            QWizard::title {{
                color: {text_color.name()};
                font-size: 16px;
                font-weight: 600;
            }}
            QWizard::subtitle {{
                color: #d6e3f5;
            }}
            QWizard QLabel#qt_wizard_title,
            QWizard QLabel#qt_wizard_subtitle {{
                color: {text_color.name()};
                background-color: transparent;
            }}
            QWizardPage {{
                background-color: {page_background.name()};
                border: none;
            }}
            QWizard QLabel {{
                color: {text_color.name()};
                background-color: transparent;
                qproperty-wordWrap: true;
            }}
            QWizard QFrame {{
                background-color: transparent;
                border: none;
            }}
            QWizard QLineEdit,
            QWizard QPlainTextEdit,
            QWizard QDoubleSpinBox,
            QWizard QSpinBox {{
                background-color: #1a2739;
                border: 1px solid {border_color};
                border-radius: 10px;
                padding: 6px 10px;
                color: {text_color.name()};
                selection-background-color: {accent_start};
                selection-color: #ffffff;
            }}
            QWizard QLineEdit:focus,
            QWizard QPlainTextEdit:focus,
            QWizard QDoubleSpinBox:focus,
            QWizard QSpinBox:focus {{
                border: 1px solid {focus_color};
            }}
            QWizard QDialogButtonBox {{
                background-color: {background.name()};
                border-top: 1px solid #24334a;
                padding: 12px 16px;
            }}
            QWizard QDialogButtonBox QPushButton {{
                background-color: #1c2940;
                color: #e8f2ff;
                border: 1px solid #2f3f57;
                border-radius: 12px;
                padding: 9px 22px;
                min-height: 38px;
                font-weight: 600;
                letter-spacing: 0.4px;
            }}
            QWizard QDialogButtonBox QPushButton:hover {{
                background-color: #23324a;
                border-color: #3b5680;
            }}
            QWizard QDialogButtonBox QPushButton:pressed {{
                background-color: #182335;
                border-color: #2b4260;
            }}
            QWizard QDialogButtonBox QPushButton:focus {{
                outline: none;
                border-color: #63d6ff;
            }}
            QWizard QDialogButtonBox QPushButton[variant="primary"] {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #3cd2ff, stop:0.5 #4a8aff, stop:1 #6f5cff);
                color: #041226;
                border: 1px solid #73d7ff;
                border-radius: 14px;
            }}
            QWizard QDialogButtonBox QPushButton[variant="primary"]:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #55dbff, stop:0.5 #5b96ff, stop:1 #8971ff);
                border-color: #90e8ff;
            }}
            QWizard QDialogButtonBox QPushButton[variant="primary"]:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                            stop:0 #2fb9f0, stop:0.5 #3d74eb, stop:1 #5b4fe0);
                border-color: #55bfff;
                color: #03101f;
            }}
            QWizard QDialogButtonBox QPushButton[variant="secondary"] {{
                background-color: rgba(59, 81, 118, 0.45);
                border: 1px solid #3c5d87;
                color: #d4e5ff;
            }}
            QWizard QDialogButtonBox QPushButton[variant="secondary"]:hover {{
                background-color: rgba(72, 103, 148, 0.6);
                border-color: #4c74a5;
            }}
            QWizard QDialogButtonBox QPushButton[variant="secondary"]:pressed {{
                background-color: rgba(46, 66, 101, 0.75);
                border-color: #33547d;
            }}
            QWizard QDialogButtonBox QPushButton[variant="ghost"] {{
                background-color: transparent;
                border: 1px solid transparent;
                color: #95a9c6;
            }}
            QWizard QDialogButtonBox QPushButton[variant="ghost"]:hover {{
                background-color: rgba(56, 76, 108, 0.45);
                border-color: rgba(104, 147, 196, 0.65);
                color: #dbe7ff;
            }}
            QWizard QDialogButtonBox QPushButton[variant="ghost"]:pressed {{
                background-color: rgba(38, 54, 79, 0.65);
                border-color: rgba(88, 122, 170, 0.7);
                color: #bfd0ec;
            }}
            QWizard QDialogButtonBox QPushButton:disabled {{
                background-color: #2a3b52;
                color: #879bb8;
                border-color: #2f3f57;
            }}
            QWizard QFrame#OnboardingCard {{
                background-color: #10243a;
                background: qradialgradient(cx:0.2, cy:0.2, radius:0.9,
                                            stop:0 rgba(60, 210, 255, 0.16),
                                            stop:1 rgba(16, 36, 58, 0.94));
                border: 1px solid #22324a;
                border-radius: 16px;
                padding: 20px 22px;
            }}
            QWizard QScrollBar:vertical,
            QWizard QScrollBar:horizontal {{
                background: #1c2940;
                border: none;
                border-radius: 6px;
            }}
            QWizard QScrollBar::handle:vertical,
            QWizard QScrollBar::handle:horizontal {{
                background: {focus_color};
                border-radius: 6px;
                min-height: 20px;
                min-width: 20px;
            }}
            QPushButton#OnboardingCloseButton {{
                font-size: 20px;
                line-height: 1;
            }}
            """
        )

        self._refresh_theming_state()

    def _refresh_theming_state(self) -> None:
        self._neutralize_system_frames()
        self._apply_button_variants()

    def _apply_button_variants(self) -> None:
        variant_map = {
            QWizard.WizardButton.NextButton: "primary",
            QWizard.WizardButton.FinishButton: "primary",
            QWizard.WizardButton.CommitButton: "primary",
            QWizard.WizardButton.BackButton: "secondary",
            QWizard.WizardButton.CustomButton1: "secondary",
            QWizard.WizardButton.CustomButton2: "secondary",
            QWizard.WizardButton.CustomButton3: "secondary",
            QWizard.WizardButton.CancelButton: "ghost",
            QWizard.WizardButton.HelpButton: "ghost",
        }

        seen_ids: set[int] = set()
        for role, variant in variant_map.items():
            button = self.button(role)
            if button is None:
                continue
            button_id = id(button)
            if button_id in seen_ids:
                continue
            seen_ids.add(button_id)

            button.setProperty("variant", variant)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            style = button.style()
            if style is not None:
                style.unpolish(button)
                style.polish(button)
            button.update()

    def _apply_page_palettes(self) -> None:
        page_background = getattr(self, "_page_background", QColor("#152133"))
        text_color = getattr(self, "_text_color", QColor("#edf2fb"))
        for page in (self._intro_page, self._api_page, self._tariff_page, self._team_page):
            page_palette = page.palette()
            page_palette.setColor(QPalette.ColorRole.Window, page_background)
            page_palette.setColor(QPalette.ColorRole.Base, page_background)
            page_palette.setColor(QPalette.ColorRole.Text, text_color)
            page_palette.setColor(QPalette.ColorRole.WindowText, text_color)
            page_palette.setColor(QPalette.ColorRole.Light, QColor("#1d2736"))
            page_palette.setColor(QPalette.ColorRole.Midlight, QColor("#1a2435"))
            page_palette.setColor(QPalette.ColorRole.Mid, QColor("#142032"))
            page_palette.setColor(QPalette.ColorRole.Dark, QColor("#0a1320"))
            page_palette.setColor(QPalette.ColorRole.Shadow, QColor("#050a14"))
            page_palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#151f30"))
            page.setPalette(page_palette)
            page.setAutoFillBackground(True)
            page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    def _neutralize_system_frames(self) -> None:
        background = getattr(self, "_background_color", QColor("#0f1724"))
        page_background = getattr(self, "_page_background", QColor("#152133"))
        for frame in self.findChildren(QFrame):
            name = frame.objectName()
            if name == "OnboardingCard":
                continue
            shape = frame.frameShape()
            has_border = frame.lineWidth() > 0 or frame.midLineWidth() > 0
            if shape in (QFrame.Shape.HLine, QFrame.Shape.VLine) or has_border:
                frame.setFrameShape(QFrame.Shape.NoFrame)
                frame.setLineWidth(0)
                frame.setMidLineWidth(0)
                frame.setStyleSheet("background-color: transparent; border: none;")

        for container in self.findChildren(
            QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly
        ):
            if container is self:
                continue
            if container.objectName():
                continue
            if isinstance(container, QWizardPage):
                continue
            pal = container.palette()
            win = pal.color(QPalette.ColorRole.Window)
            if win.red() > 230 and win.green() > 230 and win.blue() > 230:
                for role in (
                    QPalette.ColorRole.Window,
                    QPalette.ColorRole.Base,
                    QPalette.ColorRole.Button,
                    QPalette.ColorRole.Light,
                    QPalette.ColorRole.Midlight,
                    QPalette.ColorRole.Mid,
                    QPalette.ColorRole.Dark,
                    QPalette.ColorRole.Shadow,
                    QPalette.ColorRole.AlternateBase,
                ):
                    pal.setColor(
                        role, background if role != QPalette.ColorRole.Base else page_background
                    )
                container.setPalette(pal)
                container.setAutoFillBackground(True)
                container.setStyleSheet(
                    f"background-color: {background.name()}; border-top: none; border-bottom: 1px solid #1d2736;"
                )

        self._apply_page_palettes()


class OnboardingAborted(RuntimeError):
    """Raised when the onboarding flow is cancelled by the user."""


class _OnboardingTitleBar(QWidget):
    """Slimmed-down window chrome for the onboarding dialog."""

    def __init__(self, shell: "OnboardingWizard") -> None:
        super().__init__(shell)
        self._shell = shell
        self.setObjectName("OnboardingTitleBar")
        self.setFixedHeight(42)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 6, 12, 6)
        layout.setSpacing(10)

        self.title_label = QLabel(shell.windowTitle())
        self.title_label.setObjectName("OnboardingTitleLabel")
        self.title_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.title_label)
        layout.addStretch(1)

        self.close_button = QPushButton("×", self)
        self.close_button.setObjectName("OnboardingCloseButton")
        self.close_button.setProperty("chrome", True)
        self.close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.close_button.setToolTip("Cancel setup")
        layout.addWidget(self.close_button)

        self.close_button.clicked.connect(self._shell.reject)

        self.setStyleSheet(
            """
            QWidget#OnboardingTitleBar {
                background-color: #0f1623;
                border-bottom: 1px solid #1d2736;
            }
            QLabel#OnboardingTitleLabel {
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
            QPushButton#OnboardingCloseButton[chrome="true"]:hover {
                background-color: #d64545;
                color: #ffffff;
            }
            """
        )

    def set_title(self, title: str) -> None:
        self.title_label.setText(title)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._shell.begin_drag(event.globalPosition().toPoint())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._shell.drag_to(event.globalPosition().toPoint())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._shell.end_drag()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class OnboardingWizard(QDialog):
    """Frameless dialog that hosts the onboarding wizard pages."""

    def __init__(
        self,
        *,
        existing_api_key: str,
        default_flat_fee: float,
        default_fee_per_km: float,
        existing_members: Sequence[str],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle("First-time setup")
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowSystemMenuHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

        self._is_dragging = False
        self._drag_offset = QPoint()

        chrome = QWidget(self)
        chrome.setObjectName("OnboardingChrome")
        chrome_layout = QVBoxLayout(chrome)
        chrome_layout.setContentsMargins(0, 0, 0, 0)
        chrome_layout.setSpacing(0)

        self._title_bar = _OnboardingTitleBar(self)
        chrome_layout.addWidget(self._title_bar, 0)

        self._wizard = _OnboardingWizardPages(
            existing_api_key=existing_api_key,
            default_flat_fee=default_flat_fee,
            default_fee_per_km=default_fee_per_km,
            existing_members=existing_members,
            parent=self,
        )
        self._wizard.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        chrome_layout.addWidget(self._wizard, 1)
        self._wizard._apply_page_palettes()
        self._wizard._refresh_theming_state()

        self.setPalette(self._wizard.palette())
        self.setAutoFillBackground(True)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        outer_layout.addWidget(chrome)

        self._wizard.accepted.connect(self._on_child_accepted)
        self._wizard.rejected.connect(self._on_child_rejected)

        # expose inner pages for tests/backwards compatibility
        self._intro_page = self._wizard._intro_page
        self._api_page = self._wizard._api_page
        self._tariff_page = self._wizard._tariff_page
        self._team_page = self._wizard._team_page

        self.resize(720, 520)

    @property
    def result_data(self) -> OnboardingData:
        return self._wizard.result_data

    def accept(self) -> None:  # type: ignore[override]
        self._wizard.accept()

    def reject(self) -> None:  # type: ignore[override]
        self._wizard.reject()

    def begin_drag(self, global_pos: QPoint) -> None:
        self._is_dragging = True
        self._drag_offset = global_pos - self.frameGeometry().topLeft()

    def drag_to(self, global_pos: QPoint) -> None:
        if self._is_dragging:
            self.move(global_pos - self._drag_offset)

    def end_drag(self) -> None:
        self._is_dragging = False

    def styleSheet(self) -> str:  # type: ignore[override]
        return self._wizard.styleSheet()

    def setStyleSheet(self, style: str) -> None:  # type: ignore[override]
        self._wizard.setStyleSheet(style)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._title_bar.set_title(self.windowTitle())
        self._wizard._neutralize_system_frames()

    def _on_child_accepted(self) -> None:
        super().accept()

    def _on_child_rejected(self) -> None:
        super().reject()


def maybe_run_onboarding(
    *,
    settings_data: dict,
    settings_manager,
    db_manager,
    api_key: str,
    wizard_cls=OnboardingWizard,
) -> str:
    """Run the onboarding wizard if required and return the active API key.

    ``settings_manager`` and ``db_manager`` are passed in so the wizard can persist
    results. ``wizard_cls`` is injectable for testing.
    """

    existing_members = [member["name"] for member in db_manager.fetch_team_members()]
    needs_onboarding = not settings_data.get("onboarding", {}).get("completed", False)
    if not api_key:
        needs_onboarding = True
    if not existing_members:
        needs_onboarding = True

    if not needs_onboarding:
        return api_key

    wizard: OnboardingWizard = wizard_cls(
        existing_api_key=api_key,
        default_flat_fee=float(settings_data.get("default_flat_fee", 5.0)),
        default_fee_per_km=float(settings_data.get("default_fee_per_km", 0.5)),
        existing_members=existing_members,
    )
    result = wizard.exec()
    if result != QDialog.DialogCode.Accepted:
        raise OnboardingAborted()

    data = wizard.result_data
    updated_api_key = data.api_key or api_key

    settings_manager.update(
        {
            "default_flat_fee": data.default_flat_fee,
            "default_fee_per_km": data.default_fee_per_km,
            "google_maps_api_key": updated_api_key,
            "onboarding": {
                "completed": True,
                "completed_at": datetime.utcnow().isoformat(timespec="seconds"),
            },
        }
    )

    existing_lower = {member.lower() for member in existing_members}
    for name in data.team_members:
        normalized = name.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in existing_lower:
            continue
        try:
            db_manager.add_team_member(normalized, True)
        except sqlite3.IntegrityError:
            continue
        existing_lower.add(key)

    return updated_api_key
