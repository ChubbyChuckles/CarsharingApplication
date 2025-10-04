"""Utilities for exporting RideShare ledger data to professional PDFs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - assists static analysis
    from reportlab.lib import colors  # type: ignore
    from reportlab.lib.pagesizes import A4  # type: ignore
    from reportlab.lib.styles import ParagraphStyle, StyleSheet1  # type: ignore
    from reportlab.pdfbase import pdfmetrics  # type: ignore
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # type: ignore

DEFAULT_TITLE = "Table Tennis RideShare Ledger"
DEFAULT_SUBTITLE = "Outstanding balances between team members"

_REPORTLAB_CACHE: dict[str, object] | None = None


def _load_reportlab() -> dict[str, object]:
    global _REPORTLAB_CACHE
    if _REPORTLAB_CACHE is None:
        try:
            from reportlab.lib import colors  # type: ignore
            from reportlab.lib.pagesizes import A4  # type: ignore
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # type: ignore
            from reportlab.lib.units import mm  # type: ignore
            from reportlab.pdfbase import pdfmetrics  # type: ignore
            from reportlab.pdfbase.ttfonts import TTFont  # type: ignore
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "ReportLab is required for PDF export. Install it with 'pip install reportlab'."
            ) from exc
        _REPORTLAB_CACHE = {
            "colors": colors,
            "A4": A4,
            "ParagraphStyle": ParagraphStyle,
            "getSampleStyleSheet": getSampleStyleSheet,
            "mm": mm,
            "pdfmetrics": pdfmetrics,
            "TTFont": TTFont,
            "Paragraph": Paragraph,
            "SimpleDocTemplate": SimpleDocTemplate,
            "Spacer": Spacer,
            "Table": Table,
            "TableStyle": TableStyle,
        }
    return _REPORTLAB_CACHE


def _ensure_font_registered(pdfmetrics, TTFont) -> None:
    """Register Segoe UI if present to match the application's typography."""

    try:
        pdfmetrics.getFont("SegoeUI")
    except KeyError:
        # Attempt to register Segoe UI; fall back silently if unavailable
        candidate_paths = [
            "C:/Windows/Fonts/segoeui.ttf",
            "/System/Library/Fonts/Segoe UI.ttf",
        ]
        for path in candidate_paths:
            font_path = Path(path)
            if font_path.exists():
                pdfmetrics.registerFont(TTFont("SegoeUI", str(font_path)))
                break


def export_ledger_pdf(
    output_path: Path | str,
    ledger_entries: Sequence[Mapping[str, object]],
    *,
    generated_at: datetime | None = None,
    title: str = DEFAULT_TITLE,
    subtitle: str = DEFAULT_SUBTITLE,
) -> Path:
    """Render the ledger into a polished PDF and return the written path."""

    if not ledger_entries:
        raise ValueError("No ledger entries supplied for export.")

    path = Path(output_path)
    if path.suffix.lower() != ".pdf":
        path = path.with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)

    rl = _load_reportlab()
    colors = rl["colors"]
    A4 = rl["A4"]
    ParagraphStyle = rl["ParagraphStyle"]
    getSampleStyleSheet = rl["getSampleStyleSheet"]
    mm = rl["mm"]
    pdfmetrics = rl["pdfmetrics"]
    TTFont = rl["TTFont"]
    Paragraph = rl["Paragraph"]
    SimpleDocTemplate = rl["SimpleDocTemplate"]
    Spacer = rl["Spacer"]
    Table = rl["Table"]
    TableStyle = rl["TableStyle"]

    _ensure_font_registered(pdfmetrics, TTFont)

    styles = getSampleStyleSheet()
    base_font = "SegoeUI" if "SegoeUI" in pdfmetrics.getRegisteredFontNames() else "Helvetica"

    title_style = ParagraphStyle(
        "LedgerTitle",
        parent=styles["Heading1"],
        fontName=base_font,
        fontSize=20,
        leading=24,
        spaceAfter=6,
        textColor=colors.HexColor("#0f1623"),
    )
    subtitle_style = ParagraphStyle(
        "LedgerSubtitle",
        parent=styles["Normal"],
        fontName=base_font,
        fontSize=11,
        textColor=colors.HexColor("#4e5d78"),
        spaceAfter=14,
    )
    body_style = ParagraphStyle(
        "LedgerBody",
        parent=styles["Normal"],
        fontName=base_font,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1a2739"),
    )

    generated = generated_at or datetime.now()

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        topMargin=22 * mm,
        bottomMargin=18 * mm,
        title=title,
    )

    data = [["Pays", "Receives", "Net Amount (€)"]]
    total_amount = 0.0
    for entry in ledger_entries:
        owes = str(entry.get("owes_name", "—"))
        owed = str(entry.get("owed_name", "—"))
        amount = float(entry.get("amount", 0.0) or 0.0)
        total_amount += amount
        data.append([owes, owed, f"€{amount:,.2f}"])

    table = Table(data, colWidths=[75 * mm, 75 * mm, 30 * mm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), base_font),
                ("FONTSIZE", (0, 0), (-1, 0), 11),
                ("FONTSIZE", (0, 1), (-1, -1), 10),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f1623")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("ALIGN", (-1, 1), (-1, -1), "RIGHT"),
                ("LINEBELOW", (0, 0), (-1, 0), 1, colors.HexColor("#35c4c7")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.HexColor("#f5f8ff"),
                        colors.HexColor("#eaf1ff"),
                    ],
                ),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cad8f4")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d7e1f5")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    total_paragraph = Paragraph(
        f"Total outstanding amount: <b>€{total_amount:,.2f}</b>",
        body_style,
    )

    story: list = [
        Paragraph(title, title_style),
        Paragraph(subtitle, subtitle_style),
        Paragraph(
            f"Generated on {generated.strftime('%d %B %Y at %H:%M')}",
            body_style,
        ),
        Spacer(1, 16),
        table,
        Spacer(1, 18),
        total_paragraph,
        Spacer(1, 12),
        Paragraph(
            "Balances are calculated based on the current ride history. Positive amounts indicate what the payer owes the recipient.",
            body_style,
        ),
    ]

    doc.build(story)
    return path


__all__ = ["export_ledger_pdf"]
