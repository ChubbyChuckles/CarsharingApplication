"""Utilities for exporting RideShare ledger data to professional PDFs."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from math import fsum
from pathlib import Path
from typing import Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - assists static analysis
    from reportlab.lib import colors  # type: ignore
    from reportlab.lib.pagesizes import A4  # type: ignore
    from reportlab.lib.styles import ParagraphStyle, StyleSheet1  # type: ignore
    from reportlab.pdfbase import pdfmetrics  # type: ignore
    from reportlab.platypus import (  # type: ignore
        HRFlowable,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

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
            from reportlab.platypus import (  # type: ignore
                HRFlowable,
                PageBreak,
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
            )
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
            "HRFlowable": HRFlowable,
            "PageBreak": PageBreak,
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
    detailed_entries: Sequence[Mapping[str, object]] | None = None,
    rides: Sequence[Mapping[str, object]] | None = None,
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
    HRFlowable = rl["HRFlowable"]
    PageBreak = rl["PageBreak"]

    _ensure_font_registered(pdfmetrics, TTFont)

    styles = getSampleStyleSheet()
    base_font = "SegoeUI" if "SegoeUI" in pdfmetrics.getRegisteredFontNames() else "Helvetica"

    title_style = ParagraphStyle(
        "LedgerTitle",
        parent=styles["Heading1"],
        fontName=base_font,
        fontSize=22,
        leading=26,
        spaceAfter=4,
        textColor=colors.HexColor("#0f1623"),
    )
    subtitle_style = ParagraphStyle(
        "LedgerSubtitle",
        parent=styles["Normal"],
        fontName=base_font,
        fontSize=11,
        textColor=colors.HexColor("#4e5d78"),
        spaceAfter=12,
    )
    body_style = ParagraphStyle(
        "LedgerBody",
        parent=styles["Normal"],
        fontName=base_font,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1a2739"),
    )
    secondary_style = ParagraphStyle(
        "LedgerSecondary",
        parent=body_style,
        fontSize=9,
        textColor=colors.HexColor("#4e5d78"),
    )
    section_header_style = ParagraphStyle(
        "LedgerSection",
        parent=styles["Heading2"],
        fontName=base_font,
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#0f1623"),
        spaceBefore=18,
        spaceAfter=6,
    )
    pair_header_style = ParagraphStyle(
        "LedgerPair",
        parent=styles["Heading3"],
        fontName=base_font,
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#112032"),
        spaceBefore=14,
        spaceAfter=4,
    )
    small_caps_style = ParagraphStyle(
        "LedgerSmallCaps",
        parent=secondary_style,
        fontSize=8.5,
        leading=12,
    )

    generated = generated_at or datetime.now()
    detailed_entries = list(detailed_entries or [])
    rides = list(rides or [])

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        topMargin=22 * mm,
        bottomMargin=18 * mm,
        title=title,
    )

    def _scaled_widths(*fractions: float) -> list[float]:
        total = sum(fractions)
        if total == 0:
            raise ValueError("Column width fractions must not sum to zero.")
        return [doc.width * (fraction / total) for fraction in fractions]

    creditor_totals: dict[str, float] = defaultdict(float)
    debtor_totals: dict[str, float] = defaultdict(float)
    for entry in ledger_entries:
        amount = float(entry.get("amount", 0.0) or 0.0)
        creditor_totals[str(entry.get("owed_name", "—"))] += amount
        debtor_totals[str(entry.get("owes_name", "—"))] += amount

    total_outstanding = fsum(float(entry.get("amount", 0.0) or 0.0) for entry in ledger_entries)
    unique_debtors = {str(entry.get("owes_name", "—")) for entry in ledger_entries}
    unique_creditors = {str(entry.get("owed_name", "—")) for entry in ledger_entries}
    all_dates = []
    for entry in detailed_entries:
        iso = entry.get("ride_datetime")
        try:
            all_dates.append(datetime.fromisoformat(str(iso)))
        except ValueError:
            continue
    for ride in rides:
        iso = ride.get("ride_datetime")
        try:
            all_dates.append(datetime.fromisoformat(str(iso)))
        except ValueError:
            continue
    coverage_start = min(all_dates) if all_dates else None
    coverage_end = max(all_dates) if all_dates else None

    def _format_currency(value: float) -> str:
        return f"€{value:,.2f}"

    info_table_data = [
        ["Statement Date", generated.strftime("%d %B %Y %H:%M")],
        [
            "Coverage Window",
            (
                f"{coverage_start.strftime('%d %b %Y')} — {coverage_end.strftime('%d %b %Y')}"
                if coverage_start and coverage_end
                else "No rides recorded"
            ),
        ],
        ["Outstanding Balance", _format_currency(total_outstanding)],
        ["Unique Debtors", str(len(unique_debtors))],
        ["Unique Creditors", str(len(unique_creditors))],
        ["Rides Analysed", str(len(rides))],
    ]
    info_table = Table(info_table_data, colWidths=_scaled_widths(0.28, 0.72))
    info_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), base_font),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#4e5d78")),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("ALIGN", (1, 0), (-1, -1), "LEFT"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    summary_table_data = [["Debtor", "Creditor", "Net Balance"]]
    for entry in ledger_entries:
        summary_table_data.append(
            [
                str(entry.get("owes_name", "—")),
                str(entry.get("owed_name", "—")),
                _format_currency(float(entry.get("amount", 0.0) or 0.0)),
            ]
        )

    summary_table = Table(summary_table_data, colWidths=_scaled_widths(0.4, 0.35, 0.25))
    summary_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), base_font),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f1623")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("TEXTCOLOR", (0, 1), (-2, -1), colors.HexColor("#1a2739")),
                ("TEXTCOLOR", (-1, 1), (-1, -1), colors.HexColor("#0f3c4c")),
                ("ALIGN", (0, 0), (-2, -1), "LEFT"),
                ("ALIGN", (-1, 1), (-1, -1), "RIGHT"),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.HexColor("#f5f8ff"),
                        colors.HexColor("#eaf1ff"),
                    ],
                ),
                ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#cad8f4")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d7e1f5")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    creditor_table_data = [["Creditor", "Total Owed"]]
    for name, total in sorted(creditor_totals.items(), key=lambda item: -item[1]):
        creditor_table_data.append([name, _format_currency(total)])
    creditor_table = Table(creditor_table_data, colWidths=_scaled_widths(0.55, 0.45))
    creditor_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), base_font),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f1623")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (-1, 1), (-1, -1), "RIGHT"),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.HexColor("#f5f8ff"), colors.HexColor("#eaf1ff")],
                ),
                ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#cad8f4")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d7e1f5")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    debtor_table_data = [["Debtor", "Total Owed"]]
    for name, total in sorted(debtor_totals.items(), key=lambda item: -item[1]):
        debtor_table_data.append([name, _format_currency(total)])
    debtor_table = Table(debtor_table_data, colWidths=_scaled_widths(0.55, 0.45))
    debtor_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), base_font),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f1623")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (-1, 1), (-1, -1), "RIGHT"),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.HexColor("#f5f8ff"), colors.HexColor("#eaf1ff")],
                ),
                ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#cad8f4")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d7e1f5")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story: list = [
        Paragraph(title, title_style),
        Paragraph(subtitle, subtitle_style),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#35c4c7")),
        Spacer(1, 12),
        info_table,
        Spacer(1, 10),
        Paragraph("Outstanding Balances", section_header_style),
        summary_table,
        Spacer(1, 12),
        Paragraph(
            "Amounts listed represent net obligations after reconciling every recorded ride. Positive balances indicate what the debtor must reimburse the creditor to settle all trips to date.",
            secondary_style,
        ),
        Spacer(1, 6),
        Paragraph("Cumulative Balances", section_header_style),
        Paragraph("What each creditor is still owed.", secondary_style),
        creditor_table,
        Spacer(1, 10),
        Paragraph("What each debtor still owes.", secondary_style),
        debtor_table,
    ]

    if detailed_entries:
        story.extend([PageBreak(), Paragraph("Detailed Settlement Ledger", section_header_style)])
        story.append(
            Paragraph(
                "Every line item below traces the exact ride, route, and tariff that contributed to the balance. Use this as an authoritative receipt for reimbursements.",
                secondary_style,
            )
        )

        rides_by_id = {int(ride.get("id")): ride for ride in rides}
        grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
        for entry in detailed_entries:
            key = (
                str(entry.get("passenger_name", "—")),
                str(entry.get("driver_name", "—")),
            )
            grouped[key].append(entry)

        sorted_groups = sorted(
            grouped.items(),
            key=lambda item: -fsum(float(e.get("amount", 0.0) or 0.0) for e in item[1]),
        )

        for index, ((passenger_name, driver_name), entries) in enumerate(sorted_groups, start=1):
            pair_total = fsum(float(e.get("amount", 0.0) or 0.0) for e in entries)
            story.append(Spacer(1, 8))
            story.append(
                Paragraph(
                    f"{index}. <b>{passenger_name}</b> reimburses <b>{driver_name}</b>",
                    pair_header_style,
                )
            )
            story.append(
                Paragraph(
                    f"Outstanding balance: <b>{_format_currency(pair_total)}</b>",
                    secondary_style,
                )
            )

            detail_data = [
                [
                    "Ride Date",
                    "Route & Distance",
                    "Ride Cost",
                    "Share",
                    "Tariff",
                    "Vehicle & Riders",
                ]
            ]

            for detail in entries:
                ride_dt_text = "—"
                ride_dt = None
                iso = detail.get("ride_datetime")
                if iso:
                    try:
                        ride_dt = datetime.fromisoformat(str(iso))
                        ride_dt_text = ride_dt.strftime("%d %b %Y %H:%M")
                    except ValueError:
                        ride_dt_text = str(iso)

                ride = rides_by_id.get(int(detail.get("ride_id", 0)))
                distance = float(detail.get("distance_km", 0.0) or 0.0)
                route = (
                    f"{detail.get('start_address', '—')} → {detail.get('destination_address', '—')}"
                )
                distance_line = f"{distance:.2f} km round trip"
                route_paragraph = Paragraph(
                    f"{route}<br/><font size=8 color='#4e5d78'>{distance_line}</font>",
                    body_style,
                )

                vehicle_info = "—"
                passengers_info = "—"
                if ride:
                    drivers = ride.get("drivers", []) or [driver_name]
                    passengers = ride.get("passengers", [])
                    vehicle_info = "<br/>".join(drivers) if drivers else "—"
                    passengers_info = "<br/>".join(passengers) if passengers else "—"
                tariff = (
                    f"Flat €{float(detail.get('flat_fee', 0.0)):.2f} per vehicle"
                    f"<br/>Distance €{float(detail.get('fee_per_km', 0.0)):.2f}/km"
                )
                rider_block = Paragraph(
                    f"<b>Drivers</b><br/>{vehicle_info}<br/><br/><b>Passengers</b><br/>{passengers_info}",
                    small_caps_style,
                )
                if ride_dt:
                    date_text = ride_dt.strftime("%d %b %Y")
                    time_text = ride_dt.strftime("%H:%M")
                    ride_dt_display = Paragraph(
                        f"{date_text}<br/><font size=8 color='#4e5d78'>{time_text}</font>",
                        body_style,
                    )
                else:
                    ride_dt_display = Paragraph(str(ride_dt_text), body_style)

                detail_data.append(
                    [
                        ride_dt_display,
                        route_paragraph,
                        _format_currency(float(detail.get("total_cost", 0.0) or 0.0)),
                        _format_currency(float(detail.get("amount", 0.0) or 0.0)),
                        Paragraph(tariff, small_caps_style),
                        rider_block,
                    ]
                )

            detail_table = Table(
                detail_data,
                colWidths=_scaled_widths(0.13, 0.23, 0.14, 0.14, 0.18, 0.18),
                repeatRows=1,
            )
            detail_table.setStyle(
                TableStyle(
                    [
                        ("FONTNAME", (0, 0), (-1, -1), base_font),
                        ("FONTSIZE", (0, 0), (-1, 0), 9.5),
                        ("FONTSIZE", (0, 1), (-1, -1), 9),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#10243a")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("ALIGN", (0, 0), (0, -1), "LEFT"),
                        ("ALIGN", (2, 1), (3, -1), "RIGHT"),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [
                                colors.HexColor("#f5f8ff"),
                                colors.HexColor("#eef3ff"),
                            ],
                        ),
                        ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#cad8f4")),
                        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d7e1f5")),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            story.append(detail_table)
            story.append(HRFlowable(width="95%", thickness=0.6, color=colors.HexColor("#d7e1f5")))

    if rides:
        story.extend([PageBreak(), Paragraph("Ride Cost Breakdown", section_header_style)])
        story.append(
            Paragraph(
                "Complete ledger of every recorded trip, including tariff configuration and participant lists.",
                secondary_style,
            )
        )

        ride_table_data = [
            [
                "Ride Date",
                "Route",
                "Drivers",
                "Passengers",
                "Distance (km)",
                "Total (€)",
                "Per Core Rider (€)",
            ]
        ]

        for ride in sorted(rides, key=lambda r: r.get("ride_datetime") or "", reverse=True):
            try:
                ride_dt = datetime.fromisoformat(str(ride.get("ride_datetime")))
                ride_dt_display = Paragraph(
                    f"{ride_dt.strftime('%d %b %Y')}<br/><font size=8 color='#4e5d78'>{ride_dt.strftime('%H:%M')}</font>",
                    body_style,
                )
            except ValueError:
                ride_dt_display = Paragraph(str(ride.get("ride_datetime", "—")), body_style)
            route = Paragraph(
                f"{ride.get('start_address', '—')} → {ride.get('destination_address', '—')}",
                body_style,
            )
            drivers_para = Paragraph("<br/>".join(ride.get("drivers", [])) or "—", small_caps_style)
            passengers_para = Paragraph(
                "<br/>".join(ride.get("passengers", [])) or "—", small_caps_style
            )
            ride_table_data.append(
                [
                    ride_dt_display,
                    route,
                    drivers_para,
                    passengers_para,
                    f"{float(ride.get('distance_km', 0.0) or 0.0):.2f}",
                    _format_currency(float(ride.get("total_cost", 0.0) or 0.0)),
                    _format_currency(float(ride.get("cost_per_passenger", 0.0) or 0.0)),
                ]
            )

        rides_table = Table(
            ride_table_data,
            colWidths=_scaled_widths(0.13, 0.2, 0.15, 0.15, 0.1, 0.13, 0.14),
            repeatRows=1,
        )
        rides_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), base_font),
                    ("FONTSIZE", (0, 0), (-1, 0), 9.5),
                    ("FONTSIZE", (0, 1), (-1, -1), 9),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f1623")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [
                            colors.HexColor("#f5f8ff"),
                            colors.HexColor("#eef3ff"),
                        ],
                    ),
                    ("ALIGN", (4, 1), (-1, -1), "RIGHT"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#cad8f4")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d7e1f5")),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(Spacer(1, 10))
        story.append(rides_table)

    story.append(Spacer(1, 16))
    story.append(
        Paragraph(
            "Please settle outstanding balances within seven days of receiving this statement. For questions or adjustments, contact the ride coordinator.",
            secondary_style,
        )
    )

    accent_color = colors.HexColor("#0f1623")

    def _draw_header_footer(canvas, doc) -> None:  # pragma: no cover - rendering only
        canvas.saveState()
        canvas.setFillColor(accent_color)
        canvas.rect(
            doc.leftMargin,
            doc.height + doc.topMargin - 14,
            doc.width,
            0.8,
            stroke=0,
            fill=1,
        )
        canvas.rect(
            doc.leftMargin,
            doc.bottomMargin - 18,
            doc.width,
            0.6,
            stroke=0,
            fill=1,
        )
        canvas.setFont(base_font, 9)
        canvas.setFillColor(colors.white)
        canvas.drawString(
            doc.leftMargin, doc.height + doc.topMargin - 10, "Table Tennis RideShare Manager"
        )
        canvas.drawRightString(
            doc.leftMargin + doc.width,
            doc.bottomMargin - 12,
            f"Page {doc.page}",
        )
        canvas.setFillColor(colors.HexColor("#4e5d78"))
        canvas.drawString(
            doc.leftMargin,
            doc.bottomMargin - 12,
            f"Generated {generated.strftime('%d %b %Y %H:%M')}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=_draw_header_footer, onLaterPages=_draw_header_footer)
    return path


__all__ = ["export_ledger_pdf"]
