"""
Render a DeepCos report dict as Markdown or plain text.

The same report object that the React dashboard renders can also be downloaded
as a human-readable document - useful for a viva demo, for attaching an analysis
to a project report, or for printing.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from ml.rules.profile import band

BAR_CHARS = 20


def _bar(fraction: float, width: int = BAR_CHARS) -> str:
    """Unicode progress bar used in the text export."""
    filled = int(round(max(0.0, min(1.0, float(fraction))) * width))
    return "\u2588" * filled + "\u2591" * (width - filled)


def _wrap(text: str, width: int) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def to_markdown(report: Mapping) -> str:
    """Render the report as Markdown."""
    product = report.get("product", {})
    lines: list[str] = []
    lines.append(f"# DeepCos analysis report - {report.get('analysis_id', '')}")
    lines.append("")
    lines.append(f"*Generated {report.get('created_at', '')} - mode: {report.get('mode', 'text')}*")
    lines.append("")
    lines.append("## Product identification")
    lines.append("")
    lines.append(f"- **Predicted category:** {product.get('category', 'n/a')}")
    confidence = product.get("category_confidence")
    if isinstance(confidence, (int, float)):
        lines.append(f"- **Category confidence:** {float(confidence):.0%}")
    lines.append(
        f"- **Ingredients analysed:** {report.get('input', {}).get('ingredient_count', 'n/a')}"
    )
    lines.append(f"- **Summary:** {report.get('profile_summary', '')}")
    lines.append("")

    lines.append("## Formulation profile (model prediction)")
    lines.append("")
    lines.append("| Characteristic | Band | Score | Distribution |")
    lines.append("|---|---|---|---|")
    for entry in report.get("profile", []):
        lines.append(
            f"| {entry.get('label')} | {entry.get('band')} | {float(entry.get('score', 0)):.2f} | "
            f"`{_bar(entry.get('score', 0))}` |"
        )
    lines.append("")
    lines.append(
        "> These values are outputs of the trained network, not medical claims about the product."
    )
    lines.append("")

    lines.append("## Why - what moved the prediction")
    lines.append("")
    top = report.get("explainability", {}).get("top_contributors", {})
    for entry in report.get("profile", []):
        key = entry.get("key")
        drivers = [item["ingredient"] for item in top.get(key, {}).get("positive", [])][:4]
        if drivers:
            lines.append(f"- **{entry.get('label')}:** " + ", ".join(drivers))
    lines.append("")

    lines.append("## Key ingredients")
    lines.append("")
    for item in report.get("key_ingredients", []):
        lines.append(
            f"- **{item.get('ingredient')}** - "
            f"{item.get('primary_function') or 'function not in reference data'}"
            + (f" - {item['note']}" if item.get("note") else "")
        )
    lines.append("")

    concerns = report.get("concerns", {})
    lines.append("## Potential concerns (documented characteristics)")
    lines.append("")
    lines.append(concerns.get("summary", ""))
    lines.append("")
    for finding in concerns.get("findings", []):
        severity = str(finding.get("severity", "")).upper()
        lines.append(
            f"- **[{severity}] {finding.get('label')}:** "
            + ", ".join(finding.get("ingredients", []))
        )
        if finding.get("message"):
            lines.append(f"  - {finding['message']}")
    if concerns.get("regulatory_notes"):
        lines.append("")
        lines.append("### Regulatory notes")
        lines.append("")
        for note in concerns["regulatory_notes"]:
            lines.append(
                f"- **{note.get('ingredient')}** ({note.get('status')}): {note.get('note')}"
            )
    lines.append("")
    lines.append("## Ingredient assessment")
    lines.append("")
    lines.append(concerns.get("ingredient_assessment", ""))
    lines.append("")

    lines.append("## Full ingredient list")
    lines.append("")
    lines.append("| # | Ingredient | Function | Notes |")
    lines.append("|---|---|---|---|")
    for row in report.get("ingredients", []):
        lines.append(
            f"| {row.get('position')} | {row.get('ingredient')} | "
            f"{', '.join(row.get('function_labels', [])) or 'n/a'} | {row.get('note', '')} |"
        )
    lines.append("")

    if report.get("warnings"):
        lines.append("## Warnings")
        lines.append("")
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")

    if report.get("ocr"):
        ocr = report["ocr"]
        lines.append("## OCR detail")
        lines.append("")
        lines.append(f"- Engine: {ocr.get('engine')} (available: {ocr.get('available')})")
        lines.append(f"- Preprocessing variant selected: {ocr.get('variant')}")
        lines.append(f"- Mean word confidence: {ocr.get('confidence')}")
        lines.append(f"- Recognised ingredient tokens: {ocr.get('ingredient_count')}")
        lines.append("")

    lines.append("## Disclaimer")
    lines.append("")
    lines.append(report.get("disclaimer", ""))
    lines.append("")
    return "\n".join(lines)


def to_text(report: Mapping) -> str:
    """Render the report as the ASCII box used in the DeepCos design mock-up."""
    product = report.get("product", {})
    width = 62
    out: list[str] = []

    def border(char: str = "=") -> str:
        return "+" + char * width + "+"

    def row(text: str = "") -> str:
        return "| " + str(text).ljust(width - 2)[: width - 2] + " |"

    out.append(border())
    out.append(row("DEEPCOS".center(width - 2)))
    out.append(row("Cosmetic Intelligence Report".center(width - 2)))
    out.append(border())
    out.append(row("PRODUCT"))
    out.append(row(f"  Category   : {product.get('category', 'n/a')}"))
    out.append(row(f"  Confidence : {float(product.get('category_confidence') or 0):.0%}"))
    out.append(row(f"  Ingredients: {report.get('input', {}).get('ingredient_count')}"))
    out.append(row())
    out.append(border("-"))
    out.append(row("FORMULATION PROFILE"))
    for entry in report.get("profile", []):
        out.append(row(f"  {str(entry.get('label')):<18} {str(entry.get('band')).upper()}"))
        out.append(row(f"  {_bar(entry.get('score', 0), width - 6)}"))
    out.append(row())
    out.append(border("-"))
    out.append(row("KEY INGREDIENTS"))
    for item in report.get("key_ingredients", [])[:6]:
        out.append(row(f"  {item.get('ingredient')}"))
        out.append(row(f"  -> {item.get('primary_function') or 'n/a'}"))
    out.append(row())
    out.append(border("-"))
    out.append(row("POTENTIAL CONCERNS"))
    for finding in report.get("concerns", {}).get("findings", [])[:6]:
        out.append(row(f"  ! {finding.get('label')}"))
        out.append(row(f"    {', '.join(finding.get('ingredients', []))[: width - 8]}"))
    out.append(row())
    out.append(border("-"))
    out.append(row("INGREDIENT ASSESSMENT"))
    for line in _wrap(report.get("concerns", {}).get("ingredient_assessment", ""), width - 4):
        out.append(row("  " + line))
    out.append(border())
    return "\n".join(out)


def format_report(report: Mapping, fmt: str = "markdown") -> str:
    """``fmt`` is either ``markdown`` (default) or ``text``."""
    return to_text(report) if fmt == "text" else to_markdown(report)


def summary_row(report: Mapping) -> dict:
    """Compact row used for the analysis-history listing in the UI."""
    scores = report.get("profile_scores", {})
    return {
        "analysis_id": report.get("analysis_id"),
        "created_at": report.get("created_at"),
        "mode": report.get("mode"),
        "category": report.get("product", {}).get("category"),
        "category_confidence": report.get("product", {}).get("category_confidence"),
        "ingredient_count": report.get("input", {}).get("ingredient_count"),
        "summary": report.get("profile_summary"),
        "scores": {target: scores.get(target) for target in scores},
        "bands": {target: band(float(scores.get(target, 0.0))) for target in scores},
        "highest_concern": report.get("concerns", {}).get("highest_severity_label"),
        "concern_count": len(report.get("concerns", {}).get("findings", [])),
    }


def summary_rows(reports: Sequence[Mapping]) -> list[dict]:
    return [summary_row(report) for report in reports]

