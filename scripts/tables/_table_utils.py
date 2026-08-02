"""Shared utilities for MAGNETO manuscript and supplementary tables.

All table scripts should build tables only from existing result files and must
not rerun analytical computations.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def project_root() -> Path:
    """Return the repository root (scripts/tables -> ../..)."""
    return Path(__file__).resolve().parents[2]


def tables_dir() -> Path:
    """Return reports/tables, creating it if necessary."""
    path = project_root() / "reports" / "tables"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fix_negative_zero(x: float | int | None, decimals: int = 3) -> float | int | None:
    """Treat values that round to zero as positive zero."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return x
    if isinstance(x, (int, float)):
        threshold = 0.5 * 10 ** (-decimals)
        if abs(float(x)) < threshold:
            return 0
    return x


def fmt_number(x: float | None, decimals: int = 3) -> str:
    """Format a float with fixed decimals and Unicode minus."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    x = _fix_negative_zero(x, decimals)
    s = f"{float(x):.{decimals}f}"
    return s.replace("-", "\u2212")


def fmt_int(x: int | float | None) -> str:
    """Format an integer with thousands separator."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    x = _fix_negative_zero(x, decimals=0)
    return f"{int(x):,}"


def fmt_ci(low: float | None, high: float | None, decimals: int = 3) -> str:
    """Format a confidence interval using an en dash."""
    if low is None or high is None or (isinstance(low, float) and pd.isna(low)):
        return ""
    return f"[{fmt_number(low, decimals)}, {fmt_number(high, decimals)}]"


def fmt_p(p: float | None, empirical: bool = True) -> str:
    """Format a p-value.

    Empirical surrogate p-values with B=1000 are floored at 0.001.
    Nominal p-values may be reported as <0.001 but must be labelled separately.
    """
    if p is None or (isinstance(p, float) and pd.isna(p)):
        return ""
    if empirical:
        if p < 0.001:
            return "0.001"
    else:
        if p < 0.001:
            return "<0.001"
    return fmt_number(p, 3)


def fmt_percent(x: float | None, decimals: int = 3) -> str:
    """Format a proportion as percentage points."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return fmt_number(x * 100, decimals)


def fmt_method(method: str | None) -> str:
    """Convert internal method name to manuscript label."""
    if method is None:
        return ""
    mapping = {
        "harmonic": "Harmonic",
        "cyclic_spline": "Cyclic spline",
    }
    return mapping.get(str(method).lower(), str(method).replace("_", " ").title())


def df_to_markdown(df: pd.DataFrame, caption: str, notes: list[str] | None = None) -> str:
    """Render a DataFrame as a Markdown table with caption and notes."""
    lines = [f"**{caption}**", ""]
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines.extend([header, sep])
    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            v = row[c]
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, float):
                vals.append(fmt_number(v))
            elif isinstance(v, int):
                vals.append(fmt_int(v))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    if notes:
        lines.append("")
        for note in notes:
            lines.append(f"* {note}")
    return "\n".join(lines) + "\n"


def _set_cell_border(cell, **kwargs) -> None:
    """Set border properties for a docx table cell."""
    from docx.oxml import parse_xml
    from docx.oxml.ns import qn

    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()

    # remove old borders if present
    for existing in tcPr.findall(qn("w:tcBorders")):
        tcPr.remove(existing)

    tcBorders = parse_xml('<w:tcBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>')
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        if edge in kwargs:
            attrs = kwargs[edge]
            element = parse_xml(
                f'<w:{edge} xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
                f'w:val="{attrs.get("val", "single")}" '
                f'w:sz="{attrs.get("sz", 4)}" '
                f'w:space="{attrs.get("space", 0)}" '
                f'w:color="{attrs.get("color", "000000")}"/>'
            )
            tcBorders.append(element)
    tcPr.append(tcBorders)


def _format_docx_text(run, font_name: str = "Times New Roman", font_size: int = 9) -> None:
    """Apply manuscript font to a docx run."""
    from docx.shared import Pt

    run.font.name = font_name
    run.font.size = Pt(font_size)


def df_to_docx(
    df: pd.DataFrame,
    caption: str,
    notes: list[str] | None = None,
    landscape: bool = False,
    font_name: str = "Times New Roman",
    font_size: int = 9,
) -> "docx.Document":
    """Create a submission-ready DOCX table with three-line style."""
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.oxml import parse_xml
    from docx.shared import Inches, Pt

    doc = Document()
    section = doc.sections[0]
    if landscape:
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width, section.page_height = section.page_height, section.page_width

    # Caption
    cap = doc.add_paragraph()
    cap_run = cap.add_run(caption)
    _format_docx_text(cap_run, font_name, font_size)
    cap.paragraph_format.space_after = Pt(6)

    # Table
    table = doc.add_table(rows=1, cols=len(df.columns))
    table.autofit = False
    table.allow_autofit = False

    # Header row
    hdr_cells = table.rows[0].cells
    for i, col in enumerate(df.columns):
        hdr_cells[i].text = str(col)
        for paragraph in hdr_cells[i].paragraphs:
            paragraph.paragraph_format.alignment = 1  # center
            for run in paragraph.runs:
                _format_docx_text(run, font_name, font_size)
                run.font.bold = True
        _set_cell_border(hdr_cells[i], top={"sz": 12}, bottom={"sz": 8}, left={"val": "nil"}, right={"val": "nil"})

    # Data rows
    for _, row in df.iterrows():
        row_cells = table.add_row().cells
        for i, col in enumerate(df.columns):
            v = row[col]
            if pd.isna(v):
                cell_text = ""
            elif isinstance(v, float):
                cell_text = fmt_number(v)
            elif isinstance(v, int):
                cell_text = fmt_int(v)
            else:
                cell_text = str(v)
            row_cells[i].text = cell_text
            for paragraph in row_cells[i].paragraphs:
                for run in paragraph.runs:
                    _format_docx_text(run, font_name, font_size)
        for cell in row_cells:
            _set_cell_border(cell, left={"val": "nil"}, right={"val": "nil"})

    # Bottom border on last row
    for cell in table.rows[-1].cells:
        _set_cell_border(cell, bottom={"sz": 12}, left={"val": "nil"}, right={"val": "nil"})

    # Repeat header rows across pages (tblHeader) and prevent row splitting.
    from docx.oxml.ns import qn
    for row in table.rows:
        trPr = row._tr.get_or_add_trPr()
        cant_split = parse_xml(
            r'<w:cantSplit xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>'
        )
        trPr.append(cant_split)
    trPr = table.rows[0]._tr.get_or_add_trPr()
    tblHeader = parse_xml(r'<w:tblHeader xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>')
    trPr.append(tblHeader)

    # Notes
    if notes:
        doc.add_paragraph()
        for note in notes:
            p = doc.add_paragraph()
            run = p.add_run(f"Note. {note}" if note == notes[0] else note)
            _format_docx_text(run, font_name, font_size)
            p.paragraph_format.space_after = Pt(3)

    return doc


def write_table(
    df: pd.DataFrame,
    name: str,
    caption: str,
    notes: list[str] | None = None,
    landscape: bool = False,
    full_df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Write source CSV, Markdown, and DOCX for one table.

    If ``full_df`` is supplied, it is written as the source CSV while ``df`` is
    used for the Markdown and DOCX rendering. This lets a table script keep a
    compact manuscript view while archiving the complete numerical source.
    """
    out_dir = tables_dir()
    csv_path = out_dir / f"{name}_source.csv"
    md_path = out_dir / f"{name}.md"
    docx_path = out_dir / f"{name}.docx"

    (full_df if full_df is not None else df).to_csv(csv_path, index=False)
    md_path.write_text(df_to_markdown(df, caption, notes), encoding="utf-8")
    doc = df_to_docx(df, caption, notes, landscape=landscape)
    doc.save(docx_path)

    return {"csv": csv_path, "md": md_path, "docx": docx_path}


def load_json(path: Path) -> Any:
    """Load a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file, failing clearly if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required input table not found: {path}")
    return pd.read_csv(path)


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file, failing clearly if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required input table not found: {path}")
    return pd.read_parquet(path)


def scenario_display_label(scenario_id: str) -> str:
    """Return the canonical display label for a scenario_id from the registry."""
    reg_path = project_root() / "config" / "supplementary_checks.yaml"
    with open(reg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for entry in cfg.get("scenario_registry", []):
        if entry.get("scenario_id") == scenario_id:
            return entry.get("display_label", scenario_id)
    return scenario_id.replace("_", " ").title()
