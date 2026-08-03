#!/usr/bin/env python3
"""
PD Report Structured Extraction Pipeline
==========================================
Extracts a PD (Personal Discussion / Verification) report PDF into:

  1. Images     -> saved page-wise into an output folder.
  2. Structured JSON -> one JSON object with a fixed set of known sections,
                         each holding its label/value fields (or, for
                         pdRemarks, free text).
  3. Markdown   -> a clean, readable .md rendering of the same structured
                    data, ideal for feeding to an LLM.

How it works
------------
The known section headings (e.g. "Personal Details", "Residence Details",
etc.) appear as plain text directly ABOVE their own bordered table on the
page. This script:

  1. Uses pdfplumber to get each table's bounding box (top y-coordinate)
     per page, via page.find_tables().
  2. Uses pdfplumber's word-level extraction (page.extract_words()) to
     find lines of text that match one of the known headings, and their
     y-position.
  3. Matches each heading to the *next* table that appears below it
     (smallest positive vertical gap), across page boundaries if needed
     (a heading on the bottom of a page can belong to a table on the
     next page — handled by processing all pages first, then matching
     headings to tables in overall document order).
  4. For most sections: each table row becomes one {label: value} pair
     (first cell = label, remaining cells joined = value).
  5. For "Income Details" / "Expense Details" style 2-col, 1-row tables:
     handled the same way (single label/value pair) automatically.
  6. For "pdRemarks": the table's rows are treated as free text and
     concatenated (not label/value), since it's a remarks blob.

Library used: pdfplumber (best for this — gives both word positions AND
table bounding boxes needed to correlate headings -> tables).
Images: PyMuPDF (fitz) — reliable page-wise raster image extraction.

Usage:
    python pd_extract.py input.pdf --outdir ./output

Output:
    output/
      images/page_1_img_1.png ...
      structured_data.json
      structured_report.md
"""

import argparse
import json
import os
import re
import sys

import pdfplumber
import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Config: the known section headings, in the canonical order you specified.
# Matching is case-insensitive and whitespace-tolerant so minor PDF text
# extraction quirks (extra spaces, etc.) don't break detection.
# ---------------------------------------------------------------------------
KNOWN_HEADINGS = [
    "Personal Details",
    "Residence Details",
    "Employment Information",
    "Income Details",
    "Expense Details",
    "Cash Flow Details",
    "Other Details",
    "PD Done By",
    "pdRemarks",
]

# Sections whose table rows should be treated as free text (concatenated),
# not label/value pairs.
FREE_TEXT_SECTIONS = {"pdremarks"}

# Headings that appear as the FIRST CELL of their own table (not as a
# standalone text line above the table). For these, we scan every table's
# first row / first cell for a match, rather than scanning free text.
IN_TABLE_HEADINGS = {"pdremarks"}


def normalize(s: str) -> str:
    """
    Lowercase and strip ALL whitespace (not just collapse it) so headings
    match regardless of spacing/casing quirks in the PDF's text layer,
    e.g. "pdRemarks", "PD Remarks", "pd  remarks" all normalize to
    "pdremarks".
    """
    return re.sub(r"\s+", "", s or "").strip().lower()


HEADING_LOOKUP = {normalize(h): h for h in KNOWN_HEADINGS}


# ---------------------------------------------------------------------------
# Image extraction (page-wise) — same approach as the general pipeline.
# ---------------------------------------------------------------------------
def extract_images(pdf_path: str, images_dir: str) -> dict:
    os.makedirs(images_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    page_image_map = {}

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_num = page_index + 1
        image_list = page.get_images(full=True)
        if not image_list:
            continue

        saved_files = []
        for img_idx, img in enumerate(image_list, start=1):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                filename = f"page_{page_num}_img_{img_idx}.png"
                filepath = os.path.join(images_dir, filename)
                pix.save(filepath)
                saved_files.append(filename)
            except Exception as e:
                print(f"  [warn] failed to extract image {img_idx} on page {page_num}: {e}")

        if saved_files:
            page_image_map[page_num] = saved_files
            print(f"  Page {page_num}: saved {len(saved_files)} image(s)")

    doc.close()
    return page_image_map


# ---------------------------------------------------------------------------
# Heading + table detection
# ---------------------------------------------------------------------------
def find_headings_on_page(page) -> list:
    """
    Returns a list of (heading_text, top_y) tuples for lines on this page
    that match one of KNOWN_HEADINGS. Uses extract_words + line grouping
    so multi-word headings (e.g. "Personal Details") are reassembled.
    """
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    if not words:
        return []

    # Group words into lines by their 'top' coordinate (rounded, to absorb
    # tiny sub-pixel differences between words on the same visual line).
    lines = {}
    for w in words:
        key = round(w["top"], 1)
        lines.setdefault(key, []).append(w)

    found = []
    for top, ws in lines.items():
        ws_sorted = sorted(ws, key=lambda w: w["x0"])
        line_text = " ".join(w["text"] for w in ws_sorted)
        norm = normalize(line_text)
        # Skip headings that live inside a table cell instead (e.g. pdRemarks) —
        # those are handled separately in find_in_table_heading_sections().
        if norm in HEADING_LOOKUP and norm not in IN_TABLE_HEADINGS:
            found.append((HEADING_LOOKUP[norm], top))

    found.sort(key=lambda t: t[1])
    return found


def extract_tables_with_position(page) -> list:
    """
    Returns list of dicts: {"top": y, "rows": [[cell,...], ...]}
    sorted by vertical position on the page.
    """
    out = []
    for t in page.find_tables():
        data = t.extract()
        if not data:
            continue
        cleaned_rows = []
        for row in data:
            cleaned_rows.append([
                " ".join(str(c).split()) if c is not None else "" for c in row
            ])
        out.append({"top": t.bbox[1], "rows": cleaned_rows})
    out.sort(key=lambda d: d["top"])
    return out


def rows_to_fields(rows: list) -> dict:
    """
    Convert table rows into an ordered dict of label -> value.
    First cell = label, remaining cells joined by ' ' = value.
    Skips fully-empty rows.
    """
    fields = {}
    for row in rows:
        if not row or all(c == "" for c in row):
            continue
        label = row[0].strip()
        value_cells = [c for c in row[1:] if c != ""]
        value = " ".join(value_cells).strip()
        if label:
            fields[label] = value
    return fields


def rows_to_text(rows: list) -> str:
    """Flatten all non-empty cells across rows into one text blob."""
    parts = []
    for row in rows:
        for cell in row:
            if cell and cell.strip():
                parts.append(cell.strip())
    return " ".join(parts).strip()


def check_table_for_in_table_heading(rows: list):
    """
    Checks if this table's own first row / first non-empty cell is itself
    one of the IN_TABLE_HEADINGS (e.g. "pdRemarks"). If so, returns the
    canonical heading name; otherwise returns None.

    Handles both shapes described:
      Row 1: ["pdRemarks", ""]      <- label alone in its own row
      Row 2: ["<remarks text>"]     <- remarks text below
    """
    if not rows:
        return None
    first_row = rows[0]
    first_cell = next((c.strip() for c in first_row if c and c.strip()), "")
    norm = normalize(first_cell)
    if norm in HEADING_LOOKUP and norm in IN_TABLE_HEADINGS:
        return HEADING_LOOKUP[norm]
    return None


# ---------------------------------------------------------------------------
# Core: build the ordered list of (heading, table) across the whole doc
# ---------------------------------------------------------------------------
def build_structured_data(pdf_path: str) -> dict:
    all_headings = []   # list of (heading, global_order_key)
    all_tables = []      # list of (table_dict, global_order_key, page_num)

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_num = page_index + 1
            page_height = page.height

            headings = find_headings_on_page(page)
            for h_text, top in headings:
                # global key: page index dominates, top position breaks ties
                global_key = page_index * 100000 + top
                all_headings.append((h_text, global_key, page_num))

            tables = extract_tables_with_position(page)
            for t in tables:
                global_key = page_index * 100000 + t["top"]
                all_tables.append((t, global_key, page_num))

    all_headings.sort(key=lambda x: x[1])
    all_tables.sort(key=lambda x: x[1])

    structured = {}

    # --- Pass 1: pull out tables that are IN_TABLE_HEADINGS themselves
    # (e.g. pdRemarks, whose label is the table's own first cell, not a
    # free-text line above it). Remove these tables from all_tables so
    # they aren't mistakenly consumed by the next-table matching in Pass 2.
    remaining_tables = []
    for table_dict, t_key, t_page in all_tables:
        in_table_heading = check_table_for_in_table_heading(table_dict["rows"])
        if in_table_heading:
            # Remaining rows (after the label row) are the free-text content.
            content_rows = table_dict["rows"][1:] if len(table_dict["rows"]) > 1 else table_dict["rows"]
            value = rows_to_text(content_rows)
            structured[in_table_heading] = {"page": t_page, "text": value}
        else:
            remaining_tables.append((table_dict, t_key, t_page))
    all_tables = remaining_tables

    # --- Pass 2: match each remaining (free-text) heading to the next table
    # that comes after it (by global key)
    table_pointer = 0
    for h_text, h_key, h_page in all_headings:
        # advance pointer to first table with key > heading key
        while table_pointer < len(all_tables) and all_tables[table_pointer][1] < h_key:
            table_pointer += 1
        if table_pointer >= len(all_tables):
            # no table found after this heading
            structured[h_text] = {"page": h_page, "data": {}}
            continue

        table_dict, t_key, t_page = all_tables[table_pointer]
        norm_h = normalize(h_text)

        if norm_h in FREE_TEXT_SECTIONS:
            value = rows_to_text(table_dict["rows"])
            structured[h_text] = {"page": t_page, "text": value}
        else:
            fields = rows_to_fields(table_dict["rows"])
            structured[h_text] = {"page": t_page, "data": fields}

        table_pointer += 1  # this table consumed; move to next for next heading

    return structured


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def render_markdown(structured: dict, page_image_map: dict, pdf_name: str) -> str:
    lines = [f"# PD Report — {pdf_name}\n"]

    for heading in KNOWN_HEADINGS:
        if heading not in structured:
            continue
        section = structured[heading]
        lines.append(f"## {heading}\n")

        if "text" in section:
            lines.append(section["text"] if section["text"] else "*(no remarks captured)*")
            lines.append("")
        else:
            data = section.get("data", {})
            if data:
                for label, value in data.items():
                    lines.append(f"- **{label}**: {value}")
            else:
                lines.append("*(no data found for this section)*")
            lines.append("")

    if page_image_map:
        lines.append("## Images\n")
        for page_num in sorted(page_image_map.keys()):
            for fname in page_image_map[page_num]:
                lines.append(f"- Page {page_num}: images/{fname}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract structured PD report data from a PDF.")
    parser.add_argument("pdf_path", help="Path to the input PDF file")
    parser.add_argument("--outdir", default="./output", help="Output directory (default: ./output)")
    args = parser.parse_args()

    if not os.path.isfile(args.pdf_path):
        print(f"Error: file not found: {args.pdf_path}")
        sys.exit(1)

    outdir = args.outdir
    images_dir = os.path.join(outdir, "images")
    json_path = os.path.join(outdir, "structured_data.json")
    md_path = os.path.join(outdir, "structured_report.md")

    os.makedirs(outdir, exist_ok=True)

    print("Step 1/3: Extracting images (page-wise)...")
    page_image_map = extract_images(args.pdf_path, images_dir)

    print("\nStep 2/3: Detecting headings + tables, building structured data...")
    structured = build_structured_data(args.pdf_path)

    for h in KNOWN_HEADINGS:
        status = "found" if h in structured else "MISSING"
        print(f"  - {h}: {status}")

    print("\nStep 3/3: Writing JSON + Markdown outputs...")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)

    pdf_name = os.path.basename(args.pdf_path)
    md_content = render_markdown(structured, page_image_map, pdf_name)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nDone.")
    print(f"  Images -> {images_dir}")
    print(f"  JSON   -> {json_path}")
    print(f"  MD     -> {md_path}")


if __name__ == "__main__":
    main()
