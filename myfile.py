#!/usr/bin/env python3


import argparse
import json
import os
import re
import sys


from typing import Type

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
import pdfplumber
import fitz  # PyMuPDF
from schemas.structure import PDReport
from main import ImageOCR
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

# pdRemarks is handled completely separately (see extract_pdremarks below) —
# a direct raw-text search for the literal string, not table/heading logic.
STANDARD_HEADINGS = [h for h in KNOWN_HEADINGS if h.strip().lower() != "pdremarks"]


def normalize(s: str) -> str:
    """
    Lowercase and strip ALL whitespace so headings match regardless of
    spacing/casing quirks in the PDF's text layer.
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
        if norm in HEADING_LOOKUP:
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
    Kept as a light-touch backup: if a table's first cell literally equals
    "pdRemarks" (rare shape), treat the rest of that table as its content.
    The primary pdRemarks extraction is extract_pdremarks() below, which
    does a direct raw-text search across the whole document instead of
    relying on table/heading structure.
    """
    if not rows:
        return None
    first_row = rows[0]
    first_cell = next((c.strip() for c in first_row if c and c.strip()), "")
    if normalize(first_cell) == "pdremarks":
        return "pdRemarks"
    return None


def extract_pdremarks(pdf_path: str) -> dict:
    """
    Direct, structure-agnostic extraction for pdRemarks:
      1. Search the raw text of every page for a line containing the
         literal substring "pdremarks" (case-insensitive).
      2. Take whatever text follows it on that same line (after stripping
         the matched heading text itself).
      3. If nothing follows on that line, take the next non-empty line(s)
         instead, stopping at the next known heading or end of page.
      4. If nothing is found via plain text (e.g. it's only inside a table
         cell), fall back to scanning every table cell for the literal
         string and returning whatever comes right after it — same cell
         (rest of the cell text), next cell in that row, or the next row.

    Returns {"page": <page_num or None>, "text": <captured string>}.
    """
    target = "pdremarks"

    with pdfplumber.open(pdf_path) as pdf:
        # --- Attempt 1: raw text search, line by line ---
        for page_index, page in enumerate(pdf.pages):
            page_num = page_index + 1
            text = page.extract_text() or ""
            raw_lines = text.split("\n")

            for line_idx, line in enumerate(raw_lines):
                norm_line = re.sub(r"\s+", "", line).lower()
                if target in norm_line:
                    # Try to find where in the ORIGINAL (non-normalized) line
                    # the heading text ends, case-insensitively, tolerating
                    # internal spaces like "pd Remarks".
                    match = re.search(r"pd\s*remarks\s*:?", line, flags=re.IGNORECASE)
                    after = ""
                    if match:
                        after = line[match.end():].strip(" :-\t")

                    if after:
                        return {"page": page_num, "text": after}

                    # Nothing after the heading on the same line -> collect
                    # subsequent non-empty lines until the next known
                    # heading or end of page.
                    collected = []
                    for nxt in raw_lines[line_idx + 1:]:
                        nxt_stripped = nxt.strip()
                        if not nxt_stripped:
                            continue
                        norm_nxt = normalize(nxt_stripped)
                        if norm_nxt in HEADING_LOOKUP:
                            break
                        collected.append(nxt_stripped)
                    if collected:
                        return {"page": page_num, "text": " ".join(collected)}
                    # else: keep searching (maybe a later page has the real one)

        # --- Attempt 2: fall back to scanning table cells directly ---
        for page_index, page in enumerate(pdf.pages):
            page_num = page_index + 1
            for t in page.find_tables():
                data = t.extract()
                if not data:
                    continue
                for r_idx, row in enumerate(data):
                    for c_idx, cell in enumerate(row):
                        if not cell:
                            continue
                        norm_cell = re.sub(r"\s+", "", cell).lower()
                        if target in norm_cell:
                            match = re.search(r"pd\s*remarks\s*:?", cell, flags=re.IGNORECASE)
                            after = cell[match.end():].strip(" :-\t") if match else ""
                            if after:
                                return {"page": page_num, "text": after}

                            # Check next cell in the same row
                            if c_idx + 1 < len(row) and row[c_idx + 1] and row[c_idx + 1].strip():
                                return {"page": page_num, "text": row[c_idx + 1].strip()}

                            # Check next row(s) in this table
                            collected = []
                            for nxt_row in data[r_idx + 1:]:
                                for nxt_cell in nxt_row:
                                    if nxt_cell and nxt_cell.strip():
                                        collected.append(nxt_cell.strip())
                                if collected:
                                    break
                            if collected:
                                return {"page": page_num, "text": " ".join(collected)}

    return {"page": None, "text": ""}


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

    # --- Match each standard heading to the next table that comes after it
    table_pointer = 0
    for h_text, h_key, h_page in all_headings:
        if normalize(h_text) not in {normalize(x) for x in STANDARD_HEADINGS}:
            continue  # pdRemarks (if ever caught here) is handled separately below

        # advance pointer to first table with key > heading key
        while table_pointer < len(all_tables) and all_tables[table_pointer][1] < h_key:
            table_pointer += 1
        if table_pointer >= len(all_tables):
            structured[h_text] = {"page": h_page, "data": {}}
            continue

        table_dict, t_key, t_page = all_tables[table_pointer]
        fields = rows_to_fields(table_dict["rows"])
        structured[h_text] = {"page": t_page, "data": fields}
        table_pointer += 1  # this table consumed; move to next for next heading

    # --- pdRemarks: direct raw-text search, independent of table/heading
    # structure (see extract_pdremarks for why).
    remarks_result = extract_pdremarks(pdf_path)
    structured["pdRemarks"] = remarks_result

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

    # if page_image_map:
    #     lines.append("## Images\n")
    #     for page_num in sorted(page_image_map.keys()):
    #         for fname in page_image_map[page_num]:
    #             lines.append(f"- Page {page_num}: images/{fname}")
    #     lines.append("")

    return "\n".join(lines)

import base64
import json
from pathlib import Path

import fitz  # PyMuPDF


def extract_pdremarks_data(pdf_path: str) -> dict:
    """
    Search for 'pdRemarks' in a PDF and produce output.json containing:

    1. page_no:
       1-based page number where the text was first found.

    2. full_text_base64:
       UTF-8 Base64 encoding of all extracted text from the matched page.

    3. top_to_first_image_base64:
       Base64 encoding of a PNG cropped from the top of the matched page
       to the top boundary of the first raster image on that page.

       If there is no raster image on the matched page, the whole page is
       rendered and returned.

    Parameters
    ----------
    pdf_path : str
        Input PDF file path.

    Returns
    -------
    dict
        The same dictionary that is written to output.json.
    """

    search_text = "pdRemarks"
    input_path = Path(pdf_path).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"PDF file not found: {input_path}")

    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")

    if input_path.suffix.lower() != ".pdf":
        raise ValueError(f"Input file must be a PDF: {input_path}")

    # Saves output.json in the same folder as the input PDF.
    output_json_path = input_path.parent / "output.json"

    result = {
        "search_text": search_text,
        "found": False,
        "page_no": None,
        "full_text_base64": None,
        "top_to_first_image_base64": None,
        "crop_mime_type": None,
        "crop_coordinates_pdf_points": None,
        "output_json_path": str(output_json_path),
        "message": None,
    }

    document = None

    try:
        document = fitz.open(str(input_path))

        if document.needs_pass:
            raise ValueError("The PDF is password-protected.")

        matched_page = None

        # Find the first page containing pdRemarks.
        for page_index in range(document.page_count):
            page = document.load_page(page_index)

            # search_for handles PDF text positioning better than a plain
            # string search and returns rectangles for matching text.
            matches = page.search_for(search_text)

            if matches:
                matched_page = page
                result["found"] = True
                result["page_no"] = page_index + 1  # Human-readable page number
                break

        if matched_page is None:
            result["message"] = (
                f"Text '{search_text}' was not found. "
                "If the PDF is scanned, OCR must be applied first."
            )

        else:
            # ------------------------------------------------------------
            # 1. Extract and Base64-encode all text from the matched page.
            # ------------------------------------------------------------
            page_text = matched_page.get_text(
                "text",
                sort=True,
            )

            result["full_text_base64"] = base64.b64encode(
                page_text.encode("utf-8")
            ).decode("ascii")

            # ------------------------------------------------------------
            # 2. Find the topmost raster image on the matched page.
            # ------------------------------------------------------------
            page_rect = matched_page.rect
            image_top_positions = []

            # get_image_info() is more useful than get_images() here because
            # it directly provides the displayed bounding boxes.
            for image_info in matched_page.get_image_info(xrefs=True):
                bbox_values = image_info.get("bbox")

                if not bbox_values:
                    continue

                image_rect = fitz.Rect(bbox_values)

                # Ignore invalid or zero-sized image rectangles.
                if image_rect.is_empty or image_rect.is_infinite:
                    continue

                # Restrict the image rectangle to visible page boundaries.
                visible_image_rect = image_rect & page_rect

                if visible_image_rect.is_empty:
                    continue

                image_top_positions.append(visible_image_rect.y0)

            if image_top_positions:
                # Top edge of the image appearing highest on the page.
                crop_bottom = min(image_top_positions)

                # Avoid an invalid zero-height crop when an image starts
                # exactly at the top of the page.
                if crop_bottom <= page_rect.y0:
                    crop_bottom = page_rect.y1
                    result["message"] = (
                        "The first image starts at the top of the page, "
                        "so the complete page was rendered instead."
                    )
                else:
                    result["message"] = (
                        "The page was cropped from its top edge to the "
                        "top edge of the first raster image."
                    )
            else:
                # No raster image found. Return the whole page.
                crop_bottom = page_rect.y1
                result["message"] = (
                    "No raster image was found on the matched page, "
                    "so the complete page was rendered."
                )

            crop_rect = fitz.Rect(
                page_rect.x0,
                page_rect.y0,
                page_rect.x1,
                crop_bottom,
            )

            # Render at 200 DPI for a readable Base64 PNG.
            pixmap = matched_page.get_pixmap(
                dpi=200,
                clip=crop_rect,
                alpha=False,
                annots=True,
            )

            png_bytes = pixmap.tobytes("png")

            result["top_to_first_image_base64"] = base64.b64encode(
                png_bytes
            ).decode("ascii")

            result["crop_mime_type"] = "image/png"

            result["crop_coordinates_pdf_points"] = {
                "x0": round(crop_rect.x0, 3),
                "y0": round(crop_rect.y0, 3),
                "x1": round(crop_rect.x1, 3),
                "y1": round(crop_rect.y1, 3),
            }

    except fitz.FileDataError as exc:
        raise ValueError(f"Unable to open PDF. It may be damaged: {exc}") from exc

    finally:
        if document is not None:
            document.close()

    # Save the JSON result.
    with output_json_path.open("w", encoding="utf-8") as json_file:
        json.dump(
            result,
            json_file,
            ensure_ascii=False,
            indent=2,
        )

    return result


def append_pd_remarks(md_content: str, crop_result: dict) -> str:
    if not crop_result.get("found", False):
        return md_content

    image_base64 = crop_result.get("top_to_first_image_base64", "")

    if not image_base64:
        return md_content

    ocr = ImageOCR()
    ocr_result = ocr.get_ocr_text(image_b64=image_base64)

    # Normalize OCR output into a string
    if isinstance(ocr_result, list):
        ocr_text = "\n".join(
            str(item).strip()
            for item in ocr_result
            if item is not None and str(item).strip()
        )
    elif ocr_result is None:
        ocr_text = ""
    else:
        ocr_text = str(ocr_result).strip()

    if not ocr_text:
        return md_content

    return (
        f"{md_content.rstrip()}\n\n"
        f"# PD REMARKS\n\n"
        f"{ocr_text}\n"
    )
    
def extract_structured_data(
    llm,
    schema: Type[BaseModel],
    md_content: str,
):
    """
    Extract structured information from Markdown content using the supplied
    Pydantic schema.

    Returns:
        {
            "raw": AIMessage,
            "parsed": schema instance or None,
            "parsing_error": Exception or None,
        }
    """

    structured_llm = llm.with_structured_output(
        schema=schema,
        method="function_calling",
        include_raw=True,
    )

    messages = [
        SystemMessage(
            content=(
                "You are an expert OCR and document-understanding system. "
                "Extract structured information from the supplied document. "
                "Follow the provided output schema exactly. "
                "Use null when information is missing, unreadable, or not "
                "present in the document. "
                "Do not guess or fabricate values."
            )
        ),
        HumanMessage(
            content=(
                "Extract the required structured information from the "
                "following Markdown document:\n\n"
                f"{md_content}"
            )
        ),
    ]

    return structured_llm.invoke(messages)
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
    pd_json=os.path.join(outdir,"pd_json.json")
    schema_json=os.path.join(outdir,"scheam_final.json")
    os.makedirs(outdir, exist_ok=True)

    print("Step 1/3: Extracting images (page-wise)...")
    page_image_map = extract_images(args.pdf_path, images_dir)

    print("\nStep 2/3: Detecting headings + tables, building structured data...")
    structured = build_structured_data(args.pdf_path)

    for h in KNOWN_HEADINGS:
        status = "found" if h in structured else "MISSING"
        print(f"  - {h}: {status}")

    print("\nStep 3/3: Writing JSON + Markdown outputs...")
    # with open(json_path, "w", encoding="utf-8") as f:
    #     json.dump(structured, f, indent=2, ensure_ascii=False)

    pdf_name = os.path.basename(args.pdf_path)
    md_content = render_markdown(structured, page_image_map, pdf_name)
    crop_result = extract_pdremarks_data(args.pdf_path)
    md_content = append_pd_remarks(
        md_content=md_content,
        crop_result=crop_result,
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        temperature=0.1,
        max_tokens=8000,
    )

    result = extract_structured_data(
        llm=llm,
        schema=PDReport,
        md_content=md_content,
    )

    schema_output = result["parsed"]
    with open(schema_json, "w", encoding="utf-8") as file:
        json.dump(
            schema_output.model_dump(mode="json"),
            file,
            indent=4,
            ensure_ascii=False,
        )
    print(schema_output)
    # output = extract_pdremarks_data(args.pdf_path)
    
    # with open(pd_json, "w", encoding="utf-8") as f:
    #     json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nDone.")
    print(f"  Images -> {images_dir}")
    print(f"  JSON   -> {json_path}")
    print(f"  MD     -> {md_path}")


if __name__ == "__main__":
    main()















