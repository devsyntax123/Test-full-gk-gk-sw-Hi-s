#!/usr/bin/env python3
"""
PD Report extraction pipeline — in-memory version.

Public entry point:

    schema_output = run_pipeline(pdf_base64: str) -> dict

Everything else (image extraction, heading/table detection, pdRemarks
crop+OCR, LLM structured extraction) is unchanged logic from the original
script — only the I/O layer changed: no file paths in, no files written
out. Base64 in, schema dict out.
"""

import base64
import io
import json
import os
import re
from typing import Type

import fitz  # PyMuPDF
import pdfplumber
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from schemas.structure import PDReport
from main import ImageOCR

# ---------------------------------------------------------------------------
# Config: the known section headings, in the canonical order you specified.
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

FREE_TEXT_SECTIONS = {"pdremarks"}
STANDARD_HEADINGS = [h for h in KNOWN_HEADINGS if h.strip().lower() != "pdremarks"]


def normalize(s: str) -> str:
    return re.sub(r"\s+", "", s or "").strip().lower()


HEADING_LOOKUP = {normalize(h): h for h in KNOWN_HEADINGS}


# ---------------------------------------------------------------------------
# Heading + table detection (unchanged logic)
# ---------------------------------------------------------------------------
def find_headings_on_page(page) -> list:
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    if not words:
        return []

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
    parts = []
    for row in rows:
        for cell in row:
            if cell and cell.strip():
                parts.append(cell.strip())
    return " ".join(parts).strip()


def check_table_for_in_table_heading(rows: list):
    if not rows:
        return None
    first_row = rows[0]
    first_cell = next((c.strip() for c in first_row if c and c.strip()), "")
    if normalize(first_cell) == "pdremarks":
        return "pdRemarks"
    return None


# ---------------------------------------------------------------------------
# pdRemarks text search (operates on an already-open pdfplumber pdf object)
# ---------------------------------------------------------------------------
def extract_pdremarks(pdf) -> dict:
    """
    Same logic as before, but takes an already-open pdfplumber PDF object
    instead of a path, so callers control the file handle / bytes source.
    """
    target = "pdremarks"

    # --- Attempt 1: raw text search, line by line ---
    for page_index, page in enumerate(pdf.pages):
        page_num = page_index + 1
        text = page.extract_text() or ""
        raw_lines = text.split("\n")

        for line_idx, line in enumerate(raw_lines):
            norm_line = re.sub(r"\s+", "", line).lower()
            if target in norm_line:
                match = re.search(r"pd\s*remarks\s*:?", line, flags=re.IGNORECASE)
                after = ""
                if match:
                    after = line[match.end():].strip(" :-\t")

                if after:
                    return {"page": page_num, "text": after}

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

                        if c_idx + 1 < len(row) and row[c_idx + 1] and row[c_idx + 1].strip():
                            return {"page": page_num, "text": row[c_idx + 1].strip()}

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
def build_structured_data(pdf_bytes: bytes) -> dict:
    all_headings = []
    all_tables = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_num = page_index + 1

            headings = find_headings_on_page(page)
            for h_text, top in headings:
                global_key = page_index * 100000 + top
                all_headings.append((h_text, global_key, page_num))

            tables = extract_tables_with_position(page)
            for t in tables:
                global_key = page_index * 100000 + t["top"]
                all_tables.append((t, global_key, page_num))

        all_headings.sort(key=lambda x: x[1])
        all_tables.sort(key=lambda x: x[1])

        structured = {}

        table_pointer = 0
        for h_text, h_key, h_page in all_headings:
            if normalize(h_text) not in {normalize(x) for x in STANDARD_HEADINGS}:
                continue

            while table_pointer < len(all_tables) and all_tables[table_pointer][1] < h_key:
                table_pointer += 1
            if table_pointer >= len(all_tables):
                structured[h_text] = {"page": h_page, "data": {}}
                continue

            table_dict, t_key, t_page = all_tables[table_pointer]
            fields = rows_to_fields(table_dict["rows"])
            structured[h_text] = {"page": t_page, "data": fields}
            table_pointer += 1

        # pdRemarks: direct raw-text search (reuses the same open pdf)
        remarks_result = extract_pdremarks(pdf)
        structured["pdRemarks"] = remarks_result

    return structured


# ---------------------------------------------------------------------------
# Markdown rendering (unchanged)
# ---------------------------------------------------------------------------
def render_markdown(structured: dict, pdf_name: str) -> str:
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

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# pdRemarks image crop (operates on in-memory bytes instead of a path)
# ---------------------------------------------------------------------------
def extract_pdremarks_data(pdf_bytes: bytes) -> dict:
    """
    Search for 'pdRemarks' in the PDF (bytes) and return a dict containing:
      - page_no
      - full_text_base64
      - top_to_first_image_base64 (+ crop info)
    No files are written; this is all in-memory now.
    """
    search_text = "pdRemarks"

    result = {
        "search_text": search_text,
        "found": False,
        "page_no": None,
        "full_text_base64": None,
        "top_to_first_image_base64": None,
        "crop_mime_type": None,
        "crop_coordinates_pdf_points": None,
        "message": None,
    }

    document = None
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")

        if document.needs_pass:
            raise ValueError("The PDF is password-protected.")

        matched_page = None

        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            matches = page.search_for(search_text)
            if matches:
                matched_page = page
                result["found"] = True
                result["page_no"] = page_index + 1
                break

        if matched_page is None:
            result["message"] = (
                f"Text '{search_text}' was not found. "
                "If the PDF is scanned, OCR must be applied first."
            )
        else:
            page_text = matched_page.get_text("text", sort=True)
            result["full_text_base64"] = base64.b64encode(
                page_text.encode("utf-8")
            ).decode("ascii")

            page_rect = matched_page.rect
            image_top_positions = []

            for image_info in matched_page.get_image_info(xrefs=True):
                bbox_values = image_info.get("bbox")
                if not bbox_values:
                    continue
                image_rect = fitz.Rect(bbox_values)
                if image_rect.is_empty or image_rect.is_infinite:
                    continue
                visible_image_rect = image_rect & page_rect
                if visible_image_rect.is_empty:
                    continue
                image_top_positions.append(visible_image_rect.y0)

            if image_top_positions:
                crop_bottom = min(image_top_positions)
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
                crop_bottom = page_rect.y1
                result["message"] = (
                    "No raster image was found on the matched page, "
                    "so the complete page was rendered."
                )

            crop_rect = fitz.Rect(
                page_rect.x0, page_rect.y0, page_rect.x1, crop_bottom
            )

            pixmap = matched_page.get_pixmap(
                dpi=200, clip=crop_rect, alpha=False, annots=True
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

    return result


def append_pd_remarks(md_content: str, crop_result: dict) -> str:
    if not crop_result.get("found", False):
        return md_content

    image_base64 = crop_result.get("top_to_first_image_base64", "")
    if not image_base64:
        return md_content

    ocr = ImageOCR()
    ocr_result = ocr.get_ocr_text(image_b64=image_base64)

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


def extract_structured_data(llm, schema: Type[BaseModel], md_content: str):
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
# Page-wise image extraction (in-memory, base64, keyed by page number)
# ---------------------------------------------------------------------------
def extract_images_base64(pdf_bytes: bytes) -> dict:
    """
    Returns {page_num: [base64_png, base64_png, ...], ...} for every page
    that has raster images. No files written — pure in-memory.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_image_map = {}

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_num = page_index + 1
        image_list = page.get_images(full=True)
        if not image_list:
            continue

        encoded_images = []
        for img in image_list:
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                png_bytes = pix.tobytes("png")
                encoded_images.append(base64.b64encode(png_bytes).decode("ascii"))
            except Exception as e:
                print(f"  [warn] failed to extract image on page {page_num}: {e}")

        if encoded_images:
            page_image_map[page_num] = encoded_images

    doc.close()
    return page_image_map


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------
def run_pipeline(pdf_base64: str, pdf_name: str = "document.pdf") -> dict:
    """
    Take a base64-encoded PDF string, run the full extraction pipeline
    in memory, and return schema_output as a plain dict (JSON-serializable).

    No paths, no files on disk — everything stays in memory.
    """
    pdf_bytes = base64.b64decode(pdf_base64)

    # Step 1a: page-wise images (base64, keyed by page number)
    page_images_base64 = extract_images_base64(pdf_bytes)

    # Step 1b: headings + tables -> structured dict
    structured = build_structured_data(pdf_bytes)

    # Step 2: markdown rendering
    md_content = render_markdown(structured, pdf_name)

    # Step 3: pdRemarks crop + OCR, appended to markdown
    crop_result = extract_pdremarks_data(pdf_bytes)
    md_content = append_pd_remarks(md_content=md_content, crop_result=crop_result)

    # Step 4: LLM structured extraction
    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        temperature=0.1,
        max_tokens=8000,
    )

    result = extract_structured_data(llm=llm, schema=PDReport, md_content=md_content)
    schema_output = result["parsed"]

    if schema_output is None:
        raise ValueError(
            f"LLM failed to parse structured output. Parsing error: "
            f"{result.get('parsing_error')}"
        )

    return {
        "schema_output": schema_output.model_dump(mode="json"),
        "page_images_base64": page_images_base64,  # {page_num: [b64, b64, ...]}
    }


if __name__ == "__main__":
    # Simple CLI for local testing: pass a path, it base64-encodes it,
    # runs the pipeline, and prints schema_output as JSON.
    import sys

    if len(sys.argv) != 2:
        print("Usage: python pd_pipeline.py <path_to_pdf>")
        sys.exit(1)

    with open(sys.argv[1], "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    output = run_pipeline(b64, pdf_name=os.path.basename(sys.argv[1]))
    print(json.dumps(output["schema_output"], indent=2, ensure_ascii=False))
    print(f"\nPages with images: {list(output['page_images_base64'].keys())}")
