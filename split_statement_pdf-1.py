"""
Splits a PDF (given as base64) into two parts, based on the page where
"Statement of Income" first appears:

  1. "image"  -> base64 PNG(s) of every page BEFORE the split page,
                 merged/stacked into one image (in the rare case there's
                 more than one page before the split page).
  2. "pdf"    -> base64 PDF containing the split page itself and every
                 page after it.

Usage:
    result = split_pdf_on_statement_of_income(pdf_base64_string)
    result["image_base64"]      -> base64 PNG
    result["pdf_base64"]        -> base64 PDF
    result["meta"]["pages_merged_for_image"]
    result["meta"]["pages_in_pdf"]
    result["meta"]["split_page_index"]   # 0-indexed page where match was found
"""

import base64
import io
import re
from typing import Optional

import pdfplumber
from pypdf import PdfReader, PdfWriter
from pdf2image import convert_from_bytes
from PIL import Image


SEARCH_TEXT = "statement of income"


def _normalize(text: str) -> str:
    """Lowercase and collapse all whitespace, so 'Statement  of\\nIncome' still matches."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _find_split_page_index(pdf_bytes: bytes, search_text: str = SEARCH_TEXT) -> Optional[int]:
    """Return the 0-indexed page number of the first page containing search_text.
    Matching is case-insensitive and whitespace-insensitive."""
    normalized_search = _normalize(search_text)
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if normalized_search in _normalize(text):
                return i
    return None


def _first_page_as_image_base64(pdf_bytes: bytes) -> str:
    """Render just page 1 of the PDF to a base64 PNG."""
    images = convert_from_bytes(pdf_bytes, first_page=1, last_page=1, dpi=200)
    buf = io.BytesIO()
    images[0].save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _stack_images_vertically(images: list[Image.Image]) -> Image.Image:
    """Stack multiple page images into a single vertical image."""
    if len(images) == 1:
        return images[0]

    widths = [img.width for img in images]
    heights = [img.height for img in images]
    max_width = max(widths)
    total_height = sum(heights)

    combined = Image.new("RGB", (max_width, total_height), "white")
    y_offset = 0
    for img in images:
        combined.paste(img, (0, y_offset))
        y_offset += img.height

    return combined


def split_pdf_on_statement_of_income(pdf_base64: str, search_text: str = SEARCH_TEXT) -> dict:
    """
    Splits the given base64-encoded PDF into an "above" image and a "below" pdf,
    based on the first page containing `search_text`.

    Returns a dict:
        {
            "image_base64": str,   # base64 PNG of all pages before the split page
            "pdf_base64": str,     # base64 PDF of split page + everything after
            "meta": {
                "split_page_index": int,      # 0-indexed
                "pages_merged_for_image": int,
                "pages_in_pdf": int,
                "total_pages": int,
            }
        }

    If search_text is NOT found anywhere in the document, no exception is raised.
    Instead, "image_base64" is set to the first page of the PDF (as a base64 PNG),
    "pdf_base64" is set to None, and meta["found"] is False.
    """
    pdf_bytes = base64.b64decode(pdf_base64)

    reader = PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader.pages)

    split_index = _find_split_page_index(pdf_bytes, search_text)

    if split_index is None:
        # Not found: just send back the first page as an image, no pdf part.
        return {
            "image_base64": _first_page_as_image_base64(pdf_bytes),
            "pdf_base64": None,
            "meta": {
                "found": False,
                "message": f"'{search_text}' not found in document.",
                "split_page_index": None,
                "pages_merged_for_image": 1,
                "pages_in_pdf": 0,
                "total_pages": total_pages,
            },
        }

    # ---- Part 1: image of all pages BEFORE split_index ----
    pages_before_count = split_index  # 0-indexed, so this is the count of pages before it
    image_base64 = None

    if pages_before_count > 0:
        # convert_from_bytes uses 1-indexed page numbers
        page_images = convert_from_bytes(
            pdf_bytes,
            first_page=1,
            last_page=pages_before_count,
            dpi=200,
        )
        combined_image = _stack_images_vertically(page_images)
        img_buffer = io.BytesIO()
        combined_image.save(img_buffer, format="PNG")
        image_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    # else: nothing before the split page — no image content exists.

    # ---- Part 2: PDF from split_index to the end ----
    writer = PdfWriter()
    for i in range(split_index, total_pages):
        writer.add_page(reader.pages[i])

    pdf_buffer = io.BytesIO()
    writer.write(pdf_buffer)
    pdf_part_base64 = base64.b64encode(pdf_buffer.getvalue()).decode("utf-8")

    pages_in_pdf_count = total_pages - split_index

    return {
        "image_base64": image_base64,
        "pdf_base64": pdf_part_base64,
        "meta": {
            "found": True,
            "split_page_index": split_index,       # 0-indexed page number where match found
            "pages_merged_for_image": pages_before_count,
            "pages_in_pdf": pages_in_pdf_count,
            "total_pages": total_pages,
        },
    }


if __name__ == "__main__":
    # Example usage:
    # with open("input.pdf", "rb") as f:
    #     b64 = base64.b64encode(f.read()).decode("utf-8")
    # result = split_pdf_on_statement_of_income(b64)
    # print(result["meta"])
    #
    # if result["image_base64"]:
    #     with open("above_part.png", "wb") as f:
    #         f.write(base64.b64decode(result["image_base64"]))
    #
    # with open("below_part.pdf", "wb") as f:
    #     f.write(base64.b64decode(result["pdf_base64"]))
    pass
