#!/usr/bin/env python3
"""
PDF Structured Extraction Pipeline
====================================
Extracts from a PDF, page by page:
  1. Images  -> saved into an output folder, one file per image, page-wise naming.
  2. Text    -> dumped in reading order.
  3. Tables  -> dumped with a "Table Content:" header, then row-wise,
                each row's cells joined left-to-right with commas.

Everything (text + tables) is written into a single, LLM-friendly
Markdown (.md) file, clearly sectioned by page.

Libraries used:
  - pdfplumber : text + table extraction (best layout/table accuracy for
                 born-digital PDFs with ruled or whitespace-aligned tables)
  - PyMuPDF (fitz) : image extraction (more reliable & simpler than pdfplumber
                 for pulling out actual embedded raster images with correct
                 formats, and works page-wise out of the box)

Usage:
    python pdf_extract.py input.pdf --outdir ./output

Output structure:
    output/
      images/
        page_1_img_1.png
        page_1_img_2.png
        page_2_img_1.png
        ...
      extracted_content.md
"""

import argparse
import os
import sys

import pdfplumber
import fitz  # PyMuPDF


def extract_images(pdf_path: str, images_dir: str) -> dict:
    """
    Extract every embedded raster image from the PDF, page-wise, into
    images_dir. Returns a dict {page_number: [filenames]} for reference
    (e.g. if you want to mention "see images/page_3_img_1.png" in the md file).
    """
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

                # Convert CMYK / other non-RGB colorspaces to RGB
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


def format_table_markdown(table: list) -> str:
    """
    Format a single extracted table (list of rows, each row a list of cells)
    as:
        Table Content:
        row1cell1, row1cell2, row1cell3
        row2cell1, row2cell2, row2cell3

    Each row is prefixed with its row number, cells joined left-to-right
    with commas. None/empty cells become empty strings.
    """
    lines = ["Table Content:"]
    for row_idx, row in enumerate(table, start=1):
        cleaned_cells = []
        for cell in row:
            if cell is None:
                cleaned_cells.append("")
            else:
                # Collapse internal newlines/extra whitespace so the row
                # stays on a single line and commas stay unambiguous.
                cleaned = " ".join(str(cell).split())
                cleaned_cells.append(cleaned)
        row_line = ", ".join(cleaned_cells)
        lines.append(f"Row {row_idx}: {row_line}")
    return "\n".join(lines)


def extract_text_and_tables(pdf_path: str, page_image_map: dict) -> str:
    """
    Walk through the PDF page by page with pdfplumber, extracting:
      - plain text
      - tables (formatted separately, per format_table_markdown)
    Returns one combined Markdown string for the whole document.
    """
    md_parts = []
    md_parts.append("# Extracted PDF Content\n")

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            md_parts.append(f"\n---\n\n## Page {page_num}\n")

            # --- Tables first: find bounding boxes so we can exclude their
            # text from the plain-text dump (avoids duplicating table
            # content as garbled inline text).
            tables = page.find_tables()

            if tables:
                for t_idx, t in enumerate(tables, start=1):
                    table_data = t.extract()
                    if not table_data:
                        continue
                    md_parts.append(f"\n### Table {t_idx} (Page {page_num})\n")
                    md_parts.append(format_table_markdown(table_data))
                    md_parts.append("")

            # --- Plain text, with table regions cropped out to avoid
            # duplicate/garbled text where tables were.
            if tables:
                page_for_text = page
                for t in tables:
                    bbox = t.bbox  # (x0, top, x1, bottom)
                    page_for_text = page_for_text.outside_bbox(bbox)
                text = page_for_text.extract_text()
            else:
                text = page.extract_text()

            if text and text.strip():
                md_parts.append("\n### Text\n")
                md_parts.append(text.strip())
            elif not tables:
                md_parts.append("\n### Text\n")
                md_parts.append("*(no extractable text on this page)*")

            # --- Note any images that were extracted for this page, so the
            # LLM knows an image exists here and where to find it.
            if page_num in page_image_map:
                md_parts.append(f"\n### Images on this page\n")
                for fname in page_image_map[page_num]:
                    md_parts.append(f"- images/{fname}")

            print(f"  Page {page_num}/{total_pages} processed "
                  f"({len(tables)} table(s) found)")

    return "\n".join(md_parts)


def main():
    parser = argparse.ArgumentParser(description="Extract text, tables, and images from a PDF.")
    parser.add_argument("pdf_path", help="Path to the input PDF file")
    parser.add_argument("--outdir", default="./output", help="Output directory (default: ./output)")
    args = parser.parse_args()

    if not os.path.isfile(args.pdf_path):
        print(f"Error: file not found: {args.pdf_path}")
        sys.exit(1)

    outdir = args.outdir
    images_dir = os.path.join(outdir, "images")
    md_path = os.path.join(outdir, "extracted_content.md")

    os.makedirs(outdir, exist_ok=True)

    print("Step 1/2: Extracting images (page-wise)...")
    page_image_map = extract_images(args.pdf_path, images_dir)

    print("\nStep 2/2: Extracting text + tables...")
    md_content = extract_text_and_tables(args.pdf_path, page_image_map)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nDone.")
    print(f"  Images  -> {images_dir}")
    print(f"  Content -> {md_path}")


if __name__ == "__main__":
    main()
