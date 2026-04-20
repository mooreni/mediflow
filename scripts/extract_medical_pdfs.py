"""
extract_medical_pdfs.py

One-time data preprocessing tool. Run this once before the first pipeline
execution (or after adding new PDF source files) to convert raw PDF medical
documents into plain-text files that the data loader can read.

Extracts plain text from native PDF medical documents using pdfplumber.
Routes output .txt files into language-specific subdirectories based on
filename suffixes (_HE.pdf → hebrew/, _RU.pdf → russian/).

Usage:
    python scripts/extract_medical_pdfs.py
    python scripts/extract_medical_pdfs.py Form_001_HE.pdf   # single file test
"""

import os
import pdfplumber
from bidi.algorithm import get_display

# ---------------------------------------------------------------------------
# Configuration — adjust these paths as needed
# ---------------------------------------------------------------------------
LANGUAGE_DIRS = {
    "./data/informed_consent_forms/pdfs/he": "./data/informed_consent_forms/text/he",
    "./data/informed_consent_forms/pdfs/ru": "./data/informed_consent_forms/text/ru",
}

# Margin crop in points to exclude vertical/decorative text near page edges.
# (left, top, right, bottom) — increase a value to crop more from that side.
PAGE_MARGIN = {"left": 50, "top": 30,"right": 50, "bottom": 30}
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Opens a PDF and extracts all text from every page using pdfplumber.

    Uses layout=True for spatial layout reconstruction, then applies the
    Unicode Bidirectional Algorithm (python-bidi) per line to restore correct
    RTL reading order for Hebrew text.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        A single string containing the concatenated text of all pages,
        with pages separated by a form-feed character.
    """
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            # Apply bidi reordering line-by-line to fix RTL Hebrew character order
            # Skip single-character lines (vertical margin text artifacts)
            lines = [get_display(line) for line in text.split("\n") if len(line.strip()) > 1]
            pages_text.append("\n".join(lines))
    return "\f".join(pages_text)


def process_pdfs(input_dir: str, output_dir: str) -> None:
    """
    Iterates over all PDF files in input_dir, extracts their text, and writes
    .txt output files into output_dir.

    Args:
        input_dir: Path to the folder containing source PDF files.
        output_dir: Path to the folder where .txt files will be written.
    """
    if not os.path.isdir(input_dir):
        print(f"[ERROR] Input directory not found: {input_dir}")
        return

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"[WARNING] No PDF files found in: {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in '{input_dir}'. Starting extraction...\n")

    success = 0
    errors = 0

    for filename in sorted(pdf_files):
        pdf_path = os.path.join(input_dir, filename)
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_filename)

        try:
            text = extract_text_from_pdf(pdf_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"  [OK]      {filename}  →  {txt_path}")
            success += 1
        except Exception as e:
            print(f"  [ERROR]   {filename}  —  {e}")
            errors += 1

    print(f"\nDone. {success} extracted, {errors} error(s).")


def test_single_hebrew_file(filename: str = "Form_001_HE.pdf") -> None:
    """
    Extracts text from a single Hebrew PDF and prints the result to stdout.

    Useful for quickly verifying extraction quality without processing all files.

    Args:
        filename: Name of the PDF file inside the Hebrew input directory.
    """
    input_dir = "./data/informed_consent_forms/pdfs/he"
    pdf_path = os.path.join(input_dir, filename)

    if not os.path.isfile(pdf_path):
        print(f"[ERROR] File not found: {pdf_path}")
        return

    output_dir = "./data/informed_consent_forms/text/he"
    txt_filename = os.path.splitext(filename)[0] + ".txt"
    txt_path = os.path.join(output_dir, txt_filename)

    print(f"Extracting: {pdf_path}\n{'=' * 60}\n")
    try:
        text = extract_text_from_pdf(pdf_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  [OK]      {filename}  →  {txt_path}")
    except Exception as e:
        print(f"  [ERROR]   {filename}  —  {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        test_single_hebrew_file(sys.argv[1])
    else:
        for input_dir, output_dir in LANGUAGE_DIRS.items():
            process_pdfs(input_dir, output_dir)
