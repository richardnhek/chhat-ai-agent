#!/usr/bin/env python3
"""
Process the client's CHHAT Excel file:
1. Read Raw Data sheet (serial + image links)
2. For each row, fetch images and analyze with Claude Vision
3. Output Excel with: serial | embedded images | Q12A brands | Q12B SKUs | brand count
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XlImage
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter
from PIL import Image as PILImage

from image_analyzer import fetch_image, analyze_image, _resize_image, get_available_models
from brands import format_q12a, BRANDS_AND_SKUS
from corrections import find_relevant_corrections, format_corrections_for_prompt, get_correction_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Process CHHAT cigarette survey images.")
    parser.add_argument("input_file", help="Path to the CHHAT Excel file")
    parser.add_argument("--output", "-o", help="Output file path (default: <input>_results.xlsx)")
    parser.add_argument("--model", "-m", default="claude-sonnet-4-6",
                        help=f"AI model. Options: {', '.join(get_available_models())}")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between API calls (seconds)")
    parser.add_argument("--start-row", type=int, default=3, help="First data row in Raw Data sheet (default: 3)")
    parser.add_argument("--limit", type=int, help="Only process first N rows (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    parser.add_argument("--photo-cols", default="B,C,D", help="Columns with Q32 photo links (default: B,C,D)")
    return parser.parse_args()


def read_raw_data(file_path, start_row=3, photo_cols_str="B,C,D"):
    """Read the Raw Data sheet and extract serial numbers + image URLs."""
    from openpyxl import load_workbook
    from openpyxl.utils import column_index_from_string

    wb = load_workbook(file_path)
    ws = wb["Raw data "]

    photo_cols = [column_index_from_string(c.strip()) for c in photo_cols_str.split(",")]

    rows = []
    for row_num in range(start_row, ws.max_row + 1):
        serial = ws.cell(row=row_num, column=1).value
        if not serial:
            continue

        urls = []
        for col in photo_cols:
            val = ws.cell(row=row_num, column=col).value
            if val and str(val).startswith("http"):
                urls.append(str(val).strip())

        rows.append({"row_number": row_num, "serial": serial, "urls": urls})

    return rows


def create_thumbnail(image_data: bytes, max_size=(200, 200)) -> bytes:
    """Create a thumbnail for embedding in Excel."""
    img = PILImage.open(io.BytesIO(image_data))
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    img.thumbnail(max_size, PILImage.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return buf.getvalue()


def build_output(results: list[dict], output_path: str):
    """Build the output Excel with embedded images and brand analysis."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Results"

    # Headers
    headers = [
        ("A", "Serial Number", 16),
        ("B", "Q32 Image 1", 30),
        ("C", "Q32 Image 2", 30),
        ("D", "Q32 Image 3", 30),
        ("E", "Q12A - Brands", 50),
        ("F", "Q12B - SKUs", 60),
        ("G", "Brand Count", 14),
        ("H", "Unidentified Packs", 18),
        ("I", "Confidence", 14),
        ("J", "Status", 12),
    ]

    header_font = Font(bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill(start_color="2F5496", end_color="2F5496", fill_type="solid")

    for col_letter, title, width in headers:
        cell = ws[f"{col_letter}1"]
        cell.value = title
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.column_dimensions[col_letter].width = width

    # Data rows
    for i, result in enumerate(results):
        row = i + 2
        ws.row_dimensions[row].height = 120  # Tall rows for images

        # Serial number
        ws.cell(row=row, column=1, value=result["serial"])
        ws.cell(row=row, column=1).alignment = Alignment(vertical="center")

        # Embed images (up to 3)
        thumbnails = result.get("thumbnails", [])
        for img_idx, thumb_data in enumerate(thumbnails[:3]):
            col = img_idx + 2  # B=2, C=3, D=4
            try:
                img_stream = io.BytesIO(thumb_data)
                img = XlImage(img_stream)
                img.width = 180
                img.height = 110
                cell_ref = f"{get_column_letter(col)}{row}"
                ws.add_image(img, cell_ref)
            except Exception:
                ws.cell(row=row, column=col, value="[image error]")

        # Q12A - Brands
        brands = result.get("brands", [])
        q12a = format_q12a(brands)
        ws.cell(row=row, column=5, value=q12a)
        ws.cell(row=row, column=5).alignment = Alignment(wrap_text=True, vertical="center")

        # Q12B - SKUs
        skus = result.get("skus", [])
        ws.cell(row=row, column=6, value=" | ".join(skus) if skus else "")
        ws.cell(row=row, column=6).alignment = Alignment(wrap_text=True, vertical="center")

        # Brand count
        ws.cell(row=row, column=7, value=len(brands))
        ws.cell(row=row, column=7).alignment = Alignment(horizontal="center", vertical="center")
        ws.cell(row=row, column=7).font = Font(bold=True, size=14)

        # Unidentified packs
        unidentified = result.get("unidentified_packs", 0)
        cell_unid = ws.cell(row=row, column=8, value=unidentified)
        cell_unid.alignment = Alignment(horizontal="center", vertical="center")
        if unidentified and unidentified > 0:
            cell_unid.font = Font(color="FF8C00", bold=True)

        # Confidence
        confidence = result.get("confidence", "")
        cell_conf = ws.cell(row=row, column=9, value=confidence)
        cell_conf.alignment = Alignment(horizontal="center", vertical="center")
        if confidence == "high":
            cell_conf.font = Font(color="00B050", bold=True)
        elif confidence == "medium":
            cell_conf.font = Font(color="FF8C00", bold=True)
        elif confidence == "low":
            cell_conf.font = Font(color="FF0000", bold=True)

        # Status
        if result.get("error"):
            status_cell = ws.cell(row=row, column=10, value="ERROR")
            status_cell.font = Font(color="FF0000", bold=True)
        else:
            status_cell = ws.cell(row=row, column=10, value="OK")
            status_cell.font = Font(color="00B050", bold=True)
        status_cell.alignment = Alignment(horizontal="center", vertical="center")

    wb.save(output_path)
    return output_path


def main():
    load_dotenv()
    args = parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    output_path = args.output or str(input_path.with_stem(input_path.stem + "_results"))

    # Build API keys from env
    api_keys = {
        "claude": os.getenv("ANTHROPIC_API_KEY", ""),
        "gemini": os.getenv("GEMINI_API_KEY", ""),
        "fireworks": os.getenv("FIREWORKS_API_KEY", ""),
    }

    # Read data
    rows = read_raw_data(str(input_path), start_row=args.start_row, photo_cols_str=args.photo_cols)

    if args.limit:
        rows = rows[:args.limit]

    print(f"\n{'='*60}")
    print(f"  CHHAT Cigarette Brand Analyzer")
    print(f"{'='*60}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Rows:   {len(rows)}")
    print(f"  Model:  {args.model}")

    if args.dry_run:
        print(f"\n--- DRY RUN ---")
        for row in rows:
            print(f"  Serial {row['serial']}: {len(row['urls'])} image(s)")
            for u in row["urls"]:
                print(f"    {u[:80]}...")
        sys.exit(0)

    # Load past corrections for few-shot learning
    stats = get_correction_stats()
    if stats["total"] > 0:
        print(f"  Corrections loaded: {stats['total']} past corrections")
        recent_corrections = find_relevant_corrections(limit=5)
        correction_context = format_corrections_for_prompt(recent_corrections)
    else:
        print(f"  Corrections: none yet (first run)")
        correction_context = ""

    print(f"\n{'─'*60}")
    print(f"  Processing {len(rows)} outlets...\n")

    results = []
    for i, row in enumerate(rows, 1):
        serial = row["serial"]
        urls = row["urls"]

        print(f"  [{i}/{len(rows)}] Serial {serial} ({len(urls)} image(s))")

        all_brands = set()
        all_skus = set()
        thumbnails = []
        total_unidentified = 0
        worst_confidence = "high"
        error = None
        confidence_rank = {"low": 0, "medium": 1, "high": 2}

        for url_idx, url in enumerate(urls):
            try:
                image_data, media_type = fetch_image(url)
                thumbnails.append(create_thumbnail(image_data))

                # Analyze
                analysis = analyze_image(image_data, media_type, model=args.model, api_keys=api_keys, correction_context=correction_context)

                if "error" not in analysis:
                    for brand_entry in analysis.get("brands_found", []):
                        brand_name = brand_entry.get("brand", "")
                        if brand_name in BRANDS_AND_SKUS:
                            all_brands.add(brand_name)
                        for sku in brand_entry.get("skus", []):
                            if sku:
                                all_skus.add(sku)
                    total_unidentified += analysis.get("unidentified_packs", 0)
                    img_conf = analysis.get("confidence", "medium")
                    if confidence_rank.get(img_conf, 1) < confidence_rank.get(worst_confidence, 2):
                        worst_confidence = img_conf
                else:
                    error = analysis["error"]

            except Exception as e:
                print(f"           Image {url_idx+1} error: {e}")
                error = str(e)

            if url_idx < len(urls) - 1:
                time.sleep(args.delay)

        brands_sorted = sorted(all_brands)
        skus_sorted = sorted(all_skus)

        unid_str = f" | Unidentified: {total_unidentified}" if total_unidentified else ""
        print(f"           Brands: {', '.join(brands_sorted) if brands_sorted else 'None'} ({len(brands_sorted)}) | Confidence: {worst_confidence}{unid_str}")

        results.append({
            "serial": serial,
            "brands": brands_sorted,
            "skus": skus_sorted,
            "thumbnails": thumbnails,
            "unidentified_packs": total_unidentified,
            "confidence": worst_confidence,
            "error": error if not brands_sorted and error else None,
        })

        if i < len(rows):
            time.sleep(args.delay)

    # Build output
    print(f"\n{'─'*60}")
    print(f"  Writing results to: {output_path}")
    build_output(results, output_path)

    # Summary
    total_brands = set()
    for r in results:
        total_brands.update(r["brands"])

    print(f"\n{'='*60}")
    print(f"  DONE!")
    print(f"  Outlets processed: {len(results)}")
    print(f"  Unique brands across all outlets: {len(total_brands)}")
    if total_brands:
        print(f"  Brands: {', '.join(sorted(total_brands))}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
