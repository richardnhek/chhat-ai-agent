#!/usr/bin/env python3
"""
Excel Cigarette Image Analyzer
===============================
Reads an Excel file, opens image links from each row, uses Claude Vision
to identify cigarette counts and brands, and writes results back to the Excel file.

Usage:
    python main.py input.xlsx
    python main.py input.xlsx --link-column B
    python main.py input.xlsx --link-column "Image URL"
    python main.py input.xlsx --output results.xlsx --model claude-sonnet-4-6
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
import anthropic

from excel_handler import load_excel, write_results
from image_analyzer import analyze_url


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze cigarette images linked in an Excel spreadsheet using Claude Vision.",
    )
    parser.add_argument(
        "input_file",
        help="Path to the input Excel (.xlsx) file",
    )
    parser.add_argument(
        "--output", "-o",
        help="Path for the output Excel file (default: <input>_analyzed.xlsx)",
    )
    parser.add_argument(
        "--link-column", "-c",
        help="Column containing image links — letter (e.g. 'B') or header name (e.g. 'Image URL'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--sheet",
        help="Sheet name to process (default: active sheet)",
    )
    parser.add_argument(
        "--model", "-m",
        default="claude-sonnet-4-6",
        help="Claude model to use for vision analysis (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between API calls to avoid rate limiting (default: 1.0)",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        help="Start processing from this row number (1-based, excluding header). Useful for resuming.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without making API calls.",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    if input_path.suffix.lower() not in (".xlsx", ".xlsm"):
        print(f"Error: File must be .xlsx or .xlsm format, got: {input_path.suffix}")
        sys.exit(1)

    # Validate API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        print("Error: ANTHROPIC_API_KEY not set. Create a .env file or export it.")
        print("  Get your key at: https://console.anthropic.com/")
        sys.exit(1)

    # Set up output path
    if args.output:
        output_path = args.output
    else:
        output_path = str(input_path.with_stem(input_path.stem + "_analyzed"))

    # Parse link column
    link_column = args.link_column
    if link_column and link_column.isdigit():
        link_column = int(link_column)

    # Load Excel
    print(f"\n{'='*60}")
    print(f"  Excel Cigarette Image Analyzer")
    print(f"{'='*60}")
    print(f"\n  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Model:  {args.model}")

    try:
        data = load_excel(str(input_path), sheet_name=args.sheet, link_column=link_column)
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    ws = data["worksheet"]
    rows = data["rows"]
    link_col = data["link_col_index"]
    headers = data["headers"]

    print(f"  Sheet:  {ws.title}")
    print(f"  Link column: {headers[link_col - 1]} (column {link_col})")
    print(f"  Total rows:  {len(rows)}")

    # Filter rows with URLs
    rows_with_urls = [r for r in rows if r["url"]]
    rows_without_urls = [r for r in rows if not r["url"]]
    print(f"  Rows with links:    {len(rows_with_urls)}")
    print(f"  Rows without links: {len(rows_without_urls)}")

    if not rows_with_urls:
        print("\nNo image links found. Nothing to process.")
        sys.exit(0)

    # Apply start-row filter
    if args.start_row:
        rows_with_urls = [r for r in rows_with_urls if r["row_number"] >= args.start_row]
        print(f"  Processing from row {args.start_row}: {len(rows_with_urls)} rows")

    # Dry run
    if args.dry_run:
        print(f"\n--- DRY RUN ---")
        for row in rows_with_urls:
            print(f"  Row {row['row_number']}: {row['url'][:80]}...")
        print(f"\n{len(rows_with_urls)} rows would be processed.")
        sys.exit(0)

    # Initialize Claude client
    client = anthropic.Anthropic(api_key=api_key)

    # Process each row
    print(f"\n{'─'*60}")
    print(f"  Processing {len(rows_with_urls)} images...\n")

    results = []
    for i, row in enumerate(rows_with_urls, 1):
        row_num = row["row_number"]
        url = row["url"]

        print(f"  [{i}/{len(rows_with_urls)}] Row {row_num}: ", end="", flush=True)
        print(f"{url[:60]}{'...' if len(url) > 60 else ''}")

        analysis = analyze_url(url, client, model=args.model)

        if "error" in analysis:
            print(f"           ERROR: {analysis['error']}")
            results.append({
                "row_number": row_num,
                "error": analysis["error"],
            })
        else:
            total_packs = analysis.get("total_packs", "unknown")
            est_cigs = analysis.get("estimated_cigarettes", "unknown")
            confidence = analysis.get("confidence", "unknown")
            breakdown = analysis.get("brand_breakdown", [])
            visibility = analysis.get("visibility_issues", "")

            # Format brand breakdown
            brand_names = []
            breakdown_lines = []
            for item in breakdown:
                brand = item.get("brand", "Unknown")
                packs = item.get("packs", "?")
                notes = item.get("notes", "")
                brand_names.append(brand)
                line = f"{brand}: {packs} packs"
                if notes:
                    line += f" ({notes})"
                breakdown_lines.append(line)

            brand_str = ", ".join(brand_names) if brand_names else "None identified"
            breakdown_str = "\n".join(breakdown_lines) if breakdown_lines else "N/A"

            print(f"           Packs: {total_packs} | Est. cigarettes: {est_cigs} | Brands: {brand_str} | Confidence: {confidence}")

            results.append({
                "row_number": row_num,
                "total_packs": total_packs,
                "brands": brand_str,
                "brand_breakdown": breakdown_str,
                "estimated_cigarettes": est_cigs,
                "visibility_notes": visibility,
            })

        # Rate limit delay (skip after last item)
        if i < len(rows_with_urls) and args.delay > 0:
            time.sleep(args.delay)

    # Also add results for rows without URLs
    for row in rows_without_urls:
        results.append({
            "row_number": row["row_number"],
            "error": "No image link found in this row",
        })

    # Write results
    print(f"\n{'─'*60}")
    print(f"  Writing results to: {output_path}")

    saved_path = write_results(data["workbook"], ws, results, output_path)

    # Summary
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    print(f"\n{'='*60}")
    print(f"  DONE!")
    print(f"  Processed: {len(rows_with_urls)} images")
    print(f"  Success:   {len(successful)}")
    print(f"  Errors:    {len([r for r in failed if r.get('error') != 'No image link found in this row'])}")
    print(f"  Skipped:   {len(rows_without_urls)} (no link)")
    print(f"  Output:    {saved_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
