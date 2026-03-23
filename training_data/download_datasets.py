#!/usr/bin/env python3
"""
Download cigarette detection datasets from Roboflow Universe.

Usage:
    1. Get your API key from https://app.roboflow.com/settings/api
    2. Run: ROBOFLOW_API_KEY=your_key_here python3 download_datasets.py
       Or:  python3 download_datasets.py --api-key your_key_here
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download Roboflow cigarette datasets")
    parser.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"),
                        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: Roboflow API key required.")
        print("  Get yours at: https://app.roboflow.com/settings/api")
        print("  Then run: ROBOFLOW_API_KEY=<key> python3 download_datasets.py")
        sys.exit(1)

    from roboflow import Roboflow

    rf = Roboflow(api_key=args.api_key)
    output_dir = Path(__file__).parent

    datasets = [
        {
            "workspace": "florin-fn2mk",
            "project": "cigarette-packs-apcpr",
            "version": 1,
            "name": "cigarette-packs-apcpr",
            "url": "https://universe.roboflow.com/florin-fn2mk/cigarette-packs-apcpr",
        },
        {
            "workspace": "daay",
            "project": "cigarette-box",
            "version": 1,
            "name": "cigarette-box",
            "url": "https://universe.roboflow.com/daay/cigarette-box",
        },
        {
            "workspace": "irc-6dl2t",
            "project": "cigarette-il5bl",
            "version": 1,
            "name": "cigarette-il5bl",
            "url": "https://universe.roboflow.com/irc-6dl2t/cigarette-il5bl",
        },
        {
            "workspace": "myplay",
            "project": "cigarette-pack-detection",
            "version": 1,
            "name": "cigarette-pack-detection",
            "url": "https://universe.roboflow.com/myplay/cigarette-pack-detection",
        },
    ]

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Downloading: {ds['name']}")
        print(f"  Source: {ds['url']}")
        print(f"{'='*60}")

        try:
            project = rf.workspace(ds["workspace"]).project(ds["project"])
            print(f"  Project info: {project}")

            # Try version 1 first, then latest
            version = None
            for v in [1, 2, 3]:
                try:
                    version = project.version(v)
                    print(f"  Found version {v}")
                    break
                except Exception:
                    continue

            if version is None:
                print(f"  WARNING: Could not find any version for {ds['name']}")
                continue

            dest = str(output_dir / ds["name"])
            print(f"  Downloading COCO format to: {dest}")
            version.download("coco", location=dest)
            print(f"  SUCCESS: Downloaded {ds['name']}")

        except Exception as e:
            print(f"  ERROR downloading {ds['name']}: {e}")
            print(f"  You can try manually at: {ds['url']}")

    print(f"\n{'='*60}")
    print("Download complete! Check {output_dir} for datasets.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
