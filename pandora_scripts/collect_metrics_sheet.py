import os
import re
import json
import csv
import argparse
from pathlib import Path


def parse_number(value):
    """Extract a float from values like '0.83', 'tensor(0.83)',
    'tensor(0.8312, device=\"cuda:0\")', etc. Returns None if not parseable."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    return float(match.group()) if match else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default="/mnt/data/nerfstudio_ds/real_data",
                        help="Root directory containing one folder per scene.")
    parser.add_argument("--pattern", type=str, default="*results*.json",
                        help="Glob pattern for the metric JSON files to collect.")
    parser.add_argument("--output", type=str, default="metrics_sheet.csv",
                        help="Path to the output CSV sheet.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    scenes = sorted(p for p in data_root.iterdir() if p.is_dir())

    rows = []
    metric_keys = []  # preserve discovery order of metric columns

    for scene in scenes:
        json_files = sorted(scene.rglob(args.pattern))
        if not json_files:
            print(f"No metric JSON found for scene: {scene.name}")
            continue
        for json_path in json_files:
            try:
                with open(json_path) as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Could not read {json_path}: {e}")
                continue

            row = {
                "scene": scene.name,
                "result_file": json_path.name,
                "result_path": str(json_path.relative_to(data_root)),
            }
            for key, value in data.items():
                if key not in metric_keys:
                    metric_keys.append(key)
                row[key] = parse_number(value)
            rows.append(row)

    if not rows:
        print("No metric JSON files found; nothing written.")
        raise SystemExit(0)

    fieldnames = ["scene", "result_file", "result_path"] + metric_keys
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows ({len(scenes)} scenes scanned) to: {os.path.abspath(args.output)}")
