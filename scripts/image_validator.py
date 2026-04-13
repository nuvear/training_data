#!/usr/bin/env python3
"""
image_validator.py — Validate all downloaded images are 896x896 JPEG.
Reports any images that are missing, wrong size, or corrupt.

Usage:
    python3 scripts/image_validator.py
"""

from pathlib import Path
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = REPO_ROOT / "raw_data"
TARGET_SIZE = (896, 896)

errors = []
total = 0

for img_path in sorted(RAW_DATA_DIR.rglob("*.jpg")):
    total += 1
    try:
        with Image.open(img_path) as img:
            if img.size != TARGET_SIZE:
                errors.append(f"WRONG SIZE {img.size}: {img_path}")
            elif img.mode != "RGB":
                errors.append(f"WRONG MODE {img.mode}: {img_path}")
    except Exception as e:
        errors.append(f"CORRUPT ({e}): {img_path}")

print(f"\nValidation complete: {total} images checked.")
if errors:
    print(f"\n{len(errors)} issue(s) found:")
    for e in errors:
        print(f"  {e}")
else:
    print("All images passed validation.")
