#!/usr/bin/env python3
"""
process_batch.py — Innuir MedGemma Training Data Pipeline
==========================================================
For each JSON file in the input folder:
  1. Reads the rich nutritional metadata JSON.
  2. Downloads 7 purposeful training images at 896x896 px using slot-specific
     Bing image searches (per expert recommendation):
       img_01 — Hero Shot (raw/whole food)
       img_02 — Cross-section / Texture (fat distribution, freshness)
       img_03 — Prepared / Cooked State (real-world meal context)
       img_04 — Scale Reference (food next to standard object)
       img_05 — Packaging / Label (Singapore retail branding)
       img_06 — Variation A (different lighting or preparation)
       img_07 — Variation B (another local preparation style)
  3. Saves images to raw_data/batch_XXXX_to_YYYY/<food_id>/
  4. Generates a fine-tuning JSONL entry per image in hf_dataset/train/metadata.jsonl

Usage:
    python3 scripts/process_batch.py --input /path/to/50_json_files --batch_num 1

Arguments:
    --input      Path to folder containing the 50 JSON files for this batch
    --batch_num  Batch number (1 = records 1-50, 2 = records 51-100, etc.)
    --split      Fraction of records to assign to validation set (default: 0.1)
"""

import os
import sys
import json
import time
import argparse
import re
import shutil
from pathlib import Path
from io import BytesIO

import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = REPO_ROOT / "raw_data"
HF_TRAIN_DIR = REPO_ROOT / "hf_dataset" / "train"
HF_VAL_DIR = REPO_ROOT / "hf_dataset" / "validation"
SHARD_SIZE = 500
TARGET_SIZE = (896, 896)
REQUEST_TIMEOUT = 15
SLEEP_BETWEEN_SEARCHES = 1.0   # Polite delay between Bing requests
SLEEP_BETWEEN_DOWNLOADS = 0.3  # Polite delay between image downloads

# ---------------------------------------------------------------------------
# Image Slot Definitions (Expert Recommendation)
# ---------------------------------------------------------------------------
IMAGE_SLOTS = [
    {
        "slot": "img_01",
        "label": "Hero Shot (Raw/Whole)",
        "purpose": "Primary identification and classification.",
        "query_suffix": "food photography whole dish overhead",
    },
    {
        "slot": "img_02",
        "label": "Cross-section / Texture",
        "purpose": "Identifying fat distribution (marbling) or freshness.",
        "query_suffix": "cross section texture close up macro",
    },
    {
        "slot": "img_03",
        "label": "Prepared / Cooked State",
        "purpose": "Recognizing the food in a real-world meal context.",
        "query_suffix": "cooked served meal plate Singapore hawker",
    },
    {
        "slot": "img_04",
        "label": "Scale Reference",
        "purpose": "Food next to a standard object for volume training.",
        "query_suffix": "food portion size scale bowl spoon reference",
    },
    {
        "slot": "img_05",
        "label": "Packaging / Label",
        "purpose": "Extracting branding and localized Singaporean data.",
        "query_suffix": "Singapore supermarket packaging label product",
    },
    {
        "slot": "img_06",
        "label": "Variation A",
        "purpose": "Different lighting or common local preparation.",
        "query_suffix": "variation preparation style different angle",
    },
    {
        "slot": "img_07",
        "label": "Variation B",
        "purpose": "Another local preparation style (e.g., stir-fried vs. boiled).",
        "query_suffix": "local style alternative preparation method",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text[:80]


def get_shard_name(record_num: int) -> str:
    start = ((record_num - 1) // SHARD_SIZE) * SHARD_SIZE + 1
    end = start + SHARD_SIZE - 1
    return f"batch_{start:04d}_to_{end:04d}"


def search_images_bing(query: str, num_results: int = 10) -> list[str]:
    """Search Bing Images and return a list of direct image URLs."""
    try:
        search_query = query.replace(" ", "+")
        url = (
            f"https://www.bing.com/images/search"
            f"?q={search_query}&count={num_results * 2}&safeSearch=Moderate"
        )
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        urls = re.findall(r'murl&quot;:&quot;(https?://[^&"]+)&quot;', response.text)
        seen, unique = set(), []
        for u in urls:
            if u not in seen:
                seen.add(u)
                unique.append(u)
        return unique[:num_results * 2]
    except Exception as e:
        print(f"    [WARN] Bing search failed for '{query}': {e}")
        return []


def download_and_resize(url: str, save_path: Path) -> bool:
    """Download an image URL, centre-crop to square, resize to 896x896, save as JPEG."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
        img.save(save_path, "JPEG", quality=90)
        return True
    except Exception as e:
        print(f"    [WARN] Failed to download {url[:80]}...: {e}")
        return False


def collect_slot_images(food_name: str, food_dir: Path) -> dict[str, bool]:
    """
    For each of the 7 image slots, search Bing with a slot-specific query
    and download the first successful image. Returns a dict of slot -> success.
    """
    results = {}
    for slot_def in IMAGE_SLOTS:
        slot = slot_def["slot"]
        query = f"{food_name} {slot_def['query_suffix']} no people"
        save_path = food_dir / f"{slot}.jpg"

        # Skip if already downloaded (idempotent re-runs)
        if save_path.exists():
            print(f"    [{slot}] Already exists, skipping.")
            results[slot] = True
            continue

        print(f"    [{slot}] {slot_def['label']}: searching...")
        urls = search_images_bing(query, num_results=8)
        time.sleep(SLEEP_BETWEEN_SEARCHES)

        success = False
        for url in urls:
            if download_and_resize(url, save_path):
                success = True
                time.sleep(SLEEP_BETWEEN_DOWNLOADS)
                break
            time.sleep(SLEEP_BETWEEN_DOWNLOADS)

        if not success:
            print(f"    [{slot}] No valid image found.")
        results[slot] = success

    return results


def build_assistant_response(data: dict) -> str:
    """Build a rich, structured assistant response from the JSON metadata."""
    name = data.get("food_name", data.get("name", "This food"))
    per100 = data.get("per_100g", {})
    serving = data.get("serving_size_g", 100)
    gi = data.get("glycemic_index")
    gl = data.get("glycemic_load_per_serving")
    gi_src = data.get("gi_source", "")
    health = data.get("health_context", {})
    llm = data.get("llm_training", {})
    description = llm.get("natural_language_description", "")
    notes = data.get("notes", "")

    lines = []
    lines.append(f"This is **{name}**.")
    if description:
        lines.append(description)

    if per100:
        kcal = per100.get("energy_kcal", "N/A")
        protein = per100.get("protein_g", "N/A")
        fat = per100.get("fat_g", "N/A")
        carbs = per100.get("carbohydrate_g", "N/A")
        sodium = per100.get("sodium_mg", "N/A")
        fibre = per100.get("dietary_fibre_g", "N/A")
        lines.append(
            f"Per 100g: {kcal} kcal, {protein}g protein, {fat}g fat, "
            f"{carbs}g carbohydrates, {fibre}g dietary fibre, {sodium}mg sodium. "
            f"Standard serving size is approximately {serving}g."
        )

    if gi is not None:
        gi_cat = "Low (≤55)" if gi <= 55 else ("Medium (56–69)" if gi <= 69 else "High (≥70)")
        lines.append(
            f"Glycemic Index: {gi} ({gi_cat}). "
            f"Glycemic Load per serving: {gl}. "
            f"Source: {gi_src}"
        )

    for condition, ctx in health.items():
        concern = ctx.get("concern_level", "")
        factor = ctx.get("key_factor", "")
        guidance = ctx.get("guidance", "")
        if concern and factor:
            condition_label = condition.replace("_", " ").title()
            lines.append(
                f"{condition_label} — Concern level: {concern.capitalize()}. "
                f"{factor} {guidance}"
            )

    if notes:
        lines.append(f"Note: {notes}")

    return " ".join(lines)


def build_jsonl_entry(file_name: str, data: dict, slot_label: str, slot_purpose: str) -> dict:
    """Build a single JSONL training entry for one image slot."""
    food_name = data.get("food_name", data.get("name", "this food"))
    prompt = (
        f"I have taken a photo of my meal ({slot_label}). "
        f"Please identify this food and provide: "
        f"(1) its name and a brief description, "
        f"(2) nutritional profile per 100g and per serving, "
        f"(3) glycemic index and glycemic load with source, "
        f"(4) health guidance for patients with hypertension, type 2 diabetes, "
        f"and rheumatoid arthritis."
    )
    return {
        "file_name": file_name,
        "image_slot": slot_label,
        "image_purpose": slot_purpose,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": build_assistant_response(data)}
                ]
            }
        ]
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_batch(input_folder: str, batch_num: int, val_split: float):
    input_path = Path(input_folder)
    json_files = sorted([
        f for f in input_path.glob("*.json")
        if not f.name.startswith("._")
    ])

    if not json_files:
        print(f"[ERROR] No JSON files found in {input_folder}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Processing batch {batch_num}: {len(json_files)} JSON files")
    print(f"Image strategy: 7 slots per food (expert recommendation)")
    print(f"{'='*60}\n")

    train_jsonl_path = HF_TRAIN_DIR / "metadata.jsonl"
    val_jsonl_path = HF_VAL_DIR / "metadata.jsonl"
    HF_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    HF_VAL_DIR.mkdir(parents=True, exist_ok=True)
    (HF_TRAIN_DIR / "images").mkdir(exist_ok=True)
    (HF_VAL_DIR / "images").mkdir(exist_ok=True)

    train_entries = []
    val_entries = []
    summary = []

    for idx, json_file in enumerate(json_files):
        global_record_num = (batch_num - 1) * 50 + idx + 1
        shard_name = get_shard_name(global_record_num)
        is_validation = (idx % round(1 / val_split) == 0) if val_split > 0 else False

        # Load JSON — skip list-type files
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            print(f"  [SKIP] {json_file.name} is a list-type JSON, skipping.")
            continue

        food_name = data.get("food_name", data.get("name", json_file.stem))
        food_slug = slugify(food_name)
        food_id = f"{global_record_num:04d}_{food_slug}"

        print(f"\n[{global_record_num:04d}] {food_name}")

        # Create raw_data folder and copy metadata
        food_dir = RAW_DATA_DIR / shard_name / food_id
        food_dir.mkdir(parents=True, exist_ok=True)
        dest_json = food_dir / "metadata.json"
        with open(dest_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Collect slot-based images
        slot_results = collect_slot_images(food_name, food_dir)
        downloaded = sum(1 for v in slot_results.values() if v)

        # Build JSONL entries for each successful slot
        hf_dir = HF_VAL_DIR if is_validation else HF_TRAIN_DIR
        for slot_def in IMAGE_SLOTS:
            slot = slot_def["slot"]
            if not slot_results.get(slot):
                continue
            src_img = food_dir / f"{slot}.jpg"
            hf_img_name = f"{food_id}_{slot}.jpg"
            hf_img_path = hf_dir / "images" / hf_img_name
            shutil.copy2(src_img, hf_img_path)

            entry = build_jsonl_entry(
                f"images/{hf_img_name}",
                data,
                slot_def["label"],
                slot_def["purpose"]
            )
            if is_validation:
                val_entries.append(entry)
            else:
                train_entries.append(entry)

        split_label = "validation" if is_validation else "train"
        print(f"  → {downloaded}/7 images collected → {split_label}")

        summary.append({
            "record_num": global_record_num,
            "food_name": food_name,
            "food_id": food_id,
            "shard": shard_name,
            "images_downloaded": downloaded,
            "slot_results": slot_results,
            "split": split_label
        })

    # Append to JSONL files
    with open(train_jsonl_path, "a", encoding="utf-8") as f:
        for entry in train_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(val_jsonl_path, "a", encoding="utf-8") as f:
        for entry in val_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Write batch summary
    summary_path = REPO_ROOT / f"batch_{batch_num:03d}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Batch {batch_num} complete.")
    print(f"  Train entries added : {len(train_entries)}")
    print(f"  Val entries added   : {len(val_entries)}")
    print(f"  Summary saved to    : {summary_path}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Innuir MedGemma Training Data Pipeline")
    parser.add_argument("--input", required=True, help="Path to folder containing JSON files for this batch")
    parser.add_argument("--batch_num", type=int, required=True, help="Batch number (1, 2, 3, ...)")
    parser.add_argument("--split", type=float, default=0.1, help="Validation split fraction (default: 0.1)")
    args = parser.parse_args()

    process_batch(args.input, args.batch_num, args.split)
