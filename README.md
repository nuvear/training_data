# Innuir Nutrition — MedGemma Training Data

This repository contains structured training data for fine-tuning **MedGemma 4B IT** (multimodal) for on-device deployment on iPhone via LoRA.

## Dataset Overview

| Attribute | Value |
| :--- | :--- |
| Total food records | 6,600 |
| Images per food item | 5–10 |
| Total training samples | ~33,000–66,000 |
| Image resolution | 896 × 896 px (JPEG) |
| Fine-tuning method | QLoRA (Quantized Low-Rank Adaptation) |
| Target model | MedGemma 4B IT |
| Deployment target | On-device iOS (iPhone) |

## Repository Structure

```
training_data/
├── README.md
├── scripts/
│   ├── process_batch.py        # Main pipeline: JSON → JSONL + image download
│   └── image_validator.py      # Validates all images are 896x896 JPEG
├── raw_data/
│   ├── batch_0001_to_0500/     # Sharded by 500 records
│   │   ├── 0001_hainanese_chicken_rice/
│   │   │   ├── metadata.json   # Rich nutritional database JSON
│   │   │   ├── img_01.jpg      # 896x896 training images
│   │   │   └── ... (up to img_10.jpg)
│   │   └── ...
│   └── ...
└── hf_dataset/
    ├── train/
    │   ├── metadata.jsonl      # Hugging Face fine-tuning format
    │   └── images/
    └── validation/
        ├── metadata.jsonl
        └── images/
```

## Processing a Batch

```bash
python3 scripts/process_batch.py --input /path/to/batch_json_folder --batch_num 1
```

## Image Specification

- **Resolution:** 896 × 896 px (square)
- **Format:** JPEG
- **Content:** Food only, no people
- **Count:** 5–10 diverse images per food item

## Fine-Tuning Format

Each entry in `metadata.jsonl` follows the MedGemma multimodal conversational format:

```json
{
  "file_name": "images/0001_hainanese_chicken_rice_img_01.jpg",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image"},
        {"type": "text", "text": "..."}
      ]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "..."}]
    }
  ]
}
```

## License

Proprietary — Innuir Health Pte. Ltd.
