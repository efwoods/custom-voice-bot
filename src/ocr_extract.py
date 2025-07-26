import json
import os
from pathlib import Path
from typing import List, Dict
import cv2
import pytesseract
from PIL import Image
from tqdm import tqdm
import argparse


def extract_left_half_text(img_path: str, left_ratio: float = 0.55) -> str:
    """
    Very simple heuristic:
    - Assume your mom's messages are on the left of each screenshot.
    - Crop left_ratio width of the image and OCR that region.
    For higher accuracy, you can later segment bubbles by color/contour.
    """
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    left_img = img[:, : int(w * left_ratio)]
    gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    # OCR
    config = "--psm 6"  # assume uniform block of text
    text = pytesseract.image_to_string(gray, config=config)
    return text


def process_images(input_dir: str, output_file: str, min_chars: int = 5):
    records: List[Dict] = []
    paths = sorted(list(Path(input_dir).glob("*.*")))
    for p in tqdm(paths, desc="OCR-ing left side"):
        text = extract_left_half_text(str(p))
        # crude cleaning
        text = text.strip()
        if len(text) >= min_chars:
            records.append({"source_image": p.name, "text": text})
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} mom messages to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/raw_images")
    parser.add_argument(
        "--output_file", type=str, default="data/processed/mom_raw.jsonl"
    )
    args = parser.parse_args()
    process_images(args.input_dir, args.output_file)
