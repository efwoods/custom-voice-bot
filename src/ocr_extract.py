import cv2
import pytesseract
import json
import os
from pathlib import Path
from tqdm import tqdm
import argparse


def extract_mom_bubbles(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return ""

    # Convert to HSV color space for better color filtering
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for grey bubbles (tune these values to match your image bubble color)
    lower_grey = (0, 0, 120)
    upper_grey = (180, 50, 220)

    # Create mask for grey areas
    mask = cv2.inRange(hsv, lower_grey, upper_grey)

    # Crop left side of image (assumes left side always contains mom's messages)
    h, w = mask.shape
    left_crop = mask[:, : w // 2]
    original_crop = image[:, : w // 2]

    # Use mask to isolate text regions
    result = cv2.bitwise_and(original_crop, original_crop, mask=left_crop)

    # Run OCR only on the left grey bubbles
    config = "--psm 6"
    text = pytesseract.image_to_string(result, config=config)

    return text.strip()


def process_directory(input_dir, output_file):
    input_path = Path(input_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for image_file in tqdm(
        sorted(input_path.glob("*.PNG")) + sorted(input_path.glob("*.jpg"))
    ):
        mom_text = extract_mom_bubbles(image_file)
        if mom_text:
            results.append({"filename": image_file.name, "text": mom_text})

    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Extracted {len(results)} entries to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Path to directory of images"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to save .jsonl output"
    )
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_file)
