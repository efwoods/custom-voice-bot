import cv2
import numpy as np
import pytesseract
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt

# Color configuration
CHAT_BUBBLE_COLOR = "#2a2a2d"
CHAT_TEXT_COLOR = "#ffffff"


# Convert hex to RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_bgr(rgb):
    return (rgb[2], rgb[1], rgb[0])


# Loading Images
image_dir = "/home/linux-pc/gh/projects/NeuralNexus/New-Features/CustomLLM/custom-voice-bot/data/raw_images/"
image_list = sorted(glob(image_dir + "*.PNG"))
sample_image = image_list[0]


def extract_text_from_grey_bubbles(
    image_path, bubble_color="#2a2a2d", text_color="#ffffff", tolerance=20
):
    """
    Extract text from specific colored chat bubbles

    Args:
        image_path: Path to the image
        bubble_color: Hex color of the target bubble
        text_color: Hex color of the text (for validation)
        tolerance: Color tolerance for matching (0-255)

    Returns:
        extracted_text: String containing all extracted text
        processed_image: Image showing detected regions (for debugging)
    """

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert colors
    target_bgr = rgb_to_bgr(hex_to_rgb(bubble_color))

    # Create mask for the target bubble color
    lower_bound = np.array([max(0, c - tolerance) for c in target_bgr])
    upper_bound = np.array([min(255, c + tolerance) for c in target_bgr])

    # Create color mask
    bubble_mask = cv2.inRange(image, lower_bound, upper_bound)

    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bubble_mask = cv2.morphologyEx(bubble_mask, cv2.MORPH_CLOSE, kernel)
    bubble_mask = cv2.morphologyEx(bubble_mask, cv2.MORPH_OPEN, kernel)

    # Find contours of bubble regions
    contours, _ = cv2.findContours(
        bubble_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter contours by area (remove small noise)
    min_area = 1000  # Adjust based on your bubble sizes
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    extracted_texts = []
    debug_image = image.copy()

    for i, contour in enumerate(valid_contours):
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Add padding around the bubble
        padding = 10
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)

        # Extract region of interest
        roi = image[y_start:y_end, x_start:x_end]

        # Convert to PIL Image for pytesseract
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

        # OCR configuration for white text on dark background
        custom_config = custom_config = (
            """--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?:;-()[]{} """
        )

        # Extract text
        text = pytesseract.image_to_string(roi_pil, config=custom_config).strip()

        if text:  # Only add non-empty text
            extracted_texts.append(text)
            print(f"Bubble {i+1}: {text}")

        # Draw bounding box on debug image
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            debug_image,
            f"Bubble {i+1}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    return "\n".join(extracted_texts), debug_image


def process_image_enhanced(image_path, bubble_color="#2a2a2d", tolerance=20):
    """
    Enhanced version with better preprocessing for OCR
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert to different color spaces for better segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create mask using multiple approaches
    target_bgr = rgb_to_bgr(hex_to_rgb(bubble_color))

    # Method 1: Direct color matching
    lower_bound = np.array([max(0, c - tolerance) for c in target_bgr])
    upper_bound = np.array([min(255, c + tolerance) for c in target_bgr])
    mask1 = cv2.inRange(image, lower_bound, upper_bound)

    # Method 2: HSV-based matching (more robust to lighting)
    target_hsv = cv2.cvtColor(np.uint8([[target_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    lower_hsv = np.array([max(0, target_hsv[0] - 10), 50, 50])
    upper_hsv = np.array([min(179, target_hsv[0] + 10), 255, 255])
    mask2 = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Combine masks
    combined_mask = cv2.bitwise_or(mask1, mask2)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter and sort contours
    min_area = 1000
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    valid_contours = sorted(
        valid_contours, key=lambda x: cv2.boundingRect(x)[1]
    )  # Sort by y-coordinate

    extracted_texts = []
    debug_image = image.copy()

    for i, contour in enumerate(valid_contours):
        x, y, w, h = cv2.boundingRect(contour)

        # Extract ROI with padding
        padding = 15
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)

        roi = image[y_start:y_end, x_start:x_end]

        # Preprocessing for better OCR
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Enhance contrast
        roi_enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(
            roi_gray
        )

        # Convert to PIL
        roi_pil = Image.fromarray(roi_enhanced)

        # OCR with multiple configurations
        configs = [
            r"--oem 3 --psm 6",
            r"--oem 3 --psm 7",
            r"--oem 3 --psm 8",
        ]

        best_text = ""
        best_confidence = 0

        for config in configs:
            try:
                # Get text with confidence scores
                data = pytesseract.image_to_data(
                    roi_pil, config=config, output_type=pytesseract.Output.DICT
                )
                confidences = [int(conf) for conf in data["conf"] if int(conf) > 0]

                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    text = pytesseract.image_to_string(roi_pil, config=config).strip()

                    if avg_confidence > best_confidence and text:
                        best_confidence = avg_confidence
                        best_text = text
            except:
                continue

        if best_text:
            extracted_texts.append(best_text)
            print(f"Bubble {i+1} (confidence: {best_confidence:.1f}): {best_text}")

        # Draw on debug image
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            debug_image,
            f"B{i+1}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    return "\n".join(extracted_texts), debug_image, combined_mask


# Example usage
if __name__ == "__main__":
    try:
        # Process the sample image
        print("Processing image:", sample_image)

        # Basic extraction
        print("\n=== Basic Extraction ===")
        extracted_text, debug_img = extract_text_from_grey_bubbles(sample_image)
        print("Extracted text:")
        print(extracted_text)

        # Enhanced extraction
        print("\n=== Enhanced Extraction ===")
        enhanced_text, enhanced_debug, mask = process_image_enhanced(sample_image)
        print("Enhanced extracted text:")
        print(enhanced_text)

        # Display results (optional)
        # plt.figure(figsize=(15, 5))
        # plt.subplot(131)
        # plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        # plt.title('Basic Detection')
        # plt.axis('off')
        #
        # plt.subplot(132)
        # plt.imshow(cv2.cvtColor(enhanced_debug, cv2.COLOR_BGR2RGB))
        # plt.title('Enhanced Detection')
        # plt.axis('off')
        #
        # plt.subplot(133)
        # plt.imshow(mask, cmap='gray')
        # plt.title('Bubble Mask')
        # plt.axis('off')
        #
        # plt.tight_layout()
        # plt.show()

    except Exception as e:
        print(f"Error: {e}")


# Function to process all images in the directory
def process_all_images():
    """Process all images and save results"""
    results = {}

    for image_path in image_list:
        print(f"\nProcessing: {image_path}")
        try:
            text, _, _ = process_image_enhanced(image_path)
            results[image_path] = text
            print(
                f"Extracted: {text[:100]}..."
                if len(text) > 100
                else f"Extracted: {text}"
            )
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results[image_path] = ""

    return results


# Uncomment to process all images
# all_results = process_all_images()
