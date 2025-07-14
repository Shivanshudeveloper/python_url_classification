import pytesseract as pyt
import cv2
import os
import re

# Path to Tesseract-OCR
pyt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Folder containing screenshots
screenshot_folder = "./screenshots"

# Regex pattern to match URLs (focus on patterns after 'Q' or similar)
url_pattern = re.compile(
    r'Q\s*[=\d%]*\s*([\w.-]+\.(?:com|in|net|org|edu|gov|io|ai|co|uk|us|info|xyz|me)[^\s]*)',
    re.IGNORECASE
)


def extract_primary_tab_url(text):
    """Extracts the primary tab URL based on pattern."""
    match = url_pattern.search(text)
    return match.group(1) if match else None

def process_screenshots(folder):
    """Processes all images in the given folder"""
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            print(f"Processing: {img_path}")

            # Read image
            img = cv2.imread(img_path)

            # Perform OCR
            extracted_text = pyt.image_to_string(img)

            # Extract primary tab URL
            primary_url = extract_primary_tab_url(extracted_text)

            # print("\nExtracted Text:")
            print(extracted_text)
            print("\nPrimary Tab URL:")
            # print(primary_url)
            print(primary_url if primary_url else "No URL found")
            print("-" * 50)

if __name__ == "__main__":
    process_screenshots(screenshot_folder)
