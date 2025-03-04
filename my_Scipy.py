# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Load the serialized model from the "model2.pkl" file
# with open("model2.pkl", "rb") as f:
#     loaded_model = pickle.load(f)

# # Define the labels mapping (customize this based on your actual labels)
# labels = {0: 'Dumb', 1: 'Smart', 2: 'Oversmart'}

# # Input text for classification
# input_text = ["My name is Deepak"]  # Note the input as a list

# # Create and fit the TF-IDF vectorizer on your training data
# tfidf_vectorizer = TfidfVectorizer()
# training_data = ["I am Oversmart idiot"]  # Replace with your actual training data
# tfidf_vectorizer.fit(training_data)

# # Use the fitted vectorizer to transform the input text
# X_input = tfidf_vectorizer.transform(input_text)

# # Use the loaded model to make predictions
# predicted_class_indices = loaded_model.predict(X_input)
# print(predicted_class_indices)
# # Map predicted class index to label
# predicted_label = labels.get(predicted_class_indices[0])

# # Print the predicted label
# print("Predicted category:", predicted_label)


import os
import cv2
import pytesseract
import re
from PIL import Image

# Set Tesseract OCR path (only needed for Windows users)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Function to preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # Apply thresholding
    return thresh

# Function to extract text using OCR
def extract_text(image_path):
    processed_image = preprocess_image(image_path)
    text = pytesseract.image_to_string(processed_image)
    return text

# Function to extract URLs from text
def extract_urls(text):
    url_pattern = r"https?://[^\s]+"  # Regex pattern for URLs
    urls = re.findall(url_pattern, text)
    return urls

# Function to process all screenshots in a folder
def process_screenshots(folder_path):
    all_urls = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            image_path = os.path.join(folder_path, file_name)
            text = extract_text(image_path)
            urls = extract_urls(text)
            all_urls.extend(urls)
            print(f"URLs found in {file_name}: {urls}")
    return all_urls

# Run the script for the "screenshots" folder
screenshot_folder = "screenshots"  # Change this to your actual folder path
urls_found = process_screenshots(screenshot_folder)

# Print final extracted URLs
print("\nAll Extracted URLs:")
for url in urls_found:
    print(url)
