from flask import Flask, request, jsonify
import os
import pytesseract
from PIL import Image
import PyPDF2
import cv2
import numpy as np
import google.generativeai as genai
import requests
from io import BytesIO

app = Flask(__name__)

# Function to download file from URL
def download_file(url):
    print(f"Downloading file from URL: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        print("File downloaded successfully.")
        content_type = response.headers.get('content-type')
        return BytesIO(response.content), content_type
    else:
        raise Exception(f"Failed to download file from URL: {url}")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() or ""  # Ensure text is not None
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

# OCR Preprocessing functions
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def remove_noise(image):
    return cv2.medianBlur(image, 5)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)
    
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def canny(image):
    return cv2.Canny(image, 100, 200)

def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

# Function to extract text from image using OCR with preprocessing
def extract_text_from_image(image_file):
    try:
        img = Image.open(image_file)
        img = np.array(img)

        gray = get_grayscale(img)
        thresh = thresholding(gray)
        opened = opening(gray)
        edges = canny(gray)
        deskewed = deskew(gray)

        text_thresh = pytesseract.image_to_string(thresh)
        text_opened = pytesseract.image_to_string(opened)
        text_edges = pytesseract.image_to_string(edges)
        text_deskewed = pytesseract.image_to_string(deskewed)

        # Combine all extracted texts
        combined_text = "\n".join([text_thresh, text_opened, text_edges, text_deskewed])
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        combined_text = ""
    return combined_text

# Function to process each file and extract text
def process_file(file_url):
    file, content_type = download_file(file_url)
    if 'pdf' in content_type:
        print("Processing PDF:", file_url)
        return extract_text_from_pdf(file)
    elif 'image' in content_type:
        print("Processing Image:", file_url)
        return extract_text_from_image(file)
    else:
        print("Unsupported file format:", file_url)
        return ""

# Initialize Google Generative AI
GOOGLE_API_KEY = "AIzaSyB1tpMueN_3bPbnQGsNOYP7s_NvzrUEtcM"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro-latest')

@app.route('/api/process', methods=['POST'])
def process():
    data = request.get_json()
    file_url = data.get('url')
    all_extracted_text = process_file(file_url)

    prompt = f"""
    Please read and understand the provided context and extract all the tested parameters along with their values. Ensure that the parameters and their values are presented in a JSON object format.

    Context:
    {all_extracted_text}

    Task:
    1. Identify all the test parameters mentioned in the context.
    2. Extract the corresponding values for each parameter.
    3. Format the extracted parameters and values as a JSON object.

    Example format:
    {{
        "all_relevant_information_related_to_hospital/clinic/lab": "value",
        "all_relevant_information_related_to_patient": "value",
        "parameter1": "value1",
        "parameter2": "value2",
        ...
    }}

    Extracted parameters and values:
    """

    response = model.generate_content(prompt)
    result = ''.join([p.text for p in response.candidates[0].content.parts])
    return jsonify(result)

if __name__ == "__main__":
    app.run()
