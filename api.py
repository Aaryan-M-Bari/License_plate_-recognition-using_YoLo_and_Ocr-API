import os
import re
import json
import cv2
from ultralytics import YOLO
import easyocr
from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np

# --- CONFIGURATION & MODEL LOADING (Happens once on startup) ---
YOLO_MODEL_PATH = './license_plate_detector.pt'
DATABASE_FILE = 'plates_database.json'

print("Initializing models... This may take a moment.")
app = FastAPI(title="License Plate Verification API")

try:
    # Load YOLO model
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("✅ YOLO model loaded successfully.")
    
    # Load EasyOCR reader - IMPORTANT: set gpu=True for cloud deployment
    reader = easyocr.Reader(['en'], gpu=True) # Use GPU in the cloud
    print("✅ EasyOCR reader loaded successfully.")

except Exception as e:
    yolo_model = None
    reader = None
    print(f"❌ Critical error loading models: {e}")

# --- DATABASE MANAGEMENT ---
def load_vehicle_data():
    """Loads vehicle data from the JSON file."""
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, 'r') as f:
            return json.load(f)
    print("WARNING: database_file.json not found. Starting with an empty database.")
    return {}

VEHICLE_DATA = load_vehicle_data()

# --- HELPER FUNCTIONS ---
def clean_text(raw_text):
    """Cleans OCR text to keep only uppercase letters and numbers."""
    return "".join(re.findall(r'[A-Z0-9]', raw_text.upper()))

def process_image_and_get_text(image_bytes):
    """Core logic to process an image and return cleaned plate text."""
    if not yolo_model or not reader:
        raise HTTPException(status_code=503, detail="Models are not available.")

    try:
        # Convert image bytes to a NumPy array that OpenCV can read
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 1. Detect plate
        results = yolo_model(img)
        if len(results[0].boxes) == 0:
            return None # No plate detected

        # 2. Crop plate
        box = results[0].boxes[0].xyxy[0]
        x1, y1, x2, y2 = map(int, box)
        cropped_plate = img[y1:y2, x1:x2]

        # 3. Read text
        ocr_result = reader.readtext(cropped_plate, detail=0)
        if not ocr_result:
            return None # No text read

        # 4. Clean and return text
        raw_text = "".join(ocr_result)
        return clean_text(raw_text)

    except Exception as e:
        print(f"Error during image processing: {e}")
        raise HTTPException(status_code=500, detail="Error processing the image.")


# --- API ENDPOINT ---
@app.post("/verify-plate/")
async def verify_plate(file: UploadFile = File(...)):
    """
    Accepts an image file, reads the license plate, and verifies it against the database.
    """
    # Read the image content from the uploaded file
    image_content = await file.read()
    
    # Process the image to get the plate number
    plate_text = process_image_and_get_text(image_content)
    
    if not plate_text:
        raise HTTPException(status_code=404, detail="No license plate found or text could not be read.")
    
    # Check if the plate text exists in our database
    if plate_text in VEHICLE_DATA:
        details = VEHICLE_DATA[plate_text]
        return {
            "status": "match_found",
            "plate_number": plate_text,
            "details": details
        }
    else:
        return {
            "status": "no_match_found",
            "plate_number": plate_text
        }

# To run this API locally:
# 1. Open your terminal.
# 2. Run the command: uvicorn api:app --reload
# 3. Open your browser to http://127.0.0.1:8000/docs