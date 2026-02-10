from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import gdown
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "pcb_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=15NeEfT7106PH6RnolnhPdHWwHLMz49yC"

def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("Model downloaded!")
    
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded!")
    return model

model = load_model()

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "PCB Defect Detection API",
        "version": "1.0"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = float(model.predict(img_array, verbose=0)[0][0])
        
        if prediction > 0.5:
            result = "UNDEFECTIVE"
            confidence = prediction * 100
        else:
            result = "DEFECTIVE"
            confidence = (1 - prediction) * 100
        
        return {
            "prediction": result,
            "confidence": round(confidence, 2),
            "raw_score": round(prediction, 4)
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "prediction": "ERROR",
            "confidence": 0
        }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
