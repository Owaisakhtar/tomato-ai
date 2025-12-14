import os
import datetime
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download
from PIL import Image
import pyttsx3

# Local imports
from database import get_db_connection
from auth import hash_password, verify_password, create_jwt, get_current_user

app = FastAPI()

# -----------------------------
# Mount static files and templates
# -----------------------------
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
templates = Jinja2Templates(directory="templates")

# -----------------------------
# Global model variable
@app.on_event("startup")
def startup_event():
    global model

    os.makedirs("uploads", exist_ok=True)

    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN not set")

    print("⬇ Downloading model from Hugging Face...")

    MODEL_PATH = hf_hub_download(
        repo_id="abdullahzunorain/tomato_leaf_disease_det_model_v1",
        filename="best_model.h5"
    )

    print("📏 File size:", os.path.getsize(MODEL_PATH))

    # Fix for "Unrecognized keyword arguments: ['batch_shape']"
    from keras.saving.legacy.serialization import load_model_from_hdf5
    model = load_model_from_hdf5(MODEL_PATH)
    print("✅ Model loaded successfully")

print("📏 File size:", os.path.getsize(MODEL_PATH))

# Fix for "Unrecognized keyword arguments: ['batch_shape']"
from keras.saving.legacy.serialization import load_model_from_hdf5
model = load_model_from_hdf5(MODEL_PATH)
print("✅ Model loaded successfully")
)


# -----------------------------
# CLASS LABELS
# -----------------------------
CLASS_NAMES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_YellowLeaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy"
]

# -----------------------------
# DISEASE ADVICE
# -----------------------------
ADVICE_MAP = {
    "Tomato_Bacterial_spot": "Bacterial spot detected. Remove affected leaves and spray with copper-based bactericide.",
    "Tomato_Early_blight": "Early Blight detected. Remove infected leaves and apply copper-based fungicide.",
    "Tomato_Late_blight": "Late Blight detected. Use fungicide immediately and avoid overhead watering.",
    "Tomato_Leaf_Mold": "Leaf Mold detected. Improve ventilation and avoid moisture on leaves.",
    "Tomato_Septoria_leaf_spot": "Septoria Leaf Spot detected. Remove affected leaves and apply protective fungicide.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Spider mites detected. Use insecticidal soap or neem oil to control.",
    "Tomato_Target_Spot": "Target Spot detected. Remove infected leaves and apply fungicide.",
    "Tomato_Tomato_YellowLeaf_Curl_Virus": "Yellow Leaf Curl Virus detected. Remove affected plants and control whiteflies.",
    "Tomato_Tomato_mosaic_virus": "Mosaic Virus detected. Remove affected plants and disinfect tools.",
    "Tomato_healthy": "Your plant is healthy! No action is needed."
}

def generate_advice(label):
    return ADVICE_MAP.get(label, "Unable to provide advice.")

# -----------------------------
# TEXT → AUDIO
# -----------------------------
def text_to_audio(text, filename):
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    audio_path = f"uploads/{filename}.mp3"
    engine = pyttsx3.init()
    engine.save_to_file(text, audio_path)
    engine.runAndWait()
    return audio_path

# -----------------------------
# UI ROUTES
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/signup", response_class=HTMLResponse)
def signup(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

# -----------------------------
# AUTH ROUTES
# -----------------------------
@app.post("/signup")
def signup_user(username: str = Form(...), password: str = Form(...)):
    conn = get_db_connection()
    cursor = conn.cursor()

    hashed = hash_password(password)
    cursor.execute(
        "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
        (username, hashed)
    )
    conn.commit()
    conn.close()

    return {"success": True}

@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, password_hash FROM users WHERE username=%s", (username,))
    user = cursor.fetchone()
    conn.close()

    if not user:
        return {"error": "User not found"}

    user_id, stored_hash = user
    if verify_password(password, stored_hash):
        token = create_jwt({"user_id": user_id})
        return {"success": True, "user_id": user_id, "token": token}

    return {"error": "Invalid password"}

@app.get("/dashboard/{user_id}", response_class=HTMLResponse)
def dashboard(user_id: int, request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request, "user_id": user_id})

# -----------------------------
# PREDICTION API
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), user_id: int = Form(...)):
    global model

    if model is None:
        return {"error": "Model not loaded yet"}

    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    img = Image.open(file_path).resize((256, 256))
    img = np.array(img) / 255.0
    img = img.reshape((1, 256, 256, 3))

    prediction = model.predict(img)
    label = CLASS_NAMES[np.argmax(prediction)]

    advice = generate_advice(label)
    audio_path = text_to_audio(advice, file.filename.split(".")[0])

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO history (user_id, filename, prediction, advice, audio_path, upload_date) "
        "VALUES (%s,%s,%s,%s,%s,%s)",
        (user_id, file.filename, label, advice, audio_path, datetime.datetime.now())
    )
    conn.commit()
    conn.close()

    return {
        "filename": file.filename,
        "prediction": label,
        "advice": advice,
        "audio_file": audio_path
    }


# -----------------------------
# USER HISTORY
# -----------------------------
@app.get("/history/{user_id}")
def history(user_id: int, token: str):
    if get_current_user(token) != user_id:
        return {"error": "Unauthorized"}

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT filename, prediction, advice, audio_path, upload_date FROM history WHERE user_id=%s",
        (user_id,)
    )
    rows = cursor.fetchall()
    conn.close()

    return {
        "history": [
            [r[0], r[1], r[2], r[3], r[4].strftime("%Y-%m-%d %H:%M:%S")]
            for r in rows
        ]
    }

