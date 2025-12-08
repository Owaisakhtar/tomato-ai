import os
import datetime
import numpy as np
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
from gtts import gTTS
from huggingface_hub import hf_hub_download

# Local imports
from database import get_db_connection
from auth import hash_password, verify_password, create_jwt, get_current_user

app = FastAPI()
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
templates = Jinja2Templates(directory="templates")

# -----------------------------
# LAZY MODEL LOADING
# -----------------------------
model = None

def get_model():
    """Load model only when needed."""
    global model
    if model is None:
        HF_TOKEN = os.environ.get("HF_TOKEN")
        model_path = hf_hub_download(
            repo_id="abdullahzunorain/tomato_leaf_disease_det_model_v1",
            filename="best_model.h5",
            token=HF_TOKEN
        )
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        print("Model loaded successfully!")
    return model

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

def text_to_audio(text, filename):
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    audio_path = f"uploads/{filename}.mp3"
    gTTS(text=text, lang="en").save(audio_path)
    return audio_path

# -----------------------------
# ROUTES
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/signup", response_class=HTMLResponse)
def signup(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/signup")
def signup_post(username: str = Form(...), password: str = Form(...)):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        hashed = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
            (username, hashed)
        )
        conn.commit()
        return {"success": True, "message": "Account created successfully!"}
    except Exception as e:
        return {"success": False, "message": str(e)}
    finally:
        if conn:
            conn.close()

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
# PREDICT
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), user_id: int = Form(...)):

    # Load model now (lazy)
    model = get_model()

    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb+") as f:
        f.write(await file.read())

    img = Image.open(file_path).resize((256, 256))
    img = np.array(img) / 255.0
    img = img.reshape((1, 256, 256, 3))

    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    label = CLASS_NAMES[class_id]

    advice = ADVICE_MAP[label]
    audio_path = text_to_audio(advice, file.filename.split(".")[0])

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO history (user_id, filename, prediction, advice, audio_path, upload_date)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (user_id, file.filename, label, advice, audio_path, datetime.datetime.now()))
    conn.commit()
    conn.close()

    return {"filename": file.filename, "prediction": label, "advice": advice, "audio_file": audio_path}

@app.get("/history/{user_id}")
def history(user_id: int, token: str):
    auth_user = get_current_user(token)
    if auth_user != user_id:
        return {"error": "Unauthorized"}

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT filename, prediction, advice, audio_path, upload_date FROM history WHERE user_id=%s",
        (user_id,)
    )
    rows = cursor.fetchall()
    conn.close()

    history_list = []
    for row in rows:
        history_list.append([
            row[0], row[1], row[2], row[3], row[4].strftime("%Y-%m-%d %H:%M:%S")
        ])

    return {"history": history_list}

# -----------------------------
# RUN ON RAILWAY
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    PORT = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
