import os
import base64
import tempfile
import time
import json
from functools import wraps

from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai

# ---------------- LOAD ENV (Local Only) ---------------- #

load_dotenv()

# ---------------- CONFIG ---------------- #

API_KEY = os.getenv("API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set")

if not API_KEY:
    print("WARNING: API_KEY not set")

genai.configure(api_key=GEMINI_API_KEY)

SUPPORTED_LANGUAGES = [
    "Tamil",
    "English",
    "Hindi",
    "Malayalam",
    "Telugu"
]

app = Flask(__name__)

# ---------------- SECURITY ---------------- #

def require_api_key(f):
    @wraps(f)
    def wrapper(*args, **kwargs):

        key = request.headers.get("x-api-key")

        if not key or key != API_KEY:
            return jsonify({
                "status": "error",
                "message": "Invalid API key"
            }), 401

        return f(*args, **kwargs)

    return wrapper

# ---------------- DETECTOR ---------------- #

class VoiceDetector:

    def __init__(self):
        # Stable model for production
        self.model = genai.GenerativeModel("gemini-3-flash-preview")

    def analyze(self, audio_path, language):

        file = genai.upload_file(audio_path)

        while file.state.name == "PROCESSING":
            time.sleep(2)
            file = genai.get_file(file.name)

        if file.state.name == "FAILED":
            return "HUMAN", 0.6, "Audio processing failed"

        prompt = f"""
You are an audio forensics expert.

Analyze this {language} audio and decide:

AI_GENERATED or HUMAN

Check:
- Pitch
- Emotion
- Breathing
- Naturalness
- Prosody

Return ONLY JSON:

{{
  "classification":"AI_GENERATED or HUMAN",
  "confidence":0.0-1.0,
  "explanation":"short reason"
}}
"""

        response = self.model.generate_content([file, prompt])

        text = response.text.strip()

        # Remove markdown if present
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()

        result = json.loads(text)

        genai.delete_file(file.name)

        cls = result.get("classification", "HUMAN")
        conf = float(result.get("confidence", 0.6))
        exp = result.get("explanation", "Analysis done")

        conf = min(max(conf, 0.0), 1.0)

        return cls, round(conf, 2), exp

# ---------------- VALIDATION ---------------- #

def validate(data):

    if not data:
        return "Invalid JSON"

    if "language" not in data:
        return "Missing language"

    if data["language"] not in SUPPORTED_LANGUAGES:
        return "Unsupported language"

    if "audioFormat" not in data:
        return "Missing audioFormat"

    if data["audioFormat"].lower() != "mp3":
        return "Only MP3 supported"

    if "audioBase64" not in data:
        return "Missing audioBase64"

    if not data["audioBase64"]:
        return "Empty audio"

    return None

# ---------------- API ---------------- #

@app.route("/api/voice-detection", methods=["POST"])
@require_api_key
def detect():

    data = request.get_json()

    error = validate(data)

    if error:
        return jsonify({
            "status": "error",
            "message": error
        }), 400

    try:
        audio_bytes = base64.b64decode(data["audioBase64"])
    except:
        return jsonify({
            "status": "error",
            "message": "Invalid base64"
        }), 400

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        path = f.name

    try:
        detector = VoiceDetector()

        cls, conf, exp = detector.analyze(
            path,
            data["language"]
        )

        return jsonify({
            "status": "success",
            "language": data["language"],
            "classification": cls,
            "confidenceScore": conf,
            "explanation": exp
        })

    finally:
        if os.path.exists(path):
            os.remove(path)

# ---------------- HEALTH ---------------- #

@app.route("/health")
def health():

    return jsonify({
        "status": "healthy",
        "service": "voice-detection-api"
    })

# ---------------- RUN ---------------- #

if __name__ == "__main__":

    # Render provides PORT automatically
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
