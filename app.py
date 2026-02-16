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
        self.model = genai.GenerativeModel("gemini-3-flash-preview")

    def analyze(self, audio_path, language):

        file = genai.upload_file(audio_path)

        while file.state.name == "PROCESSING":
            time.sleep(2)
            file = genai.get_file(file.name)

        if file.state.name == "FAILED":
            return "HUMAN", 0.6, "Audio processing failed", [], []

        prompt = f"""
You are an advanced audio forensics expert specializing in AI voice detection.

Analyze this {language} audio and determine if it's AI_GENERATED or HUMAN.

PRIMARY ANALYSIS - Check these core indicators:
- Pitch consistency and micro-variations
- Emotional authenticity and variation
- Breathing patterns and natural pauses
- Prosody and speech rhythm
- Background noise characteristics
- Lip smacks, hesitations, filler words

EDGE CASE DETECTION - Also check for these scenarios and factor them into your analysis:

1. MIXED_LANGUAGE: Speaker switches between languages (code-switching). Should NOT cause AI classification.
2. ROBOTIC_HUMAN: Human who naturally speaks in monotone. Should still be HUMAN.
3. CLONED_VOICE: AI trying to mimic a specific human voice.
4. HIGH_QUALITY_TTS: Modern text-to-speech — smooth, no hesitations, perfect pronunciation.
5. ANIMAL/VEHICLE SOUNDS: Non-speech audio — classify as NON_HUMAN_AUDIO.
6. AUDIO QUALITY ISSUES: Background noise, silence-dominant, distorted, very short — lower confidence.
7. WHISPERED/EXTREME SPEECH: Whispered or extreme emotion.
8. SPEAKER CHARACTERISTICS: Child, elderly, speech impediment, accent — should NOT alone cause AI classification.
9. SYNTHETIC_WITH_NOISE: AI audio with artificially added noise.
10. REPEATED_PATTERNS: Unnaturally repeated phrases.
11. MULTIPLE_SPEAKERS: Multiple speakers present.
12. SCAM_INDICATORS: Scripted urgency, requests for personal info, too-perfect suspicious delivery.
13. DEEPFAKE_SUSPECTED: Potential deepfake voice synthesis.

CLASSIFICATION RULES:
- Non-speech audio (animals, vehicles, music) → "NON_HUMAN_AUDIO"
- Speech impediments, accents, monotone, age → should NOT alone cause AI classification
- Mixed language → should NOT affect classification
- Lower confidence when audio quality is poor

Return ONLY this JSON:

{{
  "classification": "AI_GENERATED or HUMAN or NON_HUMAN_AUDIO",
  "confidence": 0.0-1.0,
  "explanation": "concise reasoning for the classification",
  "edge_cases_detected": ["EDGE_CASE_CODE"],
  "audio_content_type": "SPEECH or ANIMAL or VEHICLE or MUSIC or ENVIRONMENTAL or MIXED or SILENCE",
  "reliability_score": 0.0-1.0,
  "scam_risk_score": 0.0-1.0,
  "warnings": ["short warning strings if any"]
}}
"""

        response = self.model.generate_content([file, prompt])

        text = response.text.strip()

        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()

        result = json.loads(text)

        genai.delete_file(file.name)

        cls = result.get("classification", "HUMAN")
        conf = float(result.get("confidence", 0.6))
        exp = result.get("explanation", "Analysis done")
        edge_cases = result.get("edge_cases_detected", [])
        audio_type = result.get("audio_content_type", "SPEECH")
        reliability = float(result.get("reliability_score", 0.8))
        scam_score = float(result.get("scam_risk_score", 0.0))
        warnings = result.get("warnings", [])

        conf = min(max(conf, 0.0), 1.0)

        cls, conf, warnings = self._apply_edge_case_logic(
            cls, conf, edge_cases, audio_type, reliability, scam_score, warnings
        )

        return cls, round(conf, 2), exp, edge_cases, warnings

    def _apply_edge_case_logic(self, cls, conf, edge_cases, audio_type, reliability, scam_score, warnings):
        """Internally adjust classification based on detected edge cases."""

        # Handle non-human audio
        if audio_type in ["ANIMAL", "VEHICLE", "ENVIRONMENTAL"]:
            if cls != "NON_HUMAN_AUDIO":
                cls = "NON_HUMAN_AUDIO"

        # Reduce confidence for quality issues
        quality_issues = ["DISTORTED_AUDIO", "SILENCE_DOMINANT", "VERY_SHORT_AUDIO", "ENVIRONMENTAL_NOISE"]
        for issue in quality_issues:
            if issue in edge_cases:
                conf = max(conf - 0.15, 0.3)

        # Robotic human misclassified as AI
        if "ROBOTIC_HUMAN" in edge_cases and cls == "AI_GENERATED":
            human_hints = ["SPEECH_IMPEDIMENT", "EMOTIONAL_EXTREME", "ACCENT_HEAVY", "ELDERLY_VOICE"]
            if any(h in edge_cases for h in human_hints):
                conf = max(conf - 0.2, 0.4)
                warnings.append("Human indicators present — manual review recommended")

        # Deepfake / clone + scam
        if ("DEEPFAKE_SUSPECTED" in edge_cases or "CLONED_VOICE" in edge_cases) and scam_score > 0.6:
            warnings.append("Potential deepfake with high scam risk")

        # Scam indicators
        if "SCAM_INDICATORS" in edge_cases:
            if scam_score > 0.7:
                warnings.append("High scam risk detected")
            elif scam_score > 0.4:
                warnings.append("Moderate scam risk detected")

        # Synthetic with noise
        if "SYNTHETIC_WITH_NOISE" in edge_cases and cls == "HUMAN":
            conf = max(conf - 0.1, 0.5)
            warnings.append("Possibly AI with added imperfections")

        # Repeated patterns
        if "REPEATED_PATTERNS" in edge_cases and cls == "HUMAN":
            conf = max(conf - 0.15, 0.4)
            warnings.append("Unnatural repetition detected")

        return cls, conf, warnings

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

    # Size check
    audio_size_kb = len(audio_bytes) / 1024
    size_warning = None
    if audio_size_kb < 10:
        size_warning = "Very small audio file — reliability may be reduced"
    elif audio_size_kb > 50000:
        size_warning = "Very large audio file — processing may be slower"

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        path = f.name

    try:
        detector = VoiceDetector()

        cls, conf, exp, edge_cases, warnings = detector.analyze(path, data["language"])

        if size_warning:
            warnings.append(size_warning)

        response = {
            "status": "success",
            "language": data["language"],
            "classification": cls,
            "confidenceScore": conf,
            "explanation": exp
        }

        # Include edge cases and warnings only when present
        if edge_cases:
            response["edgeCases"] = edge_cases

        if warnings:
            response["warnings"] = warnings

        return jsonify(response)

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

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )