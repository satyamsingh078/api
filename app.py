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

# ---------------- EDGE CASE DEFINITIONS ---------------- #

EDGE_CASE_DEFINITIONS = {
    "MIXED_LANGUAGE": "Audio contains multiple languages (code-switching)",
    "ROBOTIC_HUMAN": "Human speaker with unusually monotone or flat delivery",
    "CLONED_VOICE": "AI attempting to clone a specific human voice",
    "HIGH_QUALITY_TTS": "Modern text-to-speech that closely mimics human speech",
    "ANIMAL_SOUND_REAL": "Genuine animal vocalizations",
    "ANIMAL_SOUND_AI": "AI-generated animal sounds",
    "VEHICLE_SOUND_REAL": "Genuine vehicle/mechanical sounds",
    "VEHICLE_SOUND_AI": "AI-generated vehicle/mechanical sounds",
    "ENVIRONMENTAL_NOISE": "Significant background noise affecting analysis",
    "SILENCE_DOMINANT": "Audio is mostly silent or near-silent",
    "VERY_SHORT_AUDIO": "Audio too short for reliable analysis",
    "DISTORTED_AUDIO": "Audio quality severely degraded",
    "WHISPERED_SPEECH": "Whispered or very soft speech",
    "EMOTIONAL_EXTREME": "Extreme emotional speech (crying, shouting, laughing)",
    "CHILD_VOICE": "Child speaker with different vocal characteristics",
    "ELDERLY_VOICE": "Elderly speaker with potential vocal tremors",
    "SPEECH_IMPEDIMENT": "Speaker with speech disorder or impediment",
    "SYNTHETIC_WITH_NOISE": "AI audio with artificially added noise/imperfections",
    "REPEATED_PATTERNS": "Unnaturally repeated phrases or patterns",
    "NON_SPEECH_VOCAL": "Non-speech vocalizations (humming, coughing, sighing)",
    "MUSIC_WITH_VOCALS": "Music track containing singing",
    "ACCENT_HEAVY": "Strong regional accent affecting analysis",
    "MULTIPLE_SPEAKERS": "Multiple speakers in the same audio",
    "SCAM_INDICATORS": "Patterns common in scam/fraud attempts",
    "DEEPFAKE_SUSPECTED": "Potential deepfake voice synthesis detected"
}

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
            return "HUMAN", 0.6, "Audio processing failed", [], {}

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

EDGE CASE DETECTION - Identify if ANY of these apply:

1. MIXED_LANGUAGE: Does the speaker switch between languages (code-switching)?
   - Common in multilingual regions (Tamil-English, Hindi-English mix)
   - Should NOT affect human/AI classification

2. ROBOTIC_HUMAN: Is this a HUMAN who naturally speaks in monotone?
   - Professional voice-over artists often sound "perfect"
   - Some people have flat affect or speech patterns
   - Look for subtle human imperfections despite robotic tone

3. CLONED_VOICE: Does this sound like AI trying to mimic a specific person?
   - Slightly "off" quality in emotional expressions
   - Inconsistent accent or speech patterns

4. HIGH_QUALITY_TTS: Is this modern text-to-speech?
   - Very smooth, no hesitations
   - Perfect pronunciation
   - Lacks micro-expressions in voice

5. ANIMAL_SOUNDS: Is this primarily animal vocalizations?
   - Classify as ANIMAL_SOUND_REAL or ANIMAL_SOUND_AI
   - AI animal sounds often have unnatural repetition or pitch

6. VEHICLE/MECHANICAL SOUNDS: Is this vehicle or machine audio?
   - Classify as VEHICLE_SOUND_REAL or VEHICLE_SOUND_AI
   - AI-generated mechanical sounds lack authentic acoustic signatures

7. ENVIRONMENTAL_NOISE: Is background noise affecting analysis reliability?

8. SILENCE_DOMINANT: Is the audio mostly silent (>70% silence)?

9. DISTORTED_AUDIO: Is audio quality too poor for reliable analysis?

10. WHISPERED_SPEECH: Is the speech whispered or very soft?

11. EMOTIONAL_EXTREME: Is there extreme emotion (crying, shouting, laughing)?
    - Humans show more irregular patterns
    - AI struggles with authentic emotional extremes

12. CHILD_VOICE: Is this a child speaking?
    - Different vocal characteristics than adults

13. ELDERLY_VOICE: Is this an elderly speaker?
    - May have natural tremors that could seem "robotic"

14. SPEECH_IMPEDIMENT: Does speaker have stutter, lisp, or other speech pattern?
    - Should NOT be classified as AI due to irregularities

15. SYNTHETIC_WITH_NOISE: Is this AI audio with artificially added imperfections?
    - Noise seems "layered on" rather than natural
    - Imperfections don't match acoustic environment

16. REPEATED_PATTERNS: Are there unnaturally repeated phrases or patterns?
    - Common in AI-generated content
    - Humans rarely repeat exactly the same way

17. NON_SPEECH_VOCAL: Is this non-speech (humming, coughing, sighing)?

18. MUSIC_WITH_VOCALS: Is this a song or music with singing?

19. ACCENT_HEAVY: Is there a strong regional accent?
    - Should NOT affect classification

20. MULTIPLE_SPEAKERS: Are there multiple speakers?

21. SCAM_INDICATORS: Are there patterns common in scam attempts?
    - Overly polite/formal scripted language
    - Urgency phrases
    - Request patterns for personal info
    - Too-perfect delivery of suspicious content

22. DEEPFAKE_SUSPECTED: Does this seem like a deepfake of a known voice?

Return ONLY this JSON structure:

{{
  "classification": "AI_GENERATED or HUMAN or NON_HUMAN_AUDIO",
  "confidence": 0.0-1.0,
  "explanation": "detailed reasoning for classification",
  "edge_cases_detected": ["list", "of", "edge", "case", "codes"],
  "edge_case_details": {{
    "EDGE_CASE_CODE": "specific observation about this edge case"
  }},
  "audio_content_type": "SPEECH or ANIMAL or VEHICLE or MUSIC or ENVIRONMENTAL or MIXED or SILENCE",
  "reliability_score": 0.0-1.0,
  "reliability_notes": "any factors affecting analysis reliability",
  "scam_risk_score": 0.0-1.0,
  "warnings": ["any", "important", "warnings"]
}}

IMPORTANT CLASSIFICATION RULES:
- If audio is primarily non-speech (animals, vehicles, music), use "NON_HUMAN_AUDIO"
- For non-speech audio, still indicate if it's real or AI-generated in edge_cases
- Lower confidence if edge cases significantly affect reliability
- ROBOTIC_HUMAN should still be classified as HUMAN if genuine human indicators exist
- Speech impediments, accents, or monotone delivery alone should NOT cause AI classification
- Mixed language should NOT affect classification
- If reliability is very low (<0.4), explicitly warn in the response
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
        edge_cases = result.get("edge_cases_detected", [])
        
        # Build comprehensive metadata
        metadata = {
            "edge_case_details": result.get("edge_case_details", {}),
            "audio_content_type": result.get("audio_content_type", "SPEECH"),
            "reliability_score": float(result.get("reliability_score", 0.8)),
            "reliability_notes": result.get("reliability_notes", ""),
            "scam_risk_score": float(result.get("scam_risk_score", 0.0)),
            "warnings": result.get("warnings", [])
        }

        conf = min(max(conf, 0.0), 1.0)
        
        # Post-process edge cases to adjust classification
        cls, conf, exp, edge_cases, metadata = self._apply_edge_case_logic(
            cls, conf, exp, edge_cases, metadata
        )

        return cls, round(conf, 2), exp, edge_cases, metadata

    def _apply_edge_case_logic(self, cls, conf, exp, edge_cases, metadata):
        """Apply additional logic based on detected edge cases."""
        
        warnings = metadata.get("warnings", [])
        reliability = metadata.get("reliability_score", 0.8)
        
        # Handle non-human audio
        if metadata.get("audio_content_type") in ["ANIMAL", "VEHICLE", "ENVIRONMENTAL"]:
            if cls not in ["NON_HUMAN_AUDIO"]:
                cls = "NON_HUMAN_AUDIO"
                # Determine if the non-human audio is real or AI
                if any(ec in edge_cases for ec in ["ANIMAL_SOUND_AI", "VEHICLE_SOUND_AI"]):
                    metadata["non_human_source"] = "AI_GENERATED"
                else:
                    metadata["non_human_source"] = "REAL"
        
        # Reduce confidence for edge cases that affect reliability
        reliability_reducers = [
            "DISTORTED_AUDIO",
            "SILENCE_DOMINANT", 
            "VERY_SHORT_AUDIO",
            "ENVIRONMENTAL_NOISE"
        ]
        
        for reducer in reliability_reducers:
            if reducer in edge_cases:
                conf = max(conf - 0.15, 0.3)
                reliability = max(reliability - 0.2, 0.2)
        
        # Handle human-sounding-robotic edge case
        if "ROBOTIC_HUMAN" in edge_cases and cls == "AI_GENERATED":
            # Check for other human indicators
            human_indicators = ["SPEECH_IMPEDIMENT", "EMOTIONAL_EXTREME", 
                               "ACCENT_HEAVY", "ELDERLY_VOICE"]
            if any(ind in edge_cases for ind in human_indicators):
                warnings.append("Classified as AI but human indicators present - manual review recommended")
                conf = max(conf - 0.2, 0.4)
        
        # Handle deepfake/cloned voice with high scam risk
        if "DEEPFAKE_SUSPECTED" in edge_cases or "CLONED_VOICE" in edge_cases:
            if metadata.get("scam_risk_score", 0) > 0.6:
                warnings.append("CRITICAL: Potential deepfake with high scam indicators")
                metadata["fraud_alert"] = True
        
        # Handle scam indicators
        if "SCAM_INDICATORS" in edge_cases:
            scam_score = metadata.get("scam_risk_score", 0)
            if scam_score > 0.7:
                warnings.append("HIGH SCAM RISK: Audio contains multiple fraud indicators")
            elif scam_score > 0.4:
                warnings.append("MODERATE SCAM RISK: Some suspicious patterns detected")
        
        # Handle mixed language - should not affect classification
        if "MIXED_LANGUAGE" in edge_cases:
            metadata["languages_detected"] = "Multiple (code-switching detected)"
            # Don't reduce confidence for this
        
        # Handle synthetic with noise (AI trying to sound human)
        if "SYNTHETIC_WITH_NOISE" in edge_cases:
            if cls == "HUMAN":
                warnings.append("Noise patterns seem artificial - possible AI with added imperfections")
                conf = max(conf - 0.1, 0.5)
        
        # Handle repeated patterns
        if "REPEATED_PATTERNS" in edge_cases:
            if cls == "HUMAN":
                warnings.append("Unnatural repetition patterns detected")
                conf = max(conf - 0.15, 0.4)
        
        # Update metadata
        metadata["reliability_score"] = round(reliability, 2)
        metadata["warnings"] = warnings
        
        return cls, conf, exp, edge_cases, metadata

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

    # Edge case: Check audio size
    audio_size_kb = len(audio_bytes) / 1024
    size_warning = None
    
    if audio_size_kb < 10:
        size_warning = "Audio file very small - may affect analysis reliability"
    elif audio_size_kb > 50000:  # 50MB
        size_warning = "Audio file very large - processing may be slower"

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        path = f.name

    try:
        detector = VoiceDetector()

        cls, conf, exp, edge_cases, metadata = detector.analyze(
            path,
            data["language"]
        )
        
        # Add size warning if applicable
        if size_warning:
            metadata["warnings"] = metadata.get("warnings", []) + [size_warning]

        # Build comprehensive response
        response = {
            "status": "success",
            "language": data["language"],
            "classification": cls,
            "confidenceScore": conf,
            "explanation": exp,
            "edgeCasesDetected": edge_cases,
            "edgeCaseDescriptions": {
                ec: EDGE_CASE_DEFINITIONS.get(ec, "Unknown edge case")
                for ec in edge_cases
            },
            "metadata": {
                "audioContentType": metadata.get("audio_content_type", "SPEECH"),
                "reliabilityScore": metadata.get("reliability_score", 0.8),
                "reliabilityNotes": metadata.get("reliability_notes", ""),
                "scamRiskScore": metadata.get("scam_risk_score", 0.0),
                "warnings": metadata.get("warnings", []),
                "edgeCaseDetails": metadata.get("edge_case_details", {}),
                "fraudAlert": metadata.get("fraud_alert", False),
                "languagesDetected": metadata.get("languages_detected", data["language"])
            }
        }
        
        # Add non-human source classification if applicable
        if "non_human_source" in metadata:
            response["metadata"]["nonHumanSource"] = metadata["non_human_source"]

        return jsonify(response)

    finally:
        if os.path.exists(path):
            os.remove(path)

# ---------------- EDGE CASE INFO ENDPOINT ---------------- #

@app.route("/api/edge-cases", methods=["GET"])
@require_api_key
def get_edge_cases():
    """Return all edge cases the system can detect."""
    
    return jsonify({
        "status": "success",
        "edgeCases": EDGE_CASE_DEFINITIONS,
        "categories": {
            "language": ["MIXED_LANGUAGE", "ACCENT_HEAVY"],
            "speaker_characteristics": [
                "ROBOTIC_HUMAN", "CHILD_VOICE", "ELDERLY_VOICE",
                "SPEECH_IMPEDIMENT", "MULTIPLE_SPEAKERS"
            ],
            "ai_detection": [
                "CLONED_VOICE", "HIGH_QUALITY_TTS", "SYNTHETIC_WITH_NOISE",
                "REPEATED_PATTERNS", "DEEPFAKE_SUSPECTED"
            ],
            "non_speech_audio": [
                "ANIMAL_SOUND_REAL", "ANIMAL_SOUND_AI",
                "VEHICLE_SOUND_REAL", "VEHICLE_SOUND_AI",
                "MUSIC_WITH_VOCALS", "NON_SPEECH_VOCAL"
            ],
            "audio_quality": [
                "ENVIRONMENTAL_NOISE", "SILENCE_DOMINANT",
                "VERY_SHORT_AUDIO", "DISTORTED_AUDIO"
            ],
            "speech_style": [
                "WHISPERED_SPEECH", "EMOTIONAL_EXTREME"
            ],
            "security": [
                "SCAM_INDICATORS"
            ]
        }
    })

# ---------------- HEALTH ---------------- #

@app.route("/health")
def health():

    return jsonify({
        "status": "healthy",
        "service": "voice-detection-api",
        "features": {
            "edge_case_detection": True,
            "scam_risk_scoring": True,
            "non_human_audio_detection": True,
            "mixed_language_support": True
        }
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