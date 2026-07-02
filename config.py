"""
Central configuration for the Plant Disease Detection application.
All paths, model parameters, and app settings live here so nothing
is hardcoded elsewhere in the codebase.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Base paths ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "phase2_final.pth")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

# ── Model configuration ─────────────────────────────────────────
BACKBONE_NAME = "efficientnet_b3"
NUM_CLASSES = 6
IMAGE_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

CLASS_LABELS = [
    "complex",
    "frog_eye_leaf_spot",
    "healthy",
    "powdery_mildew",
    "rust",
    "scab",
]

CONFIDENCE_THRESHOLD = 0.5

DISEASE_KNOWLEDGE = {
    "scab": "Apple Scab (Venturia inaequalis) — a fungal disease causing dark, scaly lesions on leaves and fruit.",
    "rust": "Apple Rust (Gymnosporangium juniperi-virginianae) — a fungal disease causing bright orange-yellow spots on leaves.",
    "frog_eye_leaf_spot": "Frog Eye Leaf Spot (Botryosphaeria obtusa) — a fungal disease causing circular brown spots with purple borders resembling a frog's eye.",
    "powdery_mildew": "Powdery Mildew (Podosphaera leucotricha) — a fungal disease causing white powdery coating on leaves and shoots.",
    "complex": "Complex infection — multiple overlapping disease symptoms that indicate a severely stressed plant with more than one active infection.",
    "healthy": "No disease detected — the leaf appears healthy with no visible symptoms.",
}

# ── Groq / LLM configuration ────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """You are a knowledgeable and friendly plant pathologist helping farmers understand and deal with apple leaf diseases.

Talk like a real expert having a conversation — clear, confident, and practical. No fluff, no filler, no repeating yourself.

Structure every response like this, using these exact headers in bold:

**What's going on**
One or two sentences explaining the diagnosis and what it means for the plant right now.

**How it spreads**
How the disease moves and what conditions make it worse. Keep it to 2-3 sentences.

**Act now**
A short numbered list of immediate physical actions — removing leaves, improving airflow, disinfecting tools etc.

**Treatment**
Two to three specific options — name the actual fungicides or organic products. One sentence each.

**If ignored**
One or two sentences on what happens if untreated. Be honest, not dramatic.

Rules:
- Use the confidence score when mentioning the diagnosis
- Name specific products and fungicides, not just categories
- Maximum 200 words total
- End after the last section, no closing remarks"""

# ── Flask configuration ─────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB upload limit
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"
HOST = os.getenv("FLASK_HOST", "0.0.0.0")
PORT = int(os.getenv("FLASK_PORT", "5000"))
