# Aether - Plant Disease Detection

Aether is a Flask web app for detecting apple leaf diseases from an uploaded image. It uses an EfficientNet-B3 classifier built with PyTorch and timm, then generates practical explanation and treatment guidance with Groq's LLM API.

The repository is structured as the runnable project root. You do not need the outer `PlantDisease` folder or any zip archive.

## Features

- Drag-and-drop or click-to-upload image workflow
- Image validation for PNG, JPG, JPEG, and WEBP files
- EfficientNet-B3 multi-label inference across 6 apple leaf classes: `complex`, `frog_eye_leaf_spot`, `healthy`, `powdery_mildew`, `rust`, `scab`
- Groq-powered diagnosis explanation and treatment guidance
- Confidence scores for every class
- Browser-side diagnosis history
- JSON prediction API
- Runtime upload storage with randomized filenames

## Setup

```bash
git clone https://github.com/Shaheen91/Aether.git
cd Aether
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Create .env:
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
SECRET_KEY=change-this-secret
FLASK_DEBUG=False
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
Run:
python app.py
Open:
http://localhost:5000
Project Structure
.
|-- app.py
|-- config.py
|-- predictor.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- models/
|   `-- phase2_final.pth
|-- Nootbook/
|   `-- plant_disease_with_llm.ipynb
|-- static/
|   |-- css/style.css
|   |-- js/script.js
|   `-- images/
|-- templates/
|   `-- index.html
|-- uploads/
|   `-- .gitkeep
`-- utils/
    |-- __init__.py
    `-- llm.py
API
POST /api/predict
Send a multipart form request with one image field named image.
curl -X POST http://localhost:5000/api/predict -F "image=@leaf.jpg"
Success response:
{
  "success": true,
  "is_healthy": false,
  "detected_diseases": ["rust"],
  "all_predictions": [
    {"label": "rust", "confidence": 0.87},
    {"label": "healthy", "confidence": 0.05}
  ],
  "explanation": "**What's going on**\n...",
  "llm_error": null,
  "image_url": "/uploads/generated-file-name.jpg"
}
Model Details
Backbone: efficientnet_b3
Input size: 224 x 224
Classes: complex, frog_eye_leaf_spot, healthy, powdery_mildew, rust, scab
Activation: sigmoid
Detection threshold: 0.5
Model weights path: models/phase2_final.pth
The model is multi-label, so more than one disease can be detected from one leaf image.
Environment Variables
Variable	Default	Description
GROQ_API_KEY	None	Required for AI explanations
GROQ_MODEL	llama-3.3-70b-versatile	Groq model used
SECRET_KEY	Development fallback	Flask session secret
FLASK_DEBUG	False	Flask debug mode
FLASK_HOST	0.0.0.0	Flask host
FLASK_PORT	5000	Flask port

Notes
Do not commit .env or API keys.
Runtime uploads are ignored except for uploads/.gitkeep.
Python cache folders should not be committed.
The app provides diagnostic support only and is not a replacement for professional agronomic advice.
