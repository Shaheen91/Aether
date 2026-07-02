Copy-paste this exact README:

```markdown
# Aether - Plant Disease Detection

Aether is a Flask web app for detecting apple leaf diseases from an uploaded image. It uses an EfficientNet-B3 classifier built with PyTorch and timm, then generates practical explanation and treatment guidance with Groq's LLM API.

## Features

- Upload apple leaf images through a clean web interface
- Validate PNG, JPG, JPEG, and WEBP files
- Detect 6 apple leaf classes: `complex`, `frog_eye_leaf_spot`, `healthy`, `powdery_mildew`, `rust`, `scab`
- Show confidence scores for every class
- Generate AI explanations and treatment guidance with Groq
- Keep a browser-side diagnosis history
- Provide a JSON prediction API
- Store runtime uploads with randomized filenames

## Project Structure

```text
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
|   |-- css/
|   |   `-- style.css
|   |-- js/
|   |   `-- script.js
|   `-- images/
|-- templates/
|   `-- index.html
|-- uploads/
|   `-- .gitkeep
`-- utils/
    |-- __init__.py
    `-- llm.py
```

## Requirements

- Python 3.10 or newer
- Groq API key
- Dependencies from `requirements.txt`

## Setup

Clone the repository:

```bash
git clone https://github.com/Shaheen91/Aether.git
cd Aether
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
SECRET_KEY=change-this-secret
FLASK_DEBUG=False
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
```

Run the app:

```bash
python app.py
```

Open in your browser:

```text
http://localhost:5000
```

## API

### `POST /api/predict`

Send an image using a multipart form field named `image`.

Example:

```bash
curl -X POST http://localhost:5000/api/predict -F "image=@leaf.jpg"
```

Success response:

```json
{
  "success": true,
  "is_healthy": false,
  "detected_diseases": ["rust"],
  "all_predictions": [
    {
      "label": "rust",
      "confidence": 0.87
    },
    {
      "label": "healthy",
      "confidence": 0.05
    }
  ],
  "explanation": "**What's going on**\n...",
  "llm_error": null,
  "image_url": "/uploads/generated-file-name.jpg"
}
```

Error response:

```json
{
  "success": false,
  "error": "Human-readable error message."
}
```

## Model Details

- Backbone: `efficientnet_b3`
- Frameworks: PyTorch, torchvision, timm
- Input size: `224 x 224`
- Classes: `complex`, `frog_eye_leaf_spot`, `healthy`, `powdery_mildew`, `rust`, `scab`
- Activation: sigmoid
- Detection threshold: `0.5`
- Model weights path: `models/phase2_final.pth`

The model is multi-label, so more than one disease can be detected from one leaf image. The `healthy` label is not treated as a disease detection.

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `GROQ_API_KEY` | None | Required for AI explanations |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model used for explanations |
| `SECRET_KEY` | Development fallback | Flask session secret |
| `FLASK_DEBUG` | `False` | Enables Flask debug mode |
| `FLASK_HOST` | `0.0.0.0` | Flask host |
| `FLASK_PORT` | `5000` | Flask port |

## Notes

- Do not commit `.env` or API keys.
- Runtime uploads are ignored except for `uploads/.gitkeep`.
- Python cache folders should not be committed.
- The outer `PlantDisease` folder and zip archive are not part of this repository.
- This app provides diagnostic support only and is not a replacement for professional agronomic advice.
```
