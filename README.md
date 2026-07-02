# VerdantAI — Plant Disease Detection

A production-ready web app that detects apple leaf diseases from a photo using an
EfficientNet-B3 (PyTorch/timm) classifier, then explains the diagnosis and treatment
plan using the Groq LLM (Llama 3.3 70B).

## Features

- Drag-and-drop / click-to-upload image intake with client-side validation
- EfficientNet-B3 multi-label inference across 6 classes: `complex`,
  `frog_eye_leaf_spot`, `healthy`, `powdery_mildew`, `rust`, `scab`
- Groq LLM–generated explanation covering diagnosis, spread, immediate actions,
  treatment, and consequences of inaction
- Premium dark-themed, glassmorphism SaaS UI built with Bootstrap 5
- Confidence bars per class, animated confidence ring, session diagnosis history
- Robust error handling for bad uploads, missing model weights, and LLM failures

## Project structure

```
PlantDiseaseProject/
├── app.py                 # Flask app: routes, upload handling, error handling
├── predictor.py            # PyTorch model, transforms, inference logic
├── config.py               # Central configuration (paths, model + app settings)
├── requirements.txt
├── README.md
├── .env.example
├── models/
│   └── phase2_final.pth    # Trained model weights (you provide this file)
├── utils/
│   └── llm.py               # Groq LLM integration
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   └── images/
└── uploads/                 # Uploaded images are saved here at runtime
```

## Prerequisites

- Python 3.10+
- A Groq API key (free tier available at https://console.groq.com/keys)
- The trained model file `phase2_final.pth`, placed at `models/phase2_final.pth`
  (this file is **not** included in the repo — it must be added by you)

## Setup

1. **Clone / unzip the project**, then move into it:
   ```bash
   cd PlantDiseaseProject
   ```

2. **Add the trained model weights.**
   Place your trained checkpoint at exactly this path:
   ```
   models/phase2_final.pth
   ```

3. **Create your environment file:**
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and set `GROQ_API_KEY` to your real Groq API key.

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   > If you don't have a CUDA GPU, PyTorch will automatically run on CPU —
   > no changes required, inference will just be a bit slower.

5. **Run the app:**
   ```bash
   python app.py
   ```

6. Open your browser at **http://localhost:5000**

## Configuration reference (`.env`)

| Variable       | Description                                   | Default                        |
|----------------|------------------------------------------------|---------------------------------|
| `GROQ_API_KEY` | Your Groq API key (required for explanations)  | —                                |
| `GROQ_MODEL`   | Groq model used for explanations               | `llama-3.3-70b-versatile`       |
| `SECRET_KEY`   | Flask session secret                           | dev key — change in production  |
| `FLASK_DEBUG`  | Enable Flask debug mode                        | `False`                         |
| `FLASK_HOST`   | Host to bind to                                | `0.0.0.0`                       |
| `FLASK_PORT`   | Port to bind to                                | `5000`                          |

## API

### `POST /api/predict`

Multipart form upload with a single `image` field.

**Success response:**
```json
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
  "image_url": "/uploads/<generated-name>.jpg"
}
```

**Error response:**
```json
{ "success": false, "error": "Human-readable error message." }
```

## Notes on the model

- Input images are resized to 224×224, normalized with ImageNet mean/std
  (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`).
- The model outputs raw logits for 6 classes; a sigmoid is applied (multi-label,
  not softmax) and any class scoring above `0.5` is treated as "detected" — this
  matches the original notebook's inference behavior.
- The classifier head is `Dropout(0.3) → Linear(→512) → ReLU → Dropout(0.2) → Linear(→6)`
  on top of a `timm` EfficientNet-B3 backbone.

## Production notes

- Set a real `SECRET_KEY` and disable `FLASK_DEBUG` in production.
- Run behind a WSGI server, e.g.:
  ```bash
  gunicorn -w 2 -b 0.0.0.0:5000 app:app
  ```
- Uploaded images are stored under `uploads/` with randomized filenames; add a
  periodic cleanup job or object-storage backend for long-running deployments.
- This tool provides diagnostic support, not professional agronomic advice —
  the UI includes this disclaimer in the footer.
