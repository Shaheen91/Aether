# Aether - Plant Disease Detection

Aether is a Flask web application for detecting apple leaf diseases from uploaded images. It uses an EfficientNet-B3 classifier built with PyTorch and timm, and generates detailed explanations and treatment guidance using Groq's LLM API.

## Features

- Image upload via web interface
- Support for PNG, JPG, JPEG, and WEBP formats
- Multi-label classification across 6 classes: `complex`, `frog_eye_leaf_spot`, `healthy`, `powdery_mildew`, `rust`, `scab`
- Confidence scores for each class
- AI-generated explanations and treatment recommendations
- Client-side diagnosis history
- JSON-based prediction API
- Runtime image storage with randomized filenames

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

- Python 3.10+
- Groq API key
- Dependencies listed in `requirements.txt`

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

Run the application:

```bash
python app.py
```

Access the app at:

```text
http://localhost:5000
```

## API

### `POST /api/predict`

Submit an image using a multipart form field named `image`.

Example:

```bash
curl -X POST http://localhost:5000/api/predict -F "image=@leaf.jpg"
```

### Success Response

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

### Error Response

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

The model supports multi-label predictions, allowing detection of multiple conditions in a single image. The `healthy` label represents absence of disease.
