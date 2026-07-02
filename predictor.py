"""
predictor.py

Everything related to the PyTorch model lives here:
  - model architecture (EfficientNet-B3 backbone + custom classifier head)
  - image preprocessing transforms
  - weight loading
  - inference / prediction logic
  - confidence calculation and class mapping
"""

import logging

import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

import config

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PlantDiseaseModel(nn.Module):
    """EfficientNet-B3 backbone with a custom multi-label classifier head."""

    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self.backbone = timm.create_model(
            config.BACKBONE_NAME, pretrained=False, num_classes=0
        )
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)


# ── Preprocessing transforms (must match training-time transforms) ──
inference_transform = transforms.Compose(
    [
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
    ]
)


class Predictor:
    """Wraps the model and exposes a simple predict() interface."""

    def __init__(self, model_path: str = config.MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        logger.info("Loading model from %s on device %s", self.model_path, DEVICE)
        model = PlantDiseaseModel().to(DEVICE)
        state_dict = torch.load(self.model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model
        logger.info("Model loaded successfully.")

    def predict(self, image: Image.Image) -> dict:
        """
        Run inference on a PIL image.

        Returns a dict with:
          - all_predictions: list of {label, confidence} sorted descending
          - detected: list of {label, confidence} above the confidence threshold
          - is_healthy: bool, True if no disease crossed the threshold
        """
        image = image.convert("RGB")
        tensor = inference_transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        all_predictions = [
            {"label": config.CLASS_LABELS[i], "confidence": float(probs[i])}
            for i in range(len(config.CLASS_LABELS))
        ]
        all_predictions.sort(key=lambda p: p["confidence"], reverse=True)

        detected = [
            p for p in all_predictions if p["confidence"] > config.CONFIDENCE_THRESHOLD
        ]
        # "healthy" alone shouldn't count as a disease detection
        detected_diseases = [d for d in detected if d["label"] != "healthy"]

        return {
            "all_predictions": all_predictions,
            "detected": detected_diseases,
            "is_healthy": len(detected_diseases) == 0,
        }


# ── Lazy singleton so the model loads once per process ──────────
_predictor_instance: Predictor | None = None


def get_predictor() -> Predictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = Predictor()
    return _predictor_instance
