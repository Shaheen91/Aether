"""
app.py

Flask application entry point. Handles image upload, runs the
PyTorch prediction pipeline, calls the Groq LLM for an explanation,
and returns everything to the frontend as JSON.
"""

import logging
import os
import uuid

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

import config
from predictor import get_predictor
from utils.llm import get_disease_explanation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER

os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Unsupported file type. Use PNG, JPG, JPEG, or WEBP.",
                }
            ),
            400,
        )

    saved_path = None
    try:
        # ── Save the uploaded image with a unique, safe filename ──
        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        safe_name = secure_filename(unique_name)
        saved_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
        file.save(saved_path)

        # ── Open and validate the image ──
        try:
            image = Image.open(saved_path)
            image.verify()
            image = Image.open(saved_path)  # re-open after verify()
        except UnidentifiedImageError:
            return (
                jsonify({"success": False, "error": "The uploaded file is not a valid image."}),
                400,
            )

        # ── Run CNN inference ──
        predictor = get_predictor()
        result = predictor.predict(image)

        detected_labels = [d["label"] for d in result["detected"]]
        detected_scores = [d["confidence"] for d in result["detected"]]

        # ── Run LLM explanation ──
        try:
            explanation = get_disease_explanation(detected_labels, detected_scores)
            llm_error = None
        except RuntimeError as llm_exc:
            logger.warning("LLM explanation unavailable: %s", llm_exc)
            explanation = None
            llm_error = str(llm_exc)

        response_payload = {
            "success": True,
            "is_healthy": result["is_healthy"],
            "detected_diseases": detected_labels,
            "all_predictions": result["all_predictions"],
            "explanation": explanation,
            "llm_error": llm_error,
            "image_url": f"/uploads/{safe_name}",
        }
        return jsonify(response_payload)

    except FileNotFoundError:
        logger.exception("Model weights not found")
        return (
            jsonify(
                {
                    "success": False,
                    "error": (
                        "Model weights not found at models/phase2_final.pth. "
                        "Please place the trained model file there."
                    ),
                }
            ),
            500,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        return jsonify({"success": False, "error": f"Prediction failed: {exc}"}), 500


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    from flask import send_from_directory

    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"success": False, "error": "File is too large. Max size is 10MB."}), 413


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"success": False, "error": "Resource not found."}), 404


@app.errorhandler(500)
def server_error(_error):
    return jsonify({"success": False, "error": "Internal server error."}), 500


if __name__ == "__main__":
    # Warm up the model at startup so the first request isn't slow.
    logger.info("Warming up model...")
    try:
        get_predictor()
    except Exception:
        logger.exception(
            "Could not load model at startup. Ensure models/phase2_final.pth exists."
        )
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
