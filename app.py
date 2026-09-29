"""
Flask Web Application and REST API for Marathi News Classification
Provides web interface and RESTful endpoints for Marathi headline category prediction.
"""

import sys
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.predict import MarathiNewsPredictor

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Ensure JSON outputs raw UTF-8 Devanagari

# Initialize predictor singleton
try:
    predictor = MarathiNewsPredictor()
    print("MarathiNewsPredictor successfully initialized.")
except Exception as e:
    print(f"Warning: Could not initialize predictor on startup: {e}")
    predictor = None

def load_metrics():
    """Loads metrics.json from results directory."""
    metrics_path = PROJECT_ROOT / "results" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# ----------------- WEB ROUTES ----------------- #

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    input_text = ""
    error_message = None
    
    if request.method == "POST":
        input_text = request.form.get("headline", "").strip()
        if not input_text:
            error_message = "Please enter a Marathi news headline. (कृपया मराठी बातमीचे शीर्षक प्रविष्ट करा.)"
        elif predictor is None:
            error_message = "Model is not loaded. Please ensure training has been executed."
        else:
            result = predictor.predict(input_text)
            if result.get("success"):
                prediction_result = result
            else:
                error_message = result.get("error", "Prediction failed.")
                
    return render_template(
        "index.html",
        prediction=prediction_result,
        input_text=input_text,
        error_message=error_message
    )

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/model")
def model_info():
    metrics = load_metrics()
    return render_template("model.html", metrics=metrics)

@app.route("/results")
def results():
    metrics = load_metrics()
    return render_template("results.html", metrics=metrics)

# ----------------- REST API ROUTES ----------------- #

@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": predictor is not None
    }), 200

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    REST API endpoint for predicting Marathi news headline category.
    Accepts JSON: { "headline": "हृतिक करतोय बायोपिक 'सुपर 30'" }
    Returns JSON: { "headline": "...", "category": "Entertainment", "confidence": 0.55 }
    """
    if not request.is_json:
        return jsonify({
            "error": "Request body must be valid JSON with 'Content-Type: application/json'"
        }), 400
        
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({
            "error": "Malformed JSON payload in request body"
        }), 400
        
    if "headline" not in data:
        return jsonify({
            "error": "Missing required field 'headline' in JSON payload"
        }), 400
        
    headline = str(data.get("headline", "")).strip()
    if not headline:
        return jsonify({
            "error": "Please enter a Marathi news headline. (Field 'headline' cannot be empty)"
        }), 400
        
    if predictor is None:
        return jsonify({
            "error": "Model service unavailable. Predictor not initialized."
        }), 503
        
    result = predictor.predict(headline)
    if not result.get("success"):
        return jsonify({
            "error": result.get("error", "Validation error occurred"),
            "headline": headline
        }), 400
        
    return jsonify({
        "headline": result["headline"],
        "original_text": result["original_text"],
        "normalized_text": result["normalized_text"],
        "cleaned_text": result["cleaned_text"],
        "tokens": result["tokens"],
        "filtered_tokens": result["filtered_tokens"],
        "tfidf_features": result["tfidf_features"],
        "category": result["category"],
        "predicted_category": result["predicted_category"],
        "confidence": result["confidence"],
        "confidence_percent": result["confidence_percent"],
        "confidence_type": result["confidence_type"],
        "probabilities": result["probabilities"]
    }), 200

# Error Handlers
@app.errorhandler(404)
def not_found(e):
    return render_template("index.html", error_message="Requested page not found (404)."), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error occurred."}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
