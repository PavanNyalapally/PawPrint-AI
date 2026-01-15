# backend/app.py

import sys
import os
import json
import numpy as np
import logging
import logging.handlers
from datetime import datetime
from functools import wraps

# =====================================================
# PATH FIX (allow imports from project root)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image
import io

from backend.utils.preprocess import preprocess_image
from backend.utils.age_features import extract_footprint_area
from backend.utils.age_estimation import estimate_age_group
from backend.utils.severity import estimate_severity
from backend.utils.visualize import generate_debug_overlay
from backend.database import init_db, save_scan, get_recent_scans

# =====================================================
# LOGGING SETUP
# =====================================================
def setup_logging(config: dict) -> logging.Logger:
    """Configure application logging."""
    logger = logging.getLogger("PawPrintAI")
    logger.setLevel(config['logging']['level'])
    
    # Create logs directory if it doesn't exist
    log_file = config['logging']['log_file']
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(config['logging']['format'])
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# =====================================================
# LOAD CONFIG
# =====================================================
def load_config(config_path: str = "app_config.json") -> dict:
    """Load configuration from JSON file."""
    try:
        config_full_path = os.path.join(BASE_DIR, config_path)
        with open(config_full_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"❌ Config file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config file: {e}")
        raise

try:
    config = load_config()
except Exception as e:
    print(f"❌ Failed to load config: {e}")
    sys.exit(1)

logger = setup_logging(config)
logger.info("="*60)
logger.info("🚀 PawPrintAI Backend Starting Up")
logger.info("="*60)

# =====================================================
# APP SETUP
# =====================================================
app = Flask(__name__, static_folder=None)

# Initialize Database
try:
    init_db()
    logger.info("✅ Database initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize database: {e}")

# CORS Configuration
if config['security']['enable_cors']:
    CORS(app)
    logger.info("✅ CORS enabled")

# Security Headers Middleware
@app.after_request
def set_security_headers(response):
    """Add security headers to all responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data:; script-src 'self' 'unsafe-inline' 'unsafe-eval'"
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response

# Request size limit
app.config['MAX_CONTENT_LENGTH'] = config['security']['max_content_length_mb'] * 1024 * 1024
logger.info(f"✅ Max request size: {config['security']['max_content_length_mb']} MB")

# =====================================================
# CONSTANTS
# =====================================================
MODEL_DIR = os.path.join(BASE_DIR, "model")
UPLOAD_FOLDER = os.path.join(BASE_DIR, config['upload']['upload_folder'])
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

KERAS_MODEL_PATH = os.path.join(BASE_DIR, config['model']['keras_model_path'])
H5_MODEL_PATH = os.path.join(BASE_DIR, config['model']['h5_model_path'])
CLASS_INDEX_PATH = os.path.join(BASE_DIR, config['model']['class_indices_path'])

ALLOWED_EXTENSIONS = set(config['upload']['allowed_extensions'])
MAX_FILE_SIZE = config['upload']['max_file_size_mb'] * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logger.info(f"✅ Upload folder ready: {UPLOAD_FOLDER}")

# =====================================================
# MODEL LOADING (ONCE AT STARTUP)
# =====================================================
model = None
IDX_TO_CLASS = None

try:
    if os.path.exists(KERAS_MODEL_PATH):
        model = load_model(KERAS_MODEL_PATH)
        logger.info(f"✅ Loaded Keras model: {KERAS_MODEL_PATH}")
    elif os.path.exists(H5_MODEL_PATH):
        model = load_model(H5_MODEL_PATH)
        logger.info(f"✅ Loaded H5 model: {H5_MODEL_PATH}")
    else:
        logger.error("❌ No trained model found (footprint_cnn_final.keras or footprint_cnn.h5)")
        # In production, we might want to exit or raise
        # For dev/demo, we can continue but /predict will fail
    
    if model:
        logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Model loading failed: {e}")
    sys.exit(1)

# =====================================================
# CLASS LABEL MAPPING (CRITICAL)
# =====================================================
try:
    if os.path.exists(CLASS_INDEX_PATH):
        with open(CLASS_INDEX_PATH, "r") as f:
            class_indices = json.load(f)
        IDX_TO_CLASS = {int(v): k for k, v in class_indices.items()}
        logger.info(f"✅ Class indices loaded: {IDX_TO_CLASS}")
    else:
        logger.warning(f"⚠️ class_indices.json not found at {CLASS_INDEX_PATH}")
except Exception as e:
    logger.error(f"❌ Class indices loading failed: {e}")
    sys.exit(1)

# =====================================================
# HEALTH CHECK ENDPOINT
# =====================================================
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for monitoring."""
    try:
        model_ready = model is not None and IDX_TO_CLASS is not None
        if not model_ready:
            logger.warning("⚠️ Health check: Model not loaded")
        
        return jsonify({
            "status": "healthy" if model_ready else "degraded",
            "message": "API is operational" if model_ready else "Model not loaded",
            "model_loaded": model_ready,
            "classes": len(IDX_TO_CLASS) if IDX_TO_CLASS else 0,
            "timestamp": datetime.now().isoformat(),
            "version": config['api']['version']
        }), 200
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

# =====================================================
# FRONTEND SERVING
# =====================================================
@app.route("/", defaults={"path": "index.html"})
@app.route("/<path:path>")
def serve_frontend(path):
    return send_from_directory(FRONTEND_DIR, path)

# =====================================================
# HELPERS
# =====================================================
def allowed_file(filename: str) -> bool:
    """Check if file has allowed extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

def validate_image_file(file_path: str) -> bool:
    """Validate that the file is actually an image."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception as e:
        logger.warning(f"⚠️ Image validation failed: {e}")
        return False

def validate_request(file) -> tuple[bool, str]:
    """Validate file upload request."""
    if file.filename == "":
        return False, "Empty filename"
    
    if not allowed_file(file.filename):
        return False, f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        return False, f"File too large. Max: {config['upload']['max_file_size_mb']} MB"
    
    return True, "OK"

# =====================================================
# PREDICTION ROUTE
# =====================================================
@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict animal from paw print image.
    
    Returns:
        JSON: {
            "animal": str,
            "confidence": float,
            "wildness_level": str,
            "human_risk_level": str,
            "estimated_age_group": str,
            "debug_image": str (base64)
        }
    """
    file_path = None
    request_id = datetime.now().isoformat()

    try:
        logger.info(f"[{request_id}] 📥 Prediction request received")
        
        # ✅ Check if model is loaded
        if model is None or IDX_TO_CLASS is None:
            logger.error(f"[{request_id}] ❌ Model not loaded")
            return jsonify({"error": "Model not loaded. Server not ready"}), 503
        
        # ✅ Request validation
        if "file" not in request.files:
            logger.warning(f"[{request_id}] ❌ No file in request")
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        
        # Validate request
        is_valid, message = validate_request(file)
        if not is_valid:
            logger.warning(f"[{request_id}] ❌ Validation failed: {message}")
            return jsonify({"error": message}), 400

        # ✅ Save file
        if not file.filename:
            logger.warning(f"[{request_id}] ❌ Invalid filename")
            return jsonify({"error": "Invalid filename"}), 400
        
        filename = secure_filename(file.filename)
        if not filename:
            logger.warning(f"[{request_id}] ❌ Invalid filename")
            return jsonify({"error": "Invalid filename"}), 400
        
        file_path = os.path.join(UPLOAD_FOLDER, f"{request_id.replace(':', '-')}_{filename}")
        file.save(file_path)
        logger.debug(f"[{request_id}] ✅ File saved: {file_path}")

        # ✅ Validate image
        if not validate_image_file(file_path):
            logger.error(f"[{request_id}] ❌ Invalid image file")
            return jsonify({"error": "Invalid image file"}), 400

        # ✅ Image preprocessing
        try:
            image = preprocess_image(file_path)
            if image is None:
                logger.error(f"[{request_id}] ❌ Image preprocessing failed")
                return jsonify({"error": "Failed to preprocess image"}), 400
            logger.debug(f"[{request_id}] ✅ Image preprocessed")
        except Exception as e:
            logger.error(f"[{request_id}] ❌ Preprocessing error: {e}")
            return jsonify({"error": "Image preprocessing error"}), 500

        # ✅ Model prediction
        try:
            preds = model.predict(image, verbose=0)
            idx = int(np.argmax(preds, axis=1)[0])
            confidence = round(float(np.max(preds)) * 100, 2)
            animal_name = IDX_TO_CLASS.get(idx, "Unknown")
            logger.info(f"[{request_id}] ✅ Prediction: {animal_name} ({confidence}%)")
        except Exception as e:
            logger.error(f"[{request_id}] ❌ Model prediction error: {e}")
            return jsonify({"error": "Model prediction failed"}), 500

        # ✅ Post-processing (Age, Wildness, Vis)
        try:
            severity_info = estimate_severity(animal_name)
            footprint_area = extract_footprint_area(file_path)
            age_group = estimate_age_group(animal_name, footprint_area)
            debug_image = generate_debug_overlay(file_path)
            
            if not severity_info or not age_group:
                logger.warning(f"[{request_id}] ⚠️ Post-processing returned None")
                severity_info = severity_info or {"wildness": "unknown", "human_risk": "unknown"}
                age_group = age_group or "unknown"
            
            logger.debug(f"[{request_id}] ✅ Post-processing complete")
        except Exception as e:
            logger.error(f"[{request_id}] ❌ Post-processing error: {e}")
            # Return prediction even if post-processing fails
            severity_info = {"wildness": "error", "human_risk": "error"}
            age_group = "unknown"
            debug_image = ""

        # ✅ Response Data
        response_data = {
            "animal": animal_name,
            "confidence": confidence,
            "wildness_level": severity_info.get("wildness", "unknown"),
            "human_risk_level": severity_info.get("human_risk", "unknown"),
            "estimated_age_group": age_group,
            "debug_image": debug_image,
            "request_id": request_id
        }

        # ✅ Save to History
        try:
            save_scan(response_data)
            logger.debug(f"[{request_id}] ✅ Scan saved to history")
        except Exception as e:
            logger.error(f"[{request_id}] ❌ Failed to save history: {e}")
        
        logger.info(f"[{request_id}] ✅ Prediction complete")
        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"[{request_id}] ❌ Unexpected error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error", "request_id": request_id}), 500

    finally:
        # Cleanup uploaded file
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"[{request_id}] 🗑️ Temp file cleaned up")
            except Exception as e:
                logger.warning(f"[{request_id}] ⚠️ Failed to cleanup temp file: {e}")

# =====================================================
# HISTORY ROUTE
# =====================================================
@app.route("/history", methods=["GET"])
def history():
    """Get recent scans."""
    try:
        scans = get_recent_scans()
        return jsonify(scans), 200
    except Exception as e:
        logger.error(f"❌ History fetch failed: {e}")
        return jsonify({"error": "Failed to fetch history"}), 500

# =====================================================
# ERROR HANDLERS
# =====================================================
@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    logger.warning(f"⚠️ Request entity too large: {error}")
    return jsonify({
        "error": f"File too large. Max: {config['security']['max_content_length_mb']} MB"
    }), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server error."""
    logger.error(f"❌ Internal server error: {error}", exc_info=True)
    return jsonify({
        "error": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle not found error."""
    logger.warning(f"⚠️ 404 Not Found: {request.path}")
    return jsonify({
        "error": "Endpoint not found"
    }), 404

# =====================================================
# RUN APP
# =====================================================
if __name__ == "__main__":
    logger.info("="*60)
    logger.info(f"🚀 Starting {config['api']['name']} v{config['api']['version']}")
    logger.info(f"   Host: {config['flask']['host']}")
    logger.info(f"   Port: {config['flask']['port']}")
    logger.info(f"   Debug: {config['flask']['debug']}")
    logger.info("="*60)
    logger.info("✅ Health check available at: GET /health")
    logger.info("✅ Prediction endpoint available at: POST /predict")
    logger.info("✅ History endpoint available at: GET /history")
    logger.info("="*60)
    
    try:
        app.run(
            host=config['flask']['host'],
            port=config['flask']['port'],
            debug=config['flask']['debug'],
            threaded=config['flask']['threaded']
        )
    except Exception as e:
        logger.error(f"❌ Failed to start app: {e}")
        sys.exit(1)
