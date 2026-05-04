import os
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
from pydub import AudioSegment
import warnings

# -------------------- FFmpeg setup --------------------
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub")

# Explicitly set FFmpeg path
FFMPEG_BIN = r"C:\Users\TITLI DUTTA\Desktop\ffmpeg-7.1.1-essentials_build\bin"
AudioSegment.converter = os.path.join(FFMPEG_BIN, "ffmpeg.exe")
AudioSegment.ffmpeg = os.path.join(FFMPEG_BIN, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(FFMPEG_BIN, "ffprobe.exe")

# -------------------- Flask app setup --------------------
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "data", "raw")
ALLOWED_EXTENSIONS = {"mp3", "wav", "webm"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ALLOWED_EXTENSIONS"] = ALLOWED_EXTENSIONS
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------- Helper functions --------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

# -------------------- Audio preprocessing --------------------
def convert_to_wav(file_path):
    """
    Converts any audio file to standard WAV (16-bit PCM, mono, 22050 Hz).
    """
    try:
        wav_path = os.path.splitext(file_path)[0] + "_converted.wav"
        # Auto-detect format (mp3, wav, webm etc.)
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_channels(1).set_frame_rate(22050).set_sample_width(2)
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        print(f"[convert_to_wav] Error: {e}")
        return None

# -------------------- Feature extraction --------------------
from model.feature_extraction import extract_features, FEATURE_NAMES

# -------------------- Load model & encoder --------------------
MODEL_PATH = os.path.join(os.getcwd(), "model", "knn_model.pkl")
ENCODER_PATH = os.path.join(os.getcwd(), "model", "label_encoder.pkl")

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

# -------------------- Routes --------------------
@app.route("/")
def serve_index():
    return send_from_directory(directory="templates", path="index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        # Convert to WAV
        wav_path = convert_to_wav(file_path)
        if not wav_path:
            return jsonify({"error": "Failed to convert audio to WAV"}), 500

        # Extract features
        features = extract_features(wav_path)
        if not features:
            return jsonify({"error": "Failed to extract audio features"}), 500

        # Build input array in fixed order
        X_input = np.array([[features[feat] for feat in FEATURE_NAMES]])

        # Predict
        pred_numeric = model.predict(X_input)[0]
        pred_label = label_encoder.inverse_transform([pred_numeric])[0]

        return jsonify({"prediction": pred_label})

    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(traceback_str)
        return jsonify({"error": str(e), "trace": traceback_str}), 500

    finally:
        # Cleanup uploaded and converted files
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            if 'wav_path' in locals() and os.path.exists(wav_path) and wav_path != file_path:
                os.remove(wav_path)
        except Exception as cleanup_error:
            print(f"[cleanup] Failed to delete files: {cleanup_error}")

# -------------------- Main --------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
