from flask import Flask, request, jsonify
import cv2
import numpy as np
from .inference import MaskDetector

app = Flask(__name__)
detector = MaskDetector()

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file:
        return jsonify(error="No image uploaded"), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    results = detector.predict_image(image)
    return jsonify(results=results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
