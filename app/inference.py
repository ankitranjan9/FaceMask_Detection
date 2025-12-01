import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "app/models/mask_detector.h5"
FACE_PROTO = "app/face_detector/deploy.prototxt"
FACE_MODEL = "app/face_detector/res10_300x300_ssd_iter_140000.caffemodel"

class MaskDetector:
    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
        self.model = load_model(MODEL_PATH)

    def predict_image(self, image_bgr):
        (h, w) = image_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image_bgr, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        results = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = image_bgr[startY:endY, startX:endX]
            if face.size == 0:
                continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = preprocess_input(face.astype("float32"))
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = self.model.predict(face)[0]
            label = "Mask" if mask > withoutMask else "No Mask"
            prob = float(max(mask, withoutMask))

            results.append(
                {"box": [int(startX), int(startY), int(endX), int(endY)],
                 "label": label,
                 "probability": prob}
            )
        return results
