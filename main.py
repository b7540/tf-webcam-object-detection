from __future__ import annotations

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

MODEL_URL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
CONFIDENCE = 0.5
CAMERA_INDEX = 0
WINDOW_NAME = "Object Detection (press Q to quit)"

COCO_NAMES: dict[int, str] = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork",
    49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple",
    54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog",
    59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch",
    64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet", 72: "tv",
    73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone",
    78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator",
    84: "book", 85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
    89: "hair drier", 90: "toothbrush",
}


class ObjectDetector:

    def __init__(self, model_url: str) -> None:
        print("Loading model (first run downloads ~25 MB)...")
        module = hub.load(model_url)
        self._predict = self._resolve_predict_fn(module)
        print("Model ready.")

    @staticmethod
    def _resolve_predict_fn(module):
        if getattr(module, "signatures", None):
            names = list(module.signatures.keys())
            print(f"  signatures found: {names}")
            for key in ("default", "serving_default"):
                if key in module.signatures:
                    return module.signatures[key]
            return module.signatures[names[0]]
        print("  no signatures — calling module directly")
        return module

    @staticmethod
    def _call_predict(predict_fn, batch):
        try:
            return predict_fn(batch)
        except TypeError:
            return predict_fn(images=batch)

    def detect(self, bgr_frame: np.ndarray) -> list[dict]:
        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        batch = tf.expand_dims(tf.convert_to_tensor(rgb, dtype=tf.uint8), 0)

        out = self._call_predict(self._predict, batch)

        boxes = out["detection_boxes"].numpy()[0]
        classes = out["detection_classes"].numpy()[0]
        scores = out["detection_scores"].numpy()[0]
        count = int(out["num_detections"].numpy())

        results: list[dict] = []
        for i in range(min(len(scores), count)):
            if scores[i] < CONFIDENCE:
                continue

            ymin, xmin, ymax, xmax = boxes[i]
            class_id = int(classes[i])
            results.append({
                "name": COCO_NAMES.get(class_id, f"class_{class_id}"),
                "score": float(scores[i]),
                "box": (
                    int(xmin * w),
                    int(ymin * h),
                    int(xmax * w),
                    int(ymax * h),
                ),
            })
        return results


def draw_detections(frame: np.ndarray, detections: list[dict]) -> None:
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f"{det['name']} {det['score'] * 100:.0f}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    return cap


def main() -> int:
    print(f"TensorFlow {tf.__version__}")

    try:
        detector = ObjectDetector(MODEL_URL)
    except Exception as exc:
        print(f"Failed to load model: {exc}")
        return 1

    cap = open_camera(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Cannot open camera {CAMERA_INDEX}. Try CAMERA_INDEX = 1")
        return 1

    print("Camera on. Press Q in the video window to exit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            detections = detector.detect(frame)
            draw_detections(frame, detections)
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
