import base64
import json
import os
from math import log2
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing.image import img_to_array

BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve_model_dir():
    if os.getenv("FER_MODEL_DIR"):
        return Path(os.getenv("FER_MODEL_DIR"))
    if os.getenv("VERCEL"):
        return Path("/tmp/fer_model")
    return BASE_DIR / "fer_model"


MODEL_DIR = _resolve_model_dir()
MODEL_DIR.mkdir(parents=True, exist_ok=True)
EMOTION_INPUT_SIZE = (48, 48)
MIN_FACE_SIZE = 48
MTCNN_MIN_CONF = 0.90
DEFAULT_THRESHOLD = 0.30
_input_size = EMOTION_INPUT_SIZE

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
haar_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

try:
    from mtcnn.mtcnn import MTCNN
    mtcnn_detector = MTCNN()
    use_mtcnn = True
    print("Using MTCNN for face detection.")
except Exception:
    mtcnn_detector = None
    use_mtcnn = False
    print("MTCNN not available. Falling back to Haar Cascade.")

_model = None
_model_name = None
_model_labels = []
_resnet = None

# To keep memory usage low on small dynos, only load the smallest legacy model.
MODEL_CANDIDATES = [
    {
        "name": "Legacy",
        "path": MODEL_DIR / "emotion_model.h5",
        "env_url": "LEGACY_BLOB_URL",
        "labels": ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"],  # legacy 6-class FER
        "loader": lambda cfg: load_model(cfg["path"]),
    },
]


def _load_json_model(json_path: Path, weights_path: Path):
    if not json_path.exists() or not weights_path.exists():
        raise FileNotFoundError("JSON or weights file missing for CK+ model.")
    with open(json_path, "r", encoding="utf-8") as f:
        architecture = json.load(f)
    model = model_from_json(json.dumps(architecture))
    model.load_weights(weights_path)
    return model


def _autofill_labels(output_dim: int):
    if output_dim == 6:
        return ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    if output_dim == 7:
        return ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    if output_dim == 8:
        return ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Angry", "Contempt"]
    return [f"Class_{i}" for i in range(output_dim)]


def _select_largest_face(boxes):
    if not boxes:
        return None
    return max(boxes, key=lambda b: b[2] * b[3])


def _download_blob(url: str, dest: Path, timeout: int = 60):
    if not url:
        return False
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 512):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as exc:
        print(f"Download failed for {url}: {exc}")
        return False


def _ensure_candidate_files(candidate):
    tasks = []
    url = candidate.get("env_url")
    if url:
        tasks.append((url, candidate.get("path")))
    json_url = candidate.get("json_env_url")
    if json_url:
        tasks.append((json_url, candidate.get("json")))
    weights_url = candidate.get("weights_env_url")
    if weights_url:
        tasks.append((weights_url, candidate.get("weights")))

    for env_key, target in tasks:
        if not target:
            continue
        env_val = os.getenv(env_key, "").strip()
        if env_val and not target.exists():
            _download_blob(env_val, target)


def get_model(force_reload: bool = False):
    global _model, _model_name, _model_labels, _input_size
    if _model is not None and not force_reload:
        return _model

    for candidate in MODEL_CANDIDATES:
        _ensure_candidate_files(candidate)
        loader = candidate.get("loader")
        if not loader:
            continue
        if "path" in candidate and not candidate["path"].exists():
            continue
        if "json" in candidate and (not candidate["json"].exists() or not candidate["weights"].exists()):
            continue
        try:
            loaded = loader(candidate)
            input_shape = getattr(loaded, "input_shape", None)
            if not input_shape or len(input_shape) != 4 or input_shape[1] is None or input_shape[2] is None:
                print(f"Skipping {candidate.get('name')} due to incompatible input shape: {input_shape}")
                continue
            output_dim = loaded.output_shape[-1]
            labels = candidate.get("labels") or []
            if len(labels) != output_dim:
                labels = _autofill_labels(output_dim)
            _model = loaded
            _model_name = candidate["name"]
            _model_labels = labels
            _input_size = (int(input_shape[1]), int(input_shape[2]))
            print(f"Loaded {_model_name} model with {output_dim} outputs and input size {_input_size}.")
            return _model
        except Exception as exc:
            print(f"Failed to load {candidate.get('name')} model: {exc}")

    _model = None
    _model_name = None
    _model_labels = []
    print("No emotion model could be loaded.")
    return None


def get_resnet():
    global _resnet
    if _resnet is not None:
        return _resnet
    try:
        _resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
        print("ResNet50 loaded for feature extraction.")
    except Exception as exc:
        print(f"ResNet50 load failed: {exc}")
        _resnet = None
    return _resnet


def preprocess_face(image_path: str):
    if not image_path or not isinstance(image_path, str):
        return None

    img_cv = cv2.imread(image_path)
    if img_cv is None:
        return None

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    face_img = None
    box = None

    if use_mtcnn and mtcnn_detector:
        rgb_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        faces = mtcnn_detector.detect_faces(rgb_img)
        filtered = [f for f in faces if f.get("confidence", 0) >= MTCNN_MIN_CONF]
        primary = _select_largest_face([f.get("box") for f in filtered if f.get("box")])
        if primary:
            x, y, w, h = primary
            x, y = max(0, x), max(0, y)
            box = (x, y, w, h)
            face_img = gray[y:y + h, x:x + w]
    else:
        faces = haar_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
        )
        if len(faces) > 0:
            box = _select_largest_face(faces)
            x, y, w, h = box
            face_img = gray[y:y + h, x:x + w]

    if face_img is None or face_img.size == 0:
        return None

    if face_img.shape[0] < MIN_FACE_SIZE or face_img.shape[1] < MIN_FACE_SIZE:
        return None

    enhanced = clahe.apply(face_img)
    target_size = _input_size or EMOTION_INPUT_SIZE
    resized = cv2.resize(enhanced, target_size)
    arr = resized.astype("float32") / 255.0
    arr = arr.reshape(1, target_size[0], target_size[1], 1)

    ok, buffer = cv2.imencode(".png", resized)
    preview_b64 = base64.b64encode(buffer).decode("utf-8") if ok else None

    return {
        "array": arr,
        "detector": "mtcnn" if use_mtcnn and mtcnn_detector else "haar",
        "box": box,
        "preview_b64": preview_b64,
    }


def predict_emotion(face_array, threshold: float = DEFAULT_THRESHOLD, top_k: int = 3):
    model = get_model()
    if model is None or face_array is None:
        return {"label": "Model unavailable", "confidence": 0.0, "top": []}

    arr = np.asarray(face_array).astype("float32")
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]

    expected_input = getattr(model, "input_shape", None)
    if expected_input and len(expected_input) == 4 and expected_input[1] and expected_input[2]:
        target_h, target_w = int(expected_input[1]), int(expected_input[2])
        if arr.shape[1] != target_h or arr.shape[2] != target_w:
            arr = np.stack([
                cv2.resize(arr[i, ..., 0], (target_w, target_h))[:, :, None]
                for i in range(arr.shape[0])
            ], axis=0)
    elif expected_input and len(expected_input) == 2 and expected_input[1]:
        flat_size = int(expected_input[1])
        arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[1] > flat_size:
            arr = arr[:, :flat_size]
        elif arr.shape[1] < flat_size:
            pad = flat_size - arr.shape[1]
            arr = np.pad(arr, ((0, 0), (0, pad)), mode="constant")
    else:
        target_h, target_w = EMOTION_INPUT_SIZE
        if arr.shape[1] != target_h or arr.shape[2] != target_w:
            arr = np.stack([
                cv2.resize(arr[i, ..., 0], (target_w, target_h))[:, :, None]
                for i in range(arr.shape[0])
            ], axis=0)

    # Light test-time augmentation to stabilize predictions
    variants = [arr]
    if arr.ndim == 4:
        for factor in (0.9, 1.1):
            variants.append(np.clip(arr * factor, 0.0, 1.0))

    batches = []
    for v in variants:
        batches.append(v)
        if v.ndim == 4 and v.shape[-1] == 1:
            batches.append(np.flip(v, axis=2))

    batch = np.concatenate(batches, axis=0)
    preds = model.predict(batch, verbose=0).mean(axis=0)
    labels = _model_labels or _autofill_labels(len(preds))
    indexed = list(enumerate(preds))
    indexed.sort(key=lambda x: x[1], reverse=True)
    top = [(labels[i] if i < len(labels) else f"Class_{i}", float(score)) for i, score in indexed[:top_k]]

    best_label, best_conf = top[0]
    if best_conf < threshold:
        best_label = "Uncertain"

    return {
        "label": best_label,
        "confidence": best_conf,
        "top": top,
        "labels": labels,
        "model_name": _model_name or "Unknown",
    }


def extract_features_with_resnet(image_path):
    resnet = get_resnet()
    if resnet is None:
        return None
    try:
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        img_arr = img_to_array(img)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = preprocess_input(img_arr)
        return resnet.predict(img_arr, verbose=0)
    except Exception:
        return None


def summarize_emotion_stream(entries, threshold: float = DEFAULT_THRESHOLD):
    """
    Summarize a list of streaming emotion predictions.
    Each entry is expected to be a dict with keys: label, confidence, and timestamp.
    Returns distribution, dominant label, entropy, std of confidences, and coverage above threshold.
    """
    if not entries:
        return {
            "dominant": None,
            "total": 0,
            "avg_confidence": 0.0,
            "std_confidence": 0.0,
            "entropy": 0.0,
            "coverage": 0.0,
            "counts": {},
            "distribution": [],
            "last_label": None,
        }

    counts = {}
    confidences = []
    coverage_hits = 0
    last_label = entries[-1].get("label")
    for item in entries:
        label = item.get("label", "Unknown")
        counts[label] = counts.get(label, 0) + 1
        try:
            conf_val = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf_val = 0.0
        confidences.append(conf_val)
        if conf_val >= threshold:
            coverage_hits += 1

    total = len(entries)
    dominant = max(counts.items(), key=lambda kv: kv[1])[0] if counts else None
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    std_conf = float(np.std(confidences)) if confidences else 0.0

    distribution = []
    entropy = 0.0
    for label, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        p = count / total
        distribution.append((label, round(p * 100, 2)))
        if p > 0:
            entropy -= p * log2(p)

    coverage = round((coverage_hits / total) * 100, 2) if total else 0.0

    return {
        "dominant": dominant,
        "total": total,
        "avg_confidence": avg_conf,
        "std_confidence": std_conf,
        "entropy": round(entropy, 3),
        "coverage": coverage,
        "counts": counts,
        "distribution": distribution,
        "last_label": last_label,
    }


def get_suggestion(emotion):
    suggestions = {
        "Happy": "Keep smiling and share that energy with someone near you.",
        "Sad": "Pause, breathe, and give yourself permission to rest or reach out.",
        "Angry": "Step away for a minute, breathe deep, and reset.",
        "Fear": "Name the worry and talk it through with someone you trust.",
        "Disgust": "Shift attention to something you value-music, a walk, or a quick call.",
        "Surprise": "Take a beat to process what changed before reacting.",
        "Neutral": "You're steady. Use this moment to plan your next move.",
        "Contempt": "Notice the tension and reframe-what would empathy add here'",
        "Uncertain": "Try a clearer, well-lit image facing the camera.",
    }
    return suggestions.get(emotion, "Stay curious about how you feel-small check-ins help.")


def model_summary():
    return {"name": _model_name or "Not loaded", "labels": _model_labels, "threshold": DEFAULT_THRESHOLD}
