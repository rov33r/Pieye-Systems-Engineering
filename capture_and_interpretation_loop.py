import time
import cv2
import numpy as np
import threading
import base64
from flask import Flask, jsonify

from hailo_platform import VDevice
from hailo_platform.genai import VLM
from hailo_apps.python.core.common.core import resolve_hef_path
from hailo_apps.python.core.common.defines import SHARED_VDEVICE_GROUP_ID, HAILO10H_ARCH, VLM_CHAT_APP

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
CONFIG = {
    "hef_path": None,
    "v4l2_device": "/dev/video10",

    "system_prompt": "You are a UI testing assistant for android phones. Provide objective structured descriptions.",
    "user_prompt": ("Describe the UI clearly and objectively."),

    "temperature": 1.0,
    "max_tokens": 700,
    "seed": 42,
}

# ─────────────────────────────
# INIT MODEL
# ─────────────────────────────
print("Initializing Hailo...")
params = VDevice.create_params()
params.group_id = SHARED_VDEVICE_GROUP_ID
vdevice = VDevice(params)

hef_path = resolve_hef_path(CONFIG["hef_path"], app_name=VLM_CHAT_APP, arch=HAILO10H_ARCH)
vlm = VLM(vdevice, str(hef_path))
print("✓ Model ready")

# ─────────────────────────────
# GLOBAL STATE
# ─────────────────────────────
latest_result = None
latest_image = None
latest_timestamp = None
is_running = False

# ─────────────────────────────
# VIDEO CAPTURE (persistent!)
# ─────────────────────────────
cap = cv2.VideoCapture(CONFIG["v4l2_device"])
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open /dev/video0 — did you start scrcpy with --v4l2-sink?")

# ─────────────────────────────
# HELPERS
# ─────────────────────────────
def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (336, 336)).astype(np.uint8)

def encode_image(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode()

def get_latest_frame():
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read frame from scrcpy")
    return frame

def query_vlm(image):
    prompt = [
        {"role": "system", "content": [{"type": "text", "text": CONFIG["system_prompt"]}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": CONFIG["user_prompt"]}]}
    ]

    response = vlm.generate_all(
        prompt=prompt,
        frames=[image],
        temperature=CONFIG["temperature"],
        seed=CONFIG["seed"],
        max_generated_tokens=CONFIG["max_tokens"]
    )

    vlm.clear_context()
    return response.split(". [{'type'")[0].split("<|im_end|>")[0]

def run_pipeline():
    frame = get_latest_frame()
    processed = preprocess(frame)

    result = query_vlm(processed)
    img = encode_image(frame)

    return result, img

# ─────────────────────────────
# BACKGROUND THREAD
# ─────────────────────────────
def background_job():
    global latest_result, latest_image, latest_timestamp, is_running

    try:
        is_running = True
        result, img = run_pipeline()

        latest_result = result
        latest_image = img
        latest_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    except Exception as e:
        latest_result = f"ERROR: {e}"
        latest_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    finally:
        is_running = False

# ─────────────────────────────
# FLASK API
# ─────────────────────────────
app = Flask(__name__)

@app.route("/trigger", methods=["POST"])
def trigger():
    global is_running

    if is_running:
        return jsonify({"status": "busy"})

    threading.Thread(target=background_job).start()
    return jsonify({"status": "started"})

@app.route("/result", methods=["GET"])
def result():
    return jsonify({
        "running": is_running,
        "result": latest_result,
        "timestamp": latest_timestamp,
        "image": latest_image
    })

# ─────────────────────────────
# RUN
# ─────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
