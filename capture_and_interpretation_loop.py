import os
import sys
import time
import hashlib
import subprocess
import argparse

from PIL import Image
import cv2
import numpy as np

from hailo_platform import VDevice
from hailo_platform.genai import VLM

from hailo_apps.python.core.common.core import handle_list_models_flag, resolve_hef_path
from hailo_apps.python.core.common.defines import (
    VLM_CHAT_APP,
    SHARED_VDEVICE_GROUP_ID,
    HAILO10H_ARCH,
)
from hailo_apps.python.core.common.hailo_logger import get_logger


# Initialize logger
logger = get_logger(__name__)

# Directory where captured frames will be stored
CAPTURE_DIR = "captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)


def capture_frame():
    """
    Capture a screenshot from the connected Android device using ADB.

    Returns:
        bytes: Raw PNG image bytes from 'adb exec-out screencap -p'.
    """
    result = subprocess.run(
        ["adb", "exec-out", "screencap", "-p"],
        capture_output=True
    )
    return result.stdout


def get_frame_hash(png_bytes):
    """
    Compute a hash for the given PNG bytes to detect screen changes.

    Args:
        png_bytes (bytes): Screenshot data.

    Returns:
        str: MD5 hash string of the image bytes.
    """
    return hashlib.md5(png_bytes).hexdigest()


def save_frame(png_bytes, label="frame"):
    """
    Save the PNG bytes to disk under the capture directory.

    Args:
        png_bytes (bytes): Screenshot data.
        label (str): Base label for the file name.

    Returns:
        str: Path to the saved image file.
    """
    timestamp = int(time.time())
    path = os.path.join(CAPTURE_DIR, f"{label}_{timestamp}.png")
    with open(path, "wb") as f:
        f.write(png_bytes)
    print(f"Saved: {path}")
    return path


print("Starting capture loop... Press Ctrl+C to stop")

last_hash = None
frame_count = 0


def vlm_chat(image_path):
    """
    Run the Hailo VLM on a given image path and print a semantic description.

    This:
    - parses command-line arguments for HEF path / model listing,
    - initializes the Hailo VDevice and loads the VLM model,
    - prepares a multimodal prompt with the image and a text question,
    - runs generation and prints the textual response,
    - releases resources on exit.
    """
    parser = argparse.ArgumentParser(description="VLM Chat Example")
    parser.add_argument("--hef-path", type=str, default=None, help="Path to HEF model file")
    parser.add_argument("--list-models", action="store_true", help="List available models")

    # Allow listing models without full initialization
    handle_list_models_flag(parser, VLM_CHAT_APP)
    args = parser.parse_args()

    # Resolve HEF path (may auto-download); VLM is Hailo-10H only
    hef_path = resolve_hef_path(args.hef_path, app_name=VLM_CHAT_APP, arch=HAILO10H_ARCH)
    if hef_path is None:
        logger.error("Failed to resolve HEF path for VLM model.")
        sys.exit(1)

    logger.info(f"Using HEF: {hef_path}")
    print(f"Model file found: {hef_path}")

    vdevice = None
    vlm = None

    try:
        # Initialize Hailo device
        print("Initializing Hailo device...")
        params = VDevice.create_params()
        params.group_id = SHARED_VDEVICE_GROUP_ID
        vdevice = VDevice(params)
        print("Hailo device initialized")

        # Load VLM model
        print("Loading VLM model...")
        vlm = VLM(vdevice, str(hef_path))
        print("Model loaded successfully")

        # Define a simple multimodal prompt: one image + text question
        prompt = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a helpful assistant that analyzes images, "
                            "semantically interprets and describes images, and answers "
                            "questions about them."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe the android screen in detail."},
                ],
            },
        ]

        # Load image from disk
        print(f"Loading image from: {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image file: {image_path}")
        print(f"Image loaded (size: {image.shape[1]}x{image.shape[0]})")

        # Preprocess: convert BGR->RGB and resize to model input size
        print("Preprocessing image...")
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (336, 336), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        print("Image preprocessed (resized to 336x336, converted to RGB)")

        # Run VLM generation
        print("Sending prompt with image to VLM...")
        print(f"User question: '{prompt[1]['content'][1]['text']}'")
        response = vlm.generate_all(
            prompt=prompt,
            frames=[image],
            temperature=0.1,
            seed=42,
            max_generated_tokens=200,
        )

        # Print only the textual answer, trimming model-specific markers
        print("Response received:")
        print("-" * 60)
        cleaned = response.split(". [{'type'")[0].split("<|im_end|>")[0]
        print(cleaned)
        print("-" * 60)
        print("Example completed successfully")

    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        sys.exit(1)

    finally:
        # Release VLM and device resources
        if vlm:
            try:
                vlm.clear_context()
                vlm.release()
            except Exception as e:
                logger.warning(f"Error releasing VLM: {e}")

        if vdevice:
            try:
                vdevice.release()
            except Exception as e:
                logger.warning(f"Error releasing VDevice: {e}")


# Main capture loop:
# - Polls the Android device via ADB screencap
# - Detects when the screen image has changed (via hash)
# - Saves the new frame and sends it to the VLM
while True:
    frame_bytes = capture_frame()

    if not frame_bytes:
        print("Capture failed — check ADB connection")
        time.sleep(1)
        continue

    current_hash = get_frame_hash(frame_bytes)

    # Only process when the screen content changes
    if current_hash != last_hash:
        frame_count += 1
        print(f"Screen changed — capturing frame {frame_count}")
        path = save_frame(frame_bytes, label=f"screen_{frame_count:03d}")
        vlm_chat(path)
        last_hash = current_hash
    else:
        print("No change detected")

    # Wait a bit before checking again
    time.sleep(3)
