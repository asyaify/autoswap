"""
Full pipeline: Reference image + Child face swap + Face restoration
=================================================================
Step 1: Load reference image (contains: background + pose + clothing)
Step 2: Load child's real photo (source for face)
Step 3: ReActor face swap — put child's face onto reference
Step 4: GFPGAN face restoration for quality
Step 5: Save result
"""
import json
import urllib.request
import time
import sys

SERVER = "http://127.0.0.1:8188"

# --- Configuration ---
REFERENCE_IMAGE = "референс.png"     # Reference: pose + outfit + background
CHILD_PHOTO = "DSC08443.JPG"          # Real child's face photo

prompt = {
    # Load reference image (the model child in pose + outfit + background)
    "1": {
        "class_type": "LoadImage",
        "inputs": {
            "image": REFERENCE_IMAGE
        }
    },
    # Load child's real photo (source face)
    "2": {
        "class_type": "LoadImage",
        "inputs": {
            "image": CHILD_PHOTO
        }
    },
    # ReActor Face Swap: put child's face onto reference
    "3": {
        "class_type": "ReActorFaceSwap",
        "inputs": {
            "enabled": True,
            "input_image": ["1", 0],       # Reference image (target)
            "source_image": ["2", 0],       # Child photo (source face)
            "swap_model": "inswapper_128.onnx",
            "facedetection": "retinaface_resnet50",
            "face_restore_model": "GFPGANv1.4.pth",
            "face_restore_visibility": 1.0,
            "codeformer_weight": 0.5,
            "detect_gender_input": "no",
            "detect_gender_source": "no",
            "input_faces_index": "0",
            "source_faces_index": "0",
            "console_log_level": 1
        }
    },
    # Save result
    "4": {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["3", 0],
            "filename_prefix": "faceswap_result"
        }
    }
}

payload = json.dumps({"prompt": prompt}).encode("utf-8")
req = urllib.request.Request(
    f"{SERVER}/prompt",
    data=payload,
    headers={"Content-Type": "application/json"}
)

print(f"Face swap pipeline: {REFERENCE_IMAGE} + face from {CHILD_PHOTO}")
print("Submitting...")
try:
    resp = urllib.request.urlopen(req)
    result = json.loads(resp.read())
    prompt_id = result.get("prompt_id")
    print(f"Prompt ID: {prompt_id}")
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f"ERROR {e.code}: {body[:2000]}")
    sys.exit(1)

print("Waiting for execution...")
for i in range(300):
    time.sleep(2)
    try:
        hist_req = urllib.request.urlopen(f"{SERVER}/history/{prompt_id}")
        hist = json.loads(hist_req.read())
        if prompt_id in hist:
            outputs = hist[prompt_id].get("outputs", {})
            status = hist[prompt_id].get("status", {})
            print(f"Status: {status}")
            if outputs:
                for node_id, out in outputs.items():
                    if "images" in out:
                        for img_info in out["images"]:
                            fname = img_info["filename"]
                            print(f"Output: {fname}")
            break
    except Exception:
        pass
    if i % 10 == 0 and i > 0:
        print(f"  Still waiting... {i*2}s")
else:
    print("Timed out")
