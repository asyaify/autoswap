#!/usr/bin/env python3
"""
Photo Studio — Web UI for Qwen Image Edit pipeline.
Wraps outfit change & background merge ComfyUI workflows.
"""

import json
import os
import random
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, send_file

COMFYUI_SERVER = "127.0.0.1:8188"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
STATIC_DIR = os.path.join(SCRIPT_DIR, "web_ui")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")

# ── ComfyUI helpers ──────────────────────────────────────


def comfy_upload(filepath: str) -> str:
    filename = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        file_data = f.read()
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + file_data + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="type"\r\n\r\n'
        f"input\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="overwrite"\r\n\r\n'
        f"true\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    req = urllib.request.Request(
        f"http://{COMFYUI_SERVER}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    return result.get("name", filename)


def comfy_queue(prompt: dict) -> str:
    client_id = str(uuid.uuid4())
    payload = json.dumps({"prompt": prompt, "client_id": client_id}).encode()
    req = urllib.request.Request(
        f"http://{COMFYUI_SERVER}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    return result["prompt_id"]


def comfy_progress(prompt_id: str) -> dict | None:
    try:
        req = urllib.request.Request(f"http://{COMFYUI_SERVER}/history/{prompt_id}")
        with urllib.request.urlopen(req) as resp:
            history = json.loads(resp.read())
        if prompt_id in history:
            return history[prompt_id]
    except Exception:
        pass
    return None


def comfy_queue_info() -> dict:
    try:
        req = urllib.request.Request(f"http://{COMFYUI_SERVER}/queue")
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except Exception:
        return {}


# ── Workflow builders ────────────────────────────────────


def _base_model_nodes(use_lightning=True):
    p = {
        "37": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": "Qwen-Image-Edit-2509-Q5_K_M.gguf"}
        },
        "38": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "type": "qwen_image",
                "device": "default"
            }
        },
        "39": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_vae.safetensors"}
        },
    }
    if use_lightning:
        p["89"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["37", 0],
                "lora_name": "Qwen-Image-Lightning-4steps-V1.0.safetensors",
                "strength_model": 1.0
            }
        }
        model_out = ["89", 0]
    else:
        model_out = ["37", 0]
    p["66"] = {
        "class_type": "ModelSamplingAuraFlow",
        "inputs": {"model": model_out, "shift": 3.0}
    }
    p["75"] = {
        "class_type": "CFGNorm",
        "inputs": {"model": ["66", 0], "strength": 1.0}
    }
    return p


def _bfs_model_nodes(use_lightning=True):
    """Model nodes with BFS Head V3 LoRA (+ optional Lightning LoRA)."""
    p = {
        "37": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": "Qwen-Image-Edit-2509-Q5_K_M.gguf"}
        },
        "38": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "type": "qwen_image",
                "device": "default"
            }
        },
        "39": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_vae.safetensors"}
        },
        "90": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["37", 0],
                "lora_name": "bfs_head_v3_qwen_image_edit_2509.safetensors",
                "strength_model": 1.0
            }
        },
    }
    if use_lightning:
        p["89"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["90", 0],
                "lora_name": "Qwen-Image-Lightning-4steps-V1.0.safetensors",
                "strength_model": 1.0
            }
        }
        model_out = ["89", 0]
    else:
        model_out = ["90", 0]
    p["66"] = {
        "class_type": "ModelSamplingAuraFlow",
        "inputs": {"model": model_out, "shift": 3.0}
    }
    p["75"] = {
        "class_type": "CFGNorm",
        "inputs": {"model": ["66", 0], "strength": 1.0}
    }
    return p


def _build_workflow(model_nodes, img1_file, img2_file, positive, negative, seed, prefix, steps=4, megapixels=1.5):
    """Generic workflow builder reused by all modes."""
    p = dict(model_nodes)
    p.update({
        "78": {"class_type": "LoadImage", "inputs": {"image": img1_file, "upload": "image"}},
        "106": {"class_type": "LoadImage", "inputs": {"image": img2_file, "upload": "image"}},
        "93": {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {"image": ["78", 0], "upscale_method": "lanczos", "megapixels": megapixels, "resolution_steps": 1}
        },
        "88": {"class_type": "VAEEncode", "inputs": {"pixels": ["93", 0], "vae": ["39", 0]}},
        "111": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {"prompt": positive, "clip": ["38", 0], "vae": ["39", 0], "image1": ["93", 0], "image2": ["106", 0]}
        },
        "110": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {"prompt": negative, "clip": ["38", 0], "vae": ["39", 0], "image1": ["93", 0], "image2": ["106", 0]}
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["75", 0], "positive": ["111", 0], "negative": ["110", 0],
                "latent_image": ["88", 0], "seed": seed, "control_after_generate": "randomize",
                "steps": steps, "cfg": 1.0, "sampler_name": "euler", "scheduler": "beta", "denoise": 1.0
            }
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["39", 0]}},
        "60": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    })
    return p


def build_outfit_prompt(person_file, outfit_file, positive, negative, seed, prefix, steps=4, megapixels=1.5, use_lightning=True):
    return _build_workflow(
        _base_model_nodes(use_lightning), person_file, outfit_file,
        positive, negative, seed, prefix, steps, megapixels
    )


def build_background_prompt(person_file, bg_file, positive, negative, seed, prefix, steps=4, megapixels=1.5, use_lightning=True):
    return _build_workflow(
        _base_model_nodes(use_lightning), person_file, bg_file,
        positive, negative, seed, prefix, steps, megapixels
    )


def build_facefix_prompt(body_file, face_file, positive, negative, seed, prefix, steps=4, megapixels=1.5, use_lightning=True):
    """BFS Head V3 face swap: body=image1, face=image2 (inverted order)."""
    return _build_workflow(
        _bfs_model_nodes(use_lightning), body_file, face_file,
        positive, negative, seed, prefix, steps, megapixels
    )


# ── Routes ───────────────────────────────────────────────


@app.route("/")
def index():
    return send_file(os.path.join(STATIC_DIR, "index.html"))


@app.route("/api/images/input")
def list_input_images():
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    files = []
    for f in sorted(os.listdir(INPUT_DIR)):
        if Path(f).suffix.lower() in exts and not f.startswith("_"):
            files.append(f)
    return jsonify(files)


@app.route("/api/images/output")
def list_output_images():
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    files = []
    for f in sorted(os.listdir(OUTPUT_DIR), key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)), reverse=True):
        if Path(f).suffix.lower() in exts:
            files.append(f)
    return jsonify(files)


@app.route("/api/image/input/<path:filename>")
def serve_input_image(filename):
    return send_from_directory(INPUT_DIR, filename)


@app.route("/api/image/output/<path:filename>")
def serve_output_image(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    # Sanitize filename
    safe_name = os.path.basename(f.filename)
    save_path = os.path.join(INPUT_DIR, safe_name)
    f.save(save_path)
    return jsonify({"filename": safe_name})


@app.route("/api/use_output", methods=["POST"])
def use_output():
    """Copy an output image to input/ so it can be used in workflows."""
    data = request.json
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "No filename"}), 400
    safe_name = os.path.basename(filename)
    src = os.path.join(OUTPUT_DIR, safe_name)
    if not os.path.isfile(src):
        return jsonify({"error": f"Output not found: {safe_name}"}), 404
    dst = os.path.join(INPUT_DIR, safe_name)
    import shutil
    shutil.copy2(src, dst)
    return jsonify({"filename": safe_name})


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.json
    mode = data.get("mode")  # "outfit", "background", or "facefix"
    image1 = data.get("image1")  # person / body
    image2 = data.get("image2")  # outfit / background / face
    prompt = data.get("prompt", "")
    negative = data.get("negative", "blurry, distorted, artifacts")
    seed = data.get("seed")
    prefix = data.get("prefix", "web_ui")
    steps = int(data.get("steps", 4))
    megapixels = float(data.get("megapixels", 1.5))
    quality = data.get("quality", "fast")  # fast=lightning, hq=no lightning
    use_lightning = quality != "hq"
    if not use_lightning and steps <= 4:
        steps = 20  # sensible default without Lightning

    if not image1 or not image2:
        return jsonify({"error": "Two images required"}), 400

    if seed is None or seed == "":
        seed = random.randint(0, 2**53)
    else:
        seed = int(seed)

    # Upload images to ComfyUI
    img1_path = os.path.join(INPUT_DIR, image1)
    img2_path = os.path.join(INPUT_DIR, image2)

    if not os.path.isfile(img1_path):
        return jsonify({"error": f"Image not found: {image1}"}), 404
    if not os.path.isfile(img2_path):
        return jsonify({"error": f"Image not found: {image2}"}), 404

    try:
        name1 = comfy_upload(img1_path)
        name2 = comfy_upload(img2_path)
    except Exception as e:
        return jsonify({"error": f"ComfyUI upload failed: {e}"}), 502

    if mode == "outfit":
        workflow = build_outfit_prompt(name1, name2, prompt, negative, seed, prefix, steps, megapixels, use_lightning)
    elif mode == "background":
        workflow = build_background_prompt(name1, name2, prompt, negative, seed, prefix, steps, megapixels, use_lightning)
    elif mode == "facefix":
        workflow = build_facefix_prompt(name1, name2, prompt, negative, seed, prefix, steps, megapixels, use_lightning)
    else:
        return jsonify({"error": f"Unknown mode: {mode}"}), 400

    try:
        prompt_id = comfy_queue(workflow)
    except Exception as e:
        return jsonify({"error": f"ComfyUI queue failed: {e}"}), 502

    return jsonify({"prompt_id": prompt_id, "seed": seed})


@app.route("/api/status/<prompt_id>")
def status(prompt_id):
    result = comfy_progress(prompt_id)
    if result is None:
        # Check if still in queue
        qi = comfy_queue_info()
        running = qi.get("queue_running", [])
        pending = qi.get("queue_pending", [])
        in_queue = any(item[1] == prompt_id for item in running + pending)
        return jsonify({"status": "running" if in_queue else "pending"})

    # Extract output images
    images = []
    outputs = result.get("outputs", {})
    for nid, nout in outputs.items():
        for img in nout.get("images", []):
            images.append(img["filename"])

    status_info = result.get("status", {})
    if status_info.get("status_str") == "error":
        msgs = status_info.get("messages", [])
        error_text = str(msgs) if msgs else "Generation failed"
        return jsonify({"status": "error", "error": error_text})

    return jsonify({"status": "done", "images": images})


@app.route("/api/comfyui/status")
def comfyui_status():
    try:
        req = urllib.request.Request(f"http://{COMFYUI_SERVER}/system_stats")
        with urllib.request.urlopen(req, timeout=3) as resp:
            stats = json.loads(resp.read())
        return jsonify({"online": True, "stats": stats})
    except Exception:
        return jsonify({"online": False})


if __name__ == "__main__":
    os.makedirs(STATIC_DIR, exist_ok=True)
    print(f"\n  Photo Studio UI → http://localhost:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=False)
