#!/usr/bin/env python3
"""
Outfit change via Qwen Multi-Image Edit 2509 — ComfyUI API script.

Usage:
    python test_outfit_change.py [--person PATH] [--outfit PATH] [--prompt TEXT] [--seed N]

Defaults use девочка.JPG + одежда девочка.png from ComfyUI/input/.
Requires ComfyUI running at http://127.0.0.1:8188
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
import urllib.parse
import urllib.request
import uuid

SERVER = "127.0.0.1:8188"

# ── Node IDs in Qwen_Image_Edit_2509_Multi_Editing.json ──
NODE_LOAD_PERSON = "78"    # LoadImage — person photo (image1)
NODE_LOAD_OUTFIT = "106"   # LoadImage — outfit photo  (image2)
NODE_LOAD_EXTRA  = "108"   # LoadImage — extra (disabled)
NODE_POSITIVE    = "111"   # TextEncodeQwenImageEditPlus — positive prompt
NODE_NEGATIVE    = "110"   # TextEncodeQwenImageEditPlus — negative prompt
NODE_KSAMPLER    = "3"     # KSampler
NODE_SAVE        = "60"    # SaveImage


def load_workflow(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert GUI workflow (with "nodes" array) → API prompt (node_id → node_dict)
    prompt = {}
    for node in data["nodes"]:
        # Skip disabled nodes (mode 4 = bypassed/muted)
        if node.get("mode") == 4:
            continue

        class_type = node["type"]
        node_id = str(node["id"])
        inputs = {}

        # Collect widget values by mapping positionally
        widget_values = node.get("widgets_values", [])

        # Collect linked inputs
        linked = {}
        for inp in node.get("inputs", []):
            if inp.get("link") is not None:
                linked[inp["name"]] = inp["link"]

        prompt[node_id] = {
            "class_type": class_type,
            "inputs": inputs,
            "_widget_values": widget_values,
            "_linked": linked,
        }

    return data, prompt


def build_api_prompt(workflow_path: str, person_file: str, outfit_file: str,
                     positive: str, negative: str, seed: int) -> dict:
    """Build a proper ComfyUI API prompt from the workflow JSON."""

    # For the API, we construct the prompt dict directly from known node structure
    prompt = {
        # UnetLoaderGGUF (GGUF format works on Mac MPS, FP8 does not)
        "37": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {
                "unet_name": "Qwen-Image-Edit-2509-Q5_K_M.gguf"
            }
        },
        # CLIPLoader
        "38": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "type": "qwen_image",
                "device": "default"
            }
        },
        # VAELoader
        "39": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "qwen_image_vae.safetensors"
            }
        },
        # LoraLoaderModelOnly
        "89": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["37", 0],
                "lora_name": "Qwen-Image-Lightning-4steps-V1.0.safetensors",
                "strength_model": 1.0
            }
        },
        # ModelSamplingAuraFlow
        "66": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {
                "model": ["89", 0],
                "shift": 3.0
            }
        },
        # CFGNorm
        "75": {
            "class_type": "CFGNorm",
            "inputs": {
                "model": ["66", 0],
                "strength": 1.0
            }
        },
        # LoadImage — person (image1)
        NODE_LOAD_PERSON: {
            "class_type": "LoadImage",
            "inputs": {
                "image": person_file,
                "upload": "image"
            }
        },
        # LoadImage — outfit (image2)
        NODE_LOAD_OUTFIT: {
            "class_type": "LoadImage",
            "inputs": {
                "image": outfit_file,
                "upload": "image"
            }
        },
        # ImageScaleToTotalPixels — scale person image to 1MP
        "93": {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": [NODE_LOAD_PERSON, 0],
                "upscale_method": "lanczos",
                "megapixels": 1.0,
                "resolution_steps": 1
            }
        },
        # VAEEncode — encode scaled person image to latent
        "88": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["93", 0],
                "vae": ["39", 0]
            }
        },
        # TextEncodeQwenImageEditPlus — POSITIVE
        NODE_POSITIVE: {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": positive,
                "clip": ["38", 0],
                "vae": ["39", 0],
                "image1": ["93", 0],
                "image2": [NODE_LOAD_OUTFIT, 0],
            }
        },
        # TextEncodeQwenImageEditPlus — NEGATIVE
        NODE_NEGATIVE: {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": negative,
                "clip": ["38", 0],
                "vae": ["39", 0],
                "image1": ["93", 0],
                "image2": [NODE_LOAD_OUTFIT, 0],
            }
        },
        # KSampler
        NODE_KSAMPLER: {
            "class_type": "KSampler",
            "inputs": {
                "model": ["75", 0],
                "positive": [NODE_POSITIVE, 0],
                "negative": [NODE_NEGATIVE, 0],
                "latent_image": ["88", 0],
                "seed": seed,
                "control_after_generate": "randomize",
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "beta",
                "denoise": 1.0
            }
        },
        # VAEDecode
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": [NODE_KSAMPLER, 0],
                "vae": ["39", 0]
            }
        },
        # SaveImage
        NODE_SAVE: {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["8", 0],
                "filename_prefix": "outfit_change"
            }
        },
    }
    return prompt


def upload_image(filepath: str, subfolder: str = "", image_type: str = "input") -> str:
    """Upload an image to ComfyUI and return the filename."""
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
        f"{image_type}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="subfolder"\r\n\r\n'
        f"{subfolder}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="overwrite"\r\n\r\n'
        f"true\r\n"
        f"--{boundary}--\r\n"
    ).encode()

    req = urllib.request.Request(
        f"http://{SERVER}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    print(f"  Uploaded: {filename} → {result.get('name', filename)}")
    return result.get("name", filename)


def queue_prompt(prompt: dict) -> str:
    """Send prompt to ComfyUI and return prompt_id."""
    client_id = str(uuid.uuid4())
    payload = json.dumps({"prompt": prompt, "client_id": client_id}).encode()
    req = urllib.request.Request(
        f"http://{SERVER}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    prompt_id = result["prompt_id"]
    print(f"  Queued prompt: {prompt_id}")
    return prompt_id, client_id


def wait_for_completion(prompt_id: str, timeout: int = 1200) -> dict:
    """Poll /history until prompt completes."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"http://{SERVER}/history/{prompt_id}")
            with urllib.request.urlopen(req) as resp:
                history = json.loads(resp.read())
            if prompt_id in history:
                return history[prompt_id]
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"Prompt {prompt_id} did not complete within {timeout}s")


def download_result(history: dict, output_dir: str) -> list[str]:
    """Download generated images from ComfyUI output."""
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    outputs = history.get("outputs", {})
    for node_id, node_output in outputs.items():
        for img in node_output.get("images", []):
            filename = img["filename"]
            subfolder = img.get("subfolder", "")
            img_type = img.get("type", "output")
            params = urllib.parse.urlencode(
                {"filename": filename, "subfolder": subfolder, "type": img_type}
            )
            url = f"http://{SERVER}/view?{params}"
            dest = os.path.join(output_dir, filename)
            urllib.request.urlretrieve(url, dest)
            saved.append(dest)
            print(f"  Saved: {dest}")
    return saved


def main():
    global SERVER
    parser = argparse.ArgumentParser(description="Outfit change via Qwen Multi-Image Edit 2509")
    parser.add_argument("--person", default="девочка.JPG",
                        help="Person image filename (in ComfyUI/input/) or full path")
    parser.add_argument("--outfit", default="одежда девочка.png",
                        help="Outfit image filename (in ComfyUI/input/) or full path")
    parser.add_argument("--prompt", default=(
        "She is wearing the exact same outfit as shown in image 2. "
        "Keep the same pose, expression, and background as the original photo. "
        "Natural lighting."
    ))
    parser.add_argument("--negative", default="blurry, distorted, artifacts, wrong clothing, different person")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--server", default=SERVER, help="ComfyUI server address")
    args = parser.parse_args()

    SERVER = args.server
    seed = args.seed if args.seed is not None else random.randint(0, 2**53)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "input")

    # Resolve image paths
    person_path = args.person if os.path.isabs(args.person) else os.path.join(input_dir, args.person)
    outfit_path = args.outfit if os.path.isabs(args.outfit) else os.path.join(input_dir, args.outfit)

    if not os.path.isfile(person_path):
        print(f"ERROR: Person image not found: {person_path}")
        sys.exit(1)
    if not os.path.isfile(outfit_path):
        print(f"ERROR: Outfit image not found: {outfit_path}")
        sys.exit(1)

    output_dir = args.output if os.path.isabs(args.output) else os.path.join(script_dir, args.output)

    print(f"Person: {person_path}")
    print(f"Outfit: {outfit_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Negative: {args.negative}")
    print(f"Seed: {seed}")
    print()

    # Step 1: Upload images
    print("Step 1: Uploading images...")
    person_name = upload_image(person_path)
    outfit_name = upload_image(outfit_path)
    print()

    # Step 2: Build prompt
    print("Step 2: Building API prompt...")
    prompt = build_api_prompt(
        workflow_path=os.path.join(script_dir, "workflows", "Qwen_Image_Edit_2509_Multi_Editing.json"),
        person_file=person_name,
        outfit_file=outfit_name,
        positive=args.prompt,
        negative=args.negative,
        seed=seed,
    )
    print()

    # Step 3: Queue
    print("Step 3: Queuing prompt...")
    prompt_id, client_id = queue_prompt(prompt)
    print()

    # Step 4: Wait
    print("Step 4: Waiting for generation...")
    history = wait_for_completion(prompt_id)
    print()

    # Step 5: Report results (ComfyUI saves directly to its output/ folder)
    print("Step 5: Checking results...")
    outputs = history.get("outputs", {})
    for node_id, node_output in outputs.items():
        for img in node_output.get("images", []):
            filename = img["filename"]
            subfolder = img.get("subfolder", "")
            img_type = img.get("type", "output")
            # ComfyUI already saved this file to its output directory
            comfyui_path = os.path.join(script_dir, img_type, subfolder, filename)
            print(f"  Generated: {comfyui_path}")
            # Copy to our output dir if different
            if os.path.isfile(comfyui_path) and os.path.abspath(output_dir) != os.path.abspath(os.path.dirname(comfyui_path)):
                os.makedirs(output_dir, exist_ok=True)
                dest = os.path.join(output_dir, filename)
                shutil.copy2(comfyui_path, dest)
                print(f"  Copied to: {dest}")


if __name__ == "__main__":
    main()
