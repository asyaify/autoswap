#!/usr/bin/env python3
"""
Place a person onto a background using Qwen Multi-Image Edit 2509.
Image1 = person, Image2 = background.

Usage:
    python test_background_merge.py --person girl_safari.png --background bg_study.jpg \
        --prompt "..." --negative "..." --seed 42 --prefix study_scene
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


def upload_image(filepath: str) -> str:
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
        f"http://{SERVER}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    print(f"  Uploaded: {filename}")
    return result.get("name", filename)


def queue_prompt(prompt: dict) -> tuple:
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
    print(f"  Queued: {prompt_id}")
    return prompt_id, client_id


def wait_for_completion(prompt_id: str, timeout: int = 1200) -> dict:
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
        time.sleep(5)
    raise TimeoutError(f"Prompt {prompt_id} did not complete within {timeout}s")


def build_prompt(person_file: str, bg_file: str,
                 positive: str, negative: str, seed: int,
                 prefix: str = "bg_merge") -> dict:
    """
    Qwen Multi-Image Edit: person=image1, background=image2.
    Uses TextEncodeQwenImageEditPlus which accepts separate images.
    """
    return {
        # UnetLoaderGGUF
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
        # LoRA
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
        "78": {
            "class_type": "LoadImage",
            "inputs": {
                "image": person_file,
                "upload": "image"
            }
        },
        # LoadImage — background (image2)
        "106": {
            "class_type": "LoadImage",
            "inputs": {
                "image": bg_file,
                "upload": "image"
            }
        },
        # Scale person image
        "93": {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": ["78", 0],
                "upscale_method": "lanczos",
                "megapixels": 1.0,
                "resolution_steps": 1
            }
        },
        # VAEEncode person for latent
        "88": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["93", 0],
                "vae": ["39", 0]
            }
        },
        # Positive prompt
        "111": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": positive,
                "clip": ["38", 0],
                "vae": ["39", 0],
                "image1": ["93", 0],
                "image2": ["106", 0],
            }
        },
        # Negative prompt
        "110": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": negative,
                "clip": ["38", 0],
                "vae": ["39", 0],
                "image1": ["93", 0],
                "image2": ["106", 0],
            }
        },
        # KSampler
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["75", 0],
                "positive": ["111", 0],
                "negative": ["110", 0],
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
                "samples": ["3", 0],
                "vae": ["39", 0]
            }
        },
        # SaveImage
        "60": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["8", 0],
                "filename_prefix": prefix
            }
        },
    }


def main():
    global SERVER
    parser = argparse.ArgumentParser(description="Place person on background via Qwen Multi-Image Edit")
    parser.add_argument("--person", required=True, help="Person image filename in input/")
    parser.add_argument("--background", required=True, help="Background image filename in input/")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative", default="blurry, distorted, artifacts, different person, different face, different clothing")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prefix", default="bg_merge", help="Output filename prefix")
    parser.add_argument("--server", default=SERVER)
    args = parser.parse_args()

    SERVER = args.server
    seed = args.seed if args.seed is not None else random.randint(0, 2**53)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "input")

    person_path = args.person if os.path.isabs(args.person) else os.path.join(input_dir, args.person)
    bg_path = args.background if os.path.isabs(args.background) else os.path.join(input_dir, args.background)

    for p, label in [(person_path, "Person"), (bg_path, "Background")]:
        if not os.path.isfile(p):
            print(f"ERROR: {label} not found: {p}")
            sys.exit(1)

    print(f"Person:     {person_path}")
    print(f"Background: {bg_path}")
    print(f"Prompt:     {args.prompt[:80]}...")
    print(f"Seed:       {seed}")
    print()

    print("Step 1: Uploading...")
    person_name = upload_image(person_path)
    bg_name = upload_image(bg_path)
    print()

    print("Step 2: Building prompt...")
    prompt = build_prompt(person_name, bg_name, args.prompt, args.negative, seed, args.prefix)
    print()

    print("Step 3: Queuing...")
    prompt_id, _ = queue_prompt(prompt)
    print()

    print("Step 4: Waiting for generation (up to 20 min)...")
    history = wait_for_completion(prompt_id)
    print()

    print("Step 5: Results:")
    outputs = history.get("outputs", {})
    for nid, nout in outputs.items():
        for img in nout.get("images", []):
            comfyui_path = os.path.join(script_dir, img.get("type", "output"), img.get("subfolder", ""), img["filename"])
            print(f"  Generated: {comfyui_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
