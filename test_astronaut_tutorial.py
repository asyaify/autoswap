"""
Qwen Multi-Image Edit 2509 — Tutorial workflow (NextDiffusion)
==============================================================
Exact replication of the NextDiffusion tutorial:
  https://www.nextdiffusion.ai/tutorials/how-to-use-qwen-multi-image-editing-in-comfyui

Workflow:
  LoadImage x2 → ImageStitch → TextEncodeQwenImageEdit (positive, with stitched image+vae)
  TextEncodeQwenImageEdit (negative, text only)
  UnetLoader → LoRA → ModelSamplingAuraFlow(3) → CFGNorm(1)
  EmptySD3LatentImage(1024x1024) → KSampler(steps=4, cfg=1, euler, beta) → VAEDecode → SaveImage

Usage:
  python3 test_astronaut_tutorial.py
"""
import json
import urllib.request
import time
import sys

SERVER = "http://127.0.0.1:8188"

# ===== IMAGES =====
IMAGE1 = "child_girl.JPG"           # Image 1: the child (subject)
IMAGE2 = "astronaut_costume.png"    # Image 2: the astronaut costume reference

# ===== PROMPTS (following tutorial Example 3: Outfit/Costume Change) =====
POSITIVE = (
    "A 5-year-old girl from image 1, standing in a heroic pose, "
    "she is wearing the exact same astronaut costume as shown in image 2: "
    "white space suit with silver reflective panels, mission patches, "
    "JUNIOR ASTRONAUT name tag, silver zipper details, gray ribbed cuffs, "
    "silver moon boots, white gloves, and the red jet pack backpack. "
    "The helmet is removed, her face and hair are fully visible. "
    "White studio background, professional lighting, full body photo."
)
NEGATIVE = "blurry, distorted, artifacts, bad anatomy, deformed"

# ===== WORKFLOW (exact tutorial structure) =====
workflow = {
    # --- Load Models ---
    # UNETLoader (using GGUF since we have GGUF model, not fp8 safetensors)
    "1": {
        "class_type": "UnetLoaderGGUF",
        "inputs": {
            "unet_name": "Qwen-Image-Edit-2509-Q5_K_M.gguf"
        }
    },
    # CLIPLoader — text encoder
    "2": {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "type": "qwen_image"
        }
    },
    # VAELoader
    "3": {
        "class_type": "VAELoader",
        "inputs": {
            "vae_name": "qwen_image_vae.safetensors"
        }
    },

    # --- Load LoRA ---
    # LoraLoaderModelOnly: Qwen Lightning LoRA for 4-step inference
    "4": {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["1", 0],
            "lora_name": "Qwen-Image-Lightning-4steps-V1.0.safetensors",
            "strength_model": 1.0
        }
    },

    # --- Load Images ---
    "10": {
        "class_type": "LoadImage",
        "inputs": {"image": IMAGE1}
    },
    "11": {
        "class_type": "LoadImage",
        "inputs": {"image": IMAGE2}
    },

    # --- ImageStitch: combine into one image ---
    # Mode: right (image2 to the right of image1)
    # Match Image Size: true
    # Spacing Color: white, Spacing Width: 0
    "12": {
        "class_type": "ImageStitch",
        "inputs": {
            "image1": ["10", 0],
            "image2": ["11", 0],
            "direction": "right",
            "match_image_size": True,
            "spacing_width": 0,
            "spacing_color": "white"
        }
    },

    # --- Prompt: TextEncodeQwenImageEdit ---
    # Positive — with stitched image + VAE (as per tutorial)
    "20": {
        "class_type": "TextEncodeQwenImageEdit",
        "inputs": {
            "prompt": POSITIVE,
            "clip": ["2", 0],
            "vae": ["3", 0],
            "image": ["12", 0]
        }
    },
    # Negative — text only
    "21": {
        "class_type": "TextEncodeQwenImageEdit",
        "inputs": {
            "prompt": NEGATIVE,
            "clip": ["2", 0]
        }
    },

    # --- Model Flow ---
    # ModelSamplingAuraFlow (shift=3)
    "30": {
        "class_type": "ModelSamplingAuraFlow",
        "inputs": {
            "model": ["4", 0],
            "shift": 3.0
        }
    },
    # CFGNorm (strength=1)
    "31": {
        "class_type": "CFGNorm",
        "inputs": {
            "model": ["30", 0],
            "strength": 1.0
        }
    },

    # --- Resize / Latent ---
    # EmptySD3LatentImage (1024x1024, batch=1)
    "40": {
        "class_type": "EmptySD3LatentImage",
        "inputs": {
            "width": 1024,
            "height": 1024,
            "batch_size": 1
        }
    },

    # --- KSampler (Main Generation) ---
    # Steps: 4, CFG: 1, Sampler: euler, Scheduler: beta
    "41": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["31", 0],
            "positive": ["20", 0],
            "negative": ["21", 0],
            "latent_image": ["40", 0],
            "seed": 42,
            "steps": 4,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "beta",
            "denoise": 1.0
        }
    },

    # --- Decode & Save ---
    "50": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["41", 0],
            "vae": ["3", 0]
        }
    },
    "51": {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["50", 0],
            "filename_prefix": "astronaut_tutorial"
        }
    }
}

# ===== Submit & Wait =====
payload = json.dumps({"prompt": workflow}).encode("utf-8")
req = urllib.request.Request(
    f"{SERVER}/prompt",
    data=payload,
    headers={"Content-Type": "application/json"}
)

print("Qwen Multi-Image Edit — Astronaut Costume (Tutorial workflow)")
print(f"  Image 1 (child): {IMAGE1}")
print(f"  Image 2 (costume): {IMAGE2}")
print(f"  Prompt: {POSITIVE[:100]}...")
print()

try:
    resp = urllib.request.urlopen(req)
    result = json.loads(resp.read())
    prompt_id = result.get("prompt_id")
    print(f"Submitted! Prompt ID: {prompt_id}")
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f"ERROR {e.code}: {body[:2000]}")
    sys.exit(1)

print("Waiting for generation...")
for i in range(600):
    time.sleep(2)
    try:
        hist_req = urllib.request.urlopen(f"{SERVER}/history/{prompt_id}")
        hist = json.loads(hist_req.read())
        if prompt_id in hist:
            status = hist[prompt_id].get("status", {})
            outputs = hist[prompt_id].get("outputs", {})
            if status.get("status_str") == "error":
                msgs = status.get("messages", [])
                print(f"FAILED: {msgs}")
                sys.exit(1)
            if outputs:
                for node_id, out in outputs.items():
                    if "images" in out:
                        for img_info in out["images"]:
                            fname = img_info["filename"]
                            print(f"\nSUCCESS: output/{fname}")
                            sys.exit(0)
            break
    except Exception:
        pass
    if i % 15 == 0 and i > 0:
        print(f"  ...{i*2}s")

print("Timeout or no output")
