"""
Qwen Multi-Image Edit — exact tutorial workflow
================================================
Follows the NextDiffusion tutorial exactly:
  ImageStitch → TextEncodeQwenImageEdit (with stitched image) →
  ModelSamplingAuraFlow → CFGNorm → KSampler → VAEDecode → SaveImage

Usage:
  python3 test_tutorial.py
"""
import json
import urllib.request
import time
import sys

SERVER = "http://127.0.0.1:8188"

# --- Images ---
IMAGE1 = "girl.png"       # Person
IMAGE2 = "clothes.png"    # Outfit reference

# --- Prompt (from the tutorial example 3: outfit change) ---
PROMPT = "она удивлена, у нее открыт рот, на ней точно такой же комплект нижнего белья, как на картинке. Вид сбоку, крупный план, белая стильная комната, сидит на стуле, вид сбоку, руки в волосах"
NEGATIVE = "blurry, distorted, artifacts, bad anatomy, deformed"

prompt = {
    # === Model Loading ===
    # UnetLoaderGGUF (we use GGUF since we don't have the fp8 safetensors)
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
    # LoraLoaderModelOnly — Lightning LoRA for 4-step inference
    "4": {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["1", 0],
            "lora_name": "Qwen-Image-Lightning-4steps-V1.0.safetensors",
            "strength_model": 1.0
        }
    },

    # === Image Loading & Stitching (per tutorial) ===
    # LoadImage 1 — person
    "10": {
        "class_type": "LoadImage",
        "inputs": {"image": IMAGE1}
    },
    # LoadImage 2 — outfit
    "11": {
        "class_type": "LoadImage",
        "inputs": {"image": IMAGE2}
    },
    # ImageStitch — combine side by side (direction: right)
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

    # === Prompt Encoding (per tutorial: TextEncodeQwenImageEdit with image) ===
    # Positive — with stitched image
    "20": {
        "class_type": "TextEncodeQwenImageEdit",
        "inputs": {
            "prompt": PROMPT,
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

    # === Model Flow (per tutorial) ===
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

    # === Latent & Sampling ===
    # EmptySD3LatentImage (1024x1024 as in tutorial)
    "40": {
        "class_type": "EmptySD3LatentImage",
        "inputs": {
            "width": 1024,
            "height": 1024,
            "batch_size": 1
        }
    },
    # KSampler
    "41": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["31", 0],
            "positive": ["20", 0],
            "negative": ["21", 0],
            "latent_image": ["40", 0],
            "seed": 916518398297193,
            "steps": 4,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "beta",
            "denoise": 1.0
        }
    },

    # === Decode & Save ===
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
            "filename_prefix": "tutorial_test"
        }
    }
}

# Submit
payload = json.dumps({"prompt": prompt}).encode("utf-8")
req = urllib.request.Request(
    f"{SERVER}/prompt",
    data=payload,
    headers={"Content-Type": "application/json"}
)

print(f"Submitting tutorial workflow...")
print(f"  Image1: {IMAGE1}")
print(f"  Image2: {IMAGE2}")
print(f"  Prompt: {PROMPT[:80]}...")
try:
    resp = urllib.request.urlopen(req)
    result = json.loads(resp.read())
    prompt_id = result.get("prompt_id")
    print(f"  Prompt ID: {prompt_id}")
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f"  ERROR {e.code}: {body[:2000]}")
    sys.exit(1)

# Wait for result
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
                            print(f"SUCCESS: output/{fname}")
                sys.exit(0)
    except Exception:
        pass
    if i % 15 == 0 and i > 0:
        print(f"  Waiting... {i*2}s")

print("TIMEOUT — generation took too long")
