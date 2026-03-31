"""
Qwen Multi-Image Edit — Astronaut costume on child
===================================================
Takes child photo (image 1) + astronaut costume (image 2)
and generates a photo of the child wearing the astronaut costume.

Usage:
  python3 test_astronaut.py
"""
import json
import urllib.request
import time
import sys

SERVER = "http://127.0.0.1:8188"

# --- Images ---
IMAGE1 = "girl.JPG"                   # Child photo
IMAGE2 = "одежда астронавта.png"      # Astronaut costume reference

# --- Prompt ---
PROMPT = (
    "Change the girl's clothing to match the astronaut costume from the second image. "
    "She must wear the white Junior Astronaut spacesuit with silver trim, zipper front, "
    "rocket and star mission patches on the chest, silver boots, white gloves, "
    "and the red jetpack with rockets on her back. "
    "Put the white astronaut helmet with gold visor on her head. "
    "Remove her current clothing completely and replace with the full astronaut costume. "
    "Keep her exact face, eyes, and expression unchanged. "
    "Same background, same lighting, same standing pose."
)
NEGATIVE = "original clothing, casual clothes, no helmet, blurry, distorted, artifacts, deformed, bad anatomy"

prompt = {
    # === Model Loading ===
    "1": {
        "class_type": "UnetLoaderGGUF",
        "inputs": {
            "unet_name": "Qwen-Image-Edit-2509-Q5_K_M.gguf"
        }
    },
    "2": {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "type": "qwen_image"
        }
    },
    "3": {
        "class_type": "VAELoader",
        "inputs": {
            "vae_name": "qwen_image_vae.safetensors"
        }
    },
    "4": {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["1", 0],
            "lora_name": "Qwen-Image-Lightning-4steps-V1.0.safetensors",
            "strength_model": 1.0
        }
    },

    # === Image Loading ===
    "10": {
        "class_type": "LoadImage",
        "inputs": {"image": IMAGE1}
    },
    "11": {
        "class_type": "LoadImage",
        "inputs": {"image": IMAGE2}
    },
    # Scale both to 0.5 megapixels (critical for large photos like 6000x4000)
    "13": {
        "class_type": "ImageScaleToTotalPixels",
        "inputs": {
            "image": ["10", 0],
            "upscale_method": "lanczos",
            "megapixels": 0.5,
            "resolution_steps": 1
        }
    },
    "14": {
        "class_type": "ImageScaleToTotalPixels",
        "inputs": {
            "image": ["11", 0],
            "upscale_method": "lanczos",
            "megapixels": 0.5,
            "resolution_steps": 1
        }
    },
    # Stitch side by side
    "12": {
        "class_type": "ImageStitch",
        "inputs": {
            "image1": ["13", 0],
            "image2": ["14", 0],
            "direction": "right",
            "match_image_size": True,
            "spacing_width": 0,
            "spacing_color": "white"
        }
    },

    # === Prompt Encoding ===
    "20": {
        "class_type": "TextEncodeQwenImageEdit",
        "inputs": {
            "prompt": PROMPT,
            "clip": ["2", 0],
            "vae": ["3", 0],
            "image": ["12", 0]
        }
    },
    "21": {
        "class_type": "TextEncodeQwenImageEdit",
        "inputs": {
            "prompt": NEGATIVE,
            "clip": ["2", 0]
        }
    },

    # === Model Flow ===
    "30": {
        "class_type": "ModelSamplingAuraFlow",
        "inputs": {
            "model": ["4", 0],
            "shift": 3.0
        }
    },
    "31": {
        "class_type": "CFGNorm",
        "inputs": {
            "model": ["30", 0],
            "strength": 1.0
        }
    },

    # === Latent & Sampling — portrait 832x1248 ===
    "40": {
        "class_type": "EmptySD3LatentImage",
        "inputs": {
            "width": 832,
            "height": 1248,
            "batch_size": 1
        }
    },
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
            "filename_prefix": "astronaut_child"
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

print(f"Submitting astronaut costume workflow...")
print(f"  Child: {IMAGE1}")
print(f"  Costume: {IMAGE2}")
print(f"  Prompt: {PROMPT[:100]}...")
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
print("Waiting for generation (may take several minutes on MPS)...")
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
        elapsed = i * 2
        print(f"  Still waiting... {elapsed}s")

print("Timeout or no output received")
