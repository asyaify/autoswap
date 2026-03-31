"""
Outfit swap: DSC08443 girl + одежда.png (mustard sweater + burgundy beanie)
Uses exact tutorial workflow: ImageStitch → TextEncodeQwenImageEdit → ModelSamplingAuraFlow → CFGNorm → KSampler
"""
import json
import urllib.request
import time
import sys

SERVER = "http://127.0.0.1:8188"

IMAGE1 = "DSC08443.JPG"   # Girl photo
IMAGE2 = "одежда.png"     # Outfit: mustard knit sweater + burgundy beanie

PROMPT = (
    "Change the girl's clothing to match the outfit from the second image. "
    "She must wear the mustard yellow chunky knit sweater and the dark burgundy ribbed beanie hat. "
    "Remove her current pink t-shirt and polka dot pants completely and replace with the sweater and dark jeans. "
    "Keep her exact face, brown eyes, pigtails with pink ties, earrings, and pose unchanged. "
    "Same grey studio background, same lighting."
)
NEGATIVE = "pink shirt, bow pattern, polka dots, original clothing, blurry, distorted, artifacts, deformed"

prompt = {
    "1": {
        "class_type": "UnetLoaderGGUF",
        "inputs": {"unet_name": "Qwen-Image-Edit-2509-Q5_K_M.gguf"}
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
        "inputs": {"vae_name": "qwen_image_vae.safetensors"}
    },
    "4": {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["1", 0],
            "lora_name": "Qwen-Image-Lightning-4steps-V1.0.safetensors",
            "strength_model": 1.0
        }
    },
    # Load images
    "10": {
        "class_type": "LoadImage",
        "inputs": {"image": IMAGE1}
    },
    "11": {
        "class_type": "LoadImage",
        "inputs": {"image": IMAGE2}
    },
    # Scale both images to same megapixels before stitching
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
    # Positive prompt with stitched image
    "20": {
        "class_type": "TextEncodeQwenImageEdit",
        "inputs": {
            "prompt": PROMPT,
            "clip": ["2", 0],
            "vae": ["3", 0],
            "image": ["12", 0]
        }
    },
    # Negative
    "21": {
        "class_type": "TextEncodeQwenImageEdit",
        "inputs": {
            "prompt": NEGATIVE,
            "clip": ["2", 0]
        }
    },
    # Model flow
    "30": {
        "class_type": "ModelSamplingAuraFlow",
        "inputs": {"model": ["4", 0], "shift": 3.0}
    },
    "31": {
        "class_type": "CFGNorm",
        "inputs": {"model": ["30", 0], "strength": 1.0}
    },
    # Latent — portrait 832x1248
    "40": {
        "class_type": "EmptySD3LatentImage",
        "inputs": {"width": 832, "height": 1248, "batch_size": 1}
    },
    # KSampler
    "41": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["31", 0],
            "positive": ["20", 0],
            "negative": ["21", 0],
            "latent_image": ["40", 0],
            "seed": 12345,
            "steps": 4,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "beta",
            "denoise": 1.0
        }
    },
    # Decode & Save
    "50": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["41", 0], "vae": ["3", 0]}
    },
    "51": {
        "class_type": "SaveImage",
        "inputs": {"images": ["50", 0], "filename_prefix": "girl_outfit_swap"}
    }
}

payload = json.dumps({"prompt": prompt}).encode("utf-8")
req = urllib.request.Request(f"{SERVER}/prompt", data=payload, headers={"Content-Type": "application/json"})

print(f"Submitting: {IMAGE1} + {IMAGE2}")
try:
    resp = urllib.request.urlopen(req)
    result = json.loads(resp.read())
    prompt_id = result.get("prompt_id")
    print(f"Prompt ID: {prompt_id}")
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f"ERROR {e.code}: {body[:2000]}")
    sys.exit(1)

print("Generating...")
for i in range(600):
    time.sleep(2)
    try:
        hist_req = urllib.request.urlopen(f"{SERVER}/history/{prompt_id}")
        hist = json.loads(hist_req.read())
        if prompt_id in hist:
            status = hist[prompt_id].get("status", {})
            outputs = hist[prompt_id].get("outputs", {})
            if status.get("status_str") == "error":
                print(f"FAILED: {status.get('messages','')}")
                sys.exit(1)
            if outputs:
                for nid, out in outputs.items():
                    if "images" in out:
                        for img in out["images"]:
                            print(f"SUCCESS: output/{img['filename']}")
                sys.exit(0)
    except Exception:
        pass
    if i % 15 == 0 and i > 0:
        print(f"  Waiting... {i*2}s")

print("TIMEOUT")
