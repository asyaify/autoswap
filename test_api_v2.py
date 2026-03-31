import json
import urllib.request
import time
import sys

SERVER = "http://127.0.0.1:8188"

prompt = {
    "1": {
        "class_type": "UnetLoaderGGUF",
        "inputs": {
            "unet_name": "Qwen-Image-Edit-2509-Q5_K_M.gguf"
        }
    },
    "2": {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["1", 0],
            "lora_name": "Qwen-Image-Lightning-4steps-V1.0.safetensors",
            "strength_model": 1.0
        }
    },
    "3": {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "type": "qwen_image"
        }
    },
    "4": {
        "class_type": "VAELoader",
        "inputs": {
            "vae_name": "qwen_image_vae.safetensors"
        }
    },
    "5": {
        "class_type": "LoadImage",
        "inputs": {
            "image": "DSC08443.JPG"
        }
    },
    "6": {
        "class_type": "LoadImage",
        "inputs": {
            "image": "\u043e\u0434\u0435\u0436\u0434\u0430.png"
        }
    },
    "7": {
        "class_type": "ImageScaleToTotalPixels",
        "inputs": {
            "image": ["5", 0],
            "upscale_method": "lanczos",
            "megapixels": 1.0,
            "resolution_steps": 1
        }
    },
    "9": {
        "class_type": "TextEncodeQwenImageEditPlus",
        "inputs": {
            "prompt": "Make the child in image 1 wear the outfit from image 2. Keep the child face hair body proportions and background unchanged.",
            "clip": ["3", 0],
            "image1": ["7", 0],
            "image2": ["6", 0]
        }
    },
    "10": {
        "class_type": "TextEncodeQwenImageEditPlus",
        "inputs": {
            "prompt": "",
            "clip": ["3", 0]
        }
    },
    "11": {
        "class_type": "EmptySD3LatentImage",
        "inputs": {
            "width": 832,
            "height": 1248,
            "batch_size": 1
        }
    },
    "12": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["2", 0],
            "positive": ["9", 0],
            "negative": ["10", 0],
            "latent_image": ["11", 0],
            "seed": 42,
            "steps": 4,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0
        }
    },
    "13": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["12", 0],
            "vae": ["4", 0]
        }
    },
    "14": {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["13", 0],
            "filename_prefix": "test_clean"
        }
    }
}

payload = json.dumps({"prompt": prompt}).encode("utf-8")
req = urllib.request.Request(
    f"{SERVER}/prompt",
    data=payload,
    headers={"Content-Type": "application/json"}
)

print("Submitting workflow...")
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
                            subfolder = img_info.get("subfolder", "")
                            ftype = img_info.get("type", "output")
                            print(f"Output image: {ftype}/{subfolder}/{fname}")
            break
    except Exception:
        pass
    if i % 15 == 0 and i > 0:
        print(f"  Still waiting... {i*2}s elapsed")
else:
    print("Timed out")
