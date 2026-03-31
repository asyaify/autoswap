"""
Safari photoshoot: girl.JPG + одежда.png + фон сафари.png
==========================================================
Two-step pipeline:
  Step 1: Change outfit (girl + clothing reference)
  Step 2: Place on safari background with pose change (result + background)

Uses working approach: ImageScaleToTotalPixels → ImageStitch → TextEncodeQwenImageEdit → KSampler

Usage:
  python3 test_safari.py
"""
import json
import urllib.request
import time
import sys
import os

SERVER = "http://127.0.0.1:8188"

# --- Images ---
IMAGE_GIRL = "girl.JPG"              # Child photo (large 6000x4000)
IMAGE_CLOTHES = "одежда сафари.png"   # Safari outfit reference
IMAGE_BG = "фон сафари.png"          # Safari background

# =====================================================
# STEP 1: Change outfit
# =====================================================
PROMPT_OUTFIT = (
    "Change the girl's clothing to match the safari outfit from the second image exactly. "
    "She must wear the exact same safari explorer costume: khaki vest, shorts, boots, and hat. "
    "Copy every detail of the outfit from the second image. "
    "Remove her current clothing completely. "
    "Keep her exact face, eyes, hair, and expression unchanged. "
    "Same grey studio background, same lighting."
)
NEGATIVE_OUTFIT = "original clothing, casual clothes, blurry, distorted, artifacts, deformed, bad anatomy"

def build_outfit_workflow():
    return {
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
        "10": {
            "class_type": "LoadImage",
            "inputs": {"image": IMAGE_GIRL}
        },
        "11": {
            "class_type": "LoadImage",
            "inputs": {"image": IMAGE_CLOTHES}
        },
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
        "20": {
            "class_type": "TextEncodeQwenImageEdit",
            "inputs": {
                "prompt": PROMPT_OUTFIT,
                "clip": ["2", 0],
                "vae": ["3", 0],
                "image": ["12", 0]
            }
        },
        "21": {
            "class_type": "TextEncodeQwenImageEdit",
            "inputs": {
                "prompt": NEGATIVE_OUTFIT,
                "clip": ["2", 0]
            }
        },
        "30": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["4", 0], "shift": 3.0}
        },
        "31": {
            "class_type": "CFGNorm",
            "inputs": {"model": ["30", 0], "strength": 1.0}
        },
        "40": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": 832, "height": 1248, "batch_size": 1}
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
        "50": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["41", 0], "vae": ["3", 0]}
        },
        "51": {
            "class_type": "SaveImage",
            "inputs": {"images": ["50", 0], "filename_prefix": "safari_step1_outfit"}
        }
    }


# =====================================================
# STEP 2: Place on background + change pose
# =====================================================
PROMPT_BG_POSE = (
    "Place this girl onto the safari background from the second image. "
    "She is standing in the African savanna, looking through binoculars with both hands. "
    "She is slightly turned to the right, gazing into the distance. "
    "Behind her is the safari landscape with golden grass, acacia trees, and a warm sunset sky. "
    "Keep her exact face, hair and safari outfit unchanged. "
    "Cinematic golden hour lighting, warm tones, professional photography."
)
NEGATIVE_BG_POSE = "grey background, studio background, indoor, blurry, distorted, artifacts, deformed"


def build_bg_pose_workflow(step1_image):
    return {
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
        # Step 1 result
        "10": {
            "class_type": "LoadImage",
            "inputs": {"image": step1_image}
        },
        # Safari background
        "11": {
            "class_type": "LoadImage",
            "inputs": {"image": IMAGE_BG}
        },
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
        "20": {
            "class_type": "TextEncodeQwenImageEdit",
            "inputs": {
                "prompt": PROMPT_BG_POSE,
                "clip": ["2", 0],
                "vae": ["3", 0],
                "image": ["12", 0]
            }
        },
        "21": {
            "class_type": "TextEncodeQwenImageEdit",
            "inputs": {
                "prompt": NEGATIVE_BG_POSE,
                "clip": ["2", 0]
            }
        },
        "30": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["4", 0], "shift": 3.0}
        },
        "31": {
            "class_type": "CFGNorm",
            "inputs": {"model": ["30", 0], "strength": 1.0}
        },
        "40": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": 832, "height": 1248, "batch_size": 1}
        },
        "41": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["31", 0],
                "positive": ["20", 0],
                "negative": ["21", 0],
                "latent_image": ["40", 0],
                "seed": 88,
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "beta",
                "denoise": 1.0
            }
        },
        "50": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["41", 0], "vae": ["3", 0]}
        },
        "51": {
            "class_type": "SaveImage",
            "inputs": {"images": ["50", 0], "filename_prefix": "safari_final"}
        }
    }


def submit_and_wait(workflow, label, timeout_s=1200):
    """Submit workflow to ComfyUI and wait for result."""
    payload = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    try:
        resp = urllib.request.urlopen(req)
        result = json.loads(resp.read())
        prompt_id = result.get("prompt_id")
        print(f"  Prompt ID: {prompt_id}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"  ERROR {e.code}: {body[:2000]}")
        return None

    print("  Generating...")
    for i in range(timeout_s // 2):
        time.sleep(2)
        try:
            hist_req = urllib.request.urlopen(f"{SERVER}/history/{prompt_id}")
            hist = json.loads(hist_req.read())
            if prompt_id in hist:
                status = hist[prompt_id].get("status", {})
                outputs = hist[prompt_id].get("outputs", {})
                if status.get("status_str") == "error":
                    print(f"  FAILED: {status.get('messages','')}")
                    return None
                if outputs:
                    for nid, out in outputs.items():
                        if "images" in out:
                            for img in out["images"]:
                                filename = img['filename']
                                print(f"  SUCCESS: output/{filename}")
                                return filename
        except Exception:
            pass
        if i % 15 == 0 and i > 0:
            print(f"    Waiting... {i*2}s")

    print("  TIMEOUT")
    return None


# =====================================================
# MAIN: Run pipeline
# =====================================================
if __name__ == "__main__":
    # Step 1: Outfit change
    print("STEP 1: Changing outfit to safari explorer costume...")
    step1_result = submit_and_wait(
        build_outfit_workflow(),
        "STEP 1: Outfit → Safari Explorer"
    )
    if not step1_result:
        print("Step 1 failed!")
        sys.exit(1)

    # Copy step 1 result from output/ to input/ so LoadImage can find it
    import shutil
    comfyui_dir = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(comfyui_dir, "output", step1_result)
    dst = os.path.join(comfyui_dir, "input", step1_result)
    shutil.copy2(src, dst)
    print(f"  Copied output/{step1_result} → input/{step1_result}")

    # Step 2: Background + pose
    print(f"\nSTEP 2: Placing on safari background with new pose...")
    print(f"  Using step 1 result: {step1_result}")
    step2_result = submit_and_wait(
        build_bg_pose_workflow(step1_result),
        "STEP 2: Background + Pose → Safari Scene"
    )
    if not step2_result:
        print("Step 2 failed!")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  DONE! Final result: output/{step2_result}")
    print(f"{'='*50}")
