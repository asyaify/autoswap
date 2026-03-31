"""
Astronaut costume — two-step pipeline
======================================
Step 1: Qwen generates child in astronaut costume (ImageStitch approach)
Step 2: ReActor face swap inserts real child's face

Usage:
  python3 test_astronaut_v2.py
"""
import json
import urllib.request
import time
import sys

SERVER = "http://127.0.0.1:8188"

IMAGE_CHILD = "child_girl.JPG"
IMAGE_COSTUME = "astronaut_costume.png"

PROMPT = (
    "Generate a full-body photo of the child from image 1 wearing the exact astronaut costume "
    "from image 2. The child should be standing in a heroic pose wearing the complete white space suit "
    "with all details from image 2: silver reflective panels, rocket and star mission patches, "
    "JUNIOR ASTRONAUT name tag, silver zipper details, ribbed gray cuffs, white astronaut helmet "
    "with gold visor flipped up, silver moon boots, white padded gloves, and the red jet pack backpack. "
    "Preserve the child's face, hair, and identity exactly from image 1. "
    "White studio background, clean professional lighting, 8K, realistic."
)
NEGATIVE = "blurry, distorted, artifacts, bad anatomy, deformed, extra limbs, missing limbs, disfigured face"


def submit_prompt(prompt_dict):
    payload = json.dumps({"prompt": prompt_dict}).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        resp = urllib.request.urlopen(req)
        result = json.loads(resp.read())
        prompt_id = result.get("prompt_id")
        print(f"  Prompt ID: {prompt_id}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"  ERROR {e.code}: {body[:2000]}")
        return None

    print("  Waiting for generation...")
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
                    print(f"  FAILED: {msgs}")
                    return None
                if outputs:
                    for node_id, out in outputs.items():
                        if "images" in out:
                            for img_info in out["images"]:
                                return img_info["filename"]
                break
        except Exception:
            pass
        if i % 15 == 0 and i > 0:
            print(f"    ...{i*2}s")
    return None


# ============================================================
# STEP 1: Qwen Image Edit — generate child in astronaut costume
# ============================================================
print("=" * 60)
print("STEP 1: Qwen Image Edit — astronaut costume generation")
print("=" * 60)

step1_prompt = {
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
        "inputs": {"image": IMAGE_CHILD}
    },
    "11": {
        "class_type": "LoadImage",
        "inputs": {"image": IMAGE_COSTUME}
    },
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
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1}
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
        "inputs": {
            "images": ["50", 0],
            "filename_prefix": "astronaut_step1"
        }
    }
}

step1_result = submit_prompt(step1_prompt)
if not step1_result:
    print("Step 1 FAILED!")
    sys.exit(1)
print(f"  Step 1 done: {step1_result}")

# ============================================================
# STEP 2: ReActor Face Swap — put real child's face
# ============================================================
print()
print("=" * 60)
print("STEP 2: ReActor Face Swap — inserting child's real face")
print("=" * 60)

step2_prompt = {
    "1": {
        "class_type": "LoadImage",
        "inputs": {"image": step1_result}
    },
    "2": {
        "class_type": "LoadImage",
        "inputs": {"image": IMAGE_CHILD}
    },
    "3": {
        "class_type": "ReActorFaceSwap",
        "inputs": {
            "enabled": True,
            "input_image": ["1", 0],
            "source_image": ["2", 0],
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
    "4": {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["3", 0],
            "filename_prefix": "astronaut_final"
        }
    }
}

step2_result = submit_prompt(step2_prompt)
if not step2_result:
    print("Step 2 FAILED (face swap) — the costume image has no detectable face.")
    print(f"Costume-only result available: output/{step1_result}")
    sys.exit(1)

print(f"\nFINAL RESULT: output/{step2_result}")
print("Done!")
