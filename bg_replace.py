"""
Background replacement via Qwen Image Edit 2509
================================================
Uses ComfyUI API to send child photo + background to Qwen model
and asks it to place the child onto the background.

Usage:
  python3 bg_replace.py --child DSC08443.JPG --bg фон_1.jpg
  python3 bg_replace.py --child DSC08443.JPG --bg фон_1.jpg --seed 123
  python3 bg_replace.py --all
"""
import json
import urllib.request
import time
import sys
import argparse
import os
import shutil

SERVER = "http://127.0.0.1:8188"
COMFYUI_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(COMFYUI_DIR, "input")
OUTPUT_DIR = os.path.join(COMFYUI_DIR, "output")


def submit_and_wait(prompt_dict, timeout_s=600):
    """Submit workflow to ComfyUI, wait for result, return output filename."""
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
        print(f"  ERROR {e.code}: {body[:1000]}")
        return None

    for i in range(timeout_s // 2):
        time.sleep(2)
        try:
            hist_req = urllib.request.urlopen(f"{SERVER}/history/{prompt_id}")
            hist = json.loads(hist_req.read())
            if prompt_id in hist:
                status = hist[prompt_id].get("status", {})
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [])
                    print(f"  FAILED: {msgs}")
                    return None
                outputs = hist[prompt_id].get("outputs", {})
                if outputs:
                    for node_id, out in outputs.items():
                        if "images" in out:
                            for img_info in out["images"]:
                                return img_info["filename"]
                    break
        except Exception:
            pass
        if i % 15 == 0 and i > 0:
            print(f"  Waiting... {i*2}s")
    return None


def build_bg_replace_workflow(child_image, bg_image, prefix="bg_result", seed=42):
    """Build ComfyUI workflow: Qwen Image Edit to place child onto background."""
    return {
        # Load Qwen GGUF model
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {
                "unet_name": "Qwen-Image-Edit-2509-Q5_K_M.gguf"
            }
        },
        # LoRA for 4-step fast inference
        "2": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": "Qwen-Image-Lightning-4steps-V1.0.safetensors",
                "strength_model": 1.0
            }
        },
        # Text encoder
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "type": "qwen_image"
            }
        },
        # VAE
        "4": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "qwen_image_vae.safetensors"
            }
        },
        # Child photo (img1)
        "5": {
            "class_type": "LoadImage",
            "inputs": {"image": child_image}
        },
        # Background (img2)
        "6": {
            "class_type": "LoadImage",
            "inputs": {"image": bg_image}
        },
        # Scale child photo
        "7": {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": ["5", 0],
                "upscale_method": "lanczos",
                "megapixels": 1.0,
                "resolution_steps": 1
            }
        },
        # Scale background
        "8": {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": ["6", 0],
                "upscale_method": "lanczos",
                "megapixels": 1.0,
                "resolution_steps": 1
            }
        },
        # Positive prompt: place child from img1 onto background from img2
        "9": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": "Take the child from <img1> and place them naturally into the scene shown in <img2>. "
                          "Keep the child's face, hair, body, clothing, and proportions exactly the same. "
                          "The child should look like they are standing in the scene from <img2>. "
                          "Match the lighting and color tone of <img2>. "
                          "The final image should look like a professional studio photograph.",
                "clip": ["3", 0],
                "image1": ["7", 0],
                "image2": ["8", 0]
            }
        },
        # Negative prompt
        "10": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": "blurry, distorted face, changed clothing, wrong proportions, artifacts, collage look, visible seams",
                "clip": ["3", 0]
            }
        },
        # Empty latent (output resolution — portrait 832x1248)
        "11": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {
                "width": 832,
                "height": 1248,
                "batch_size": 1
            }
        },
        # KSampler
        "12": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "positive": ["9", 0],
                "negative": ["10", 0],
                "latent_image": ["11", 0],
                "seed": seed,
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0
            }
        },
        # VAE Decode
        "13": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["12", 0],
                "vae": ["4", 0]
            }
        },
        # Save
        "14": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["13", 0],
                "filename_prefix": prefix
            }
        }
    }


def build_faceswap_workflow(target_image, source_face_image, prefix="faceswap"):
    """Build ReActor face swap workflow: put source face onto target image."""
    return {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": target_image}
        },
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": source_face_image}
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
                "filename_prefix": prefix
            }
        }
    }


def process_one(child, bg, seed=42, skip_faceswap=False):
    prefix = f"{os.path.splitext(child)[0]}_on_{os.path.splitext(bg)[0]}"
    print(f"\n[{prefix}]  child={child}  bg={bg}  seed={seed}")

    # Step 1: Qwen Image Edit — place child on background
    print("  Step 1: Qwen background replacement...")
    workflow = build_bg_replace_workflow(child, bg, prefix=f"{prefix}_step1", seed=seed)
    step1_result = submit_and_wait(workflow)
    if not step1_result:
        print(f"  Step 1 FAILED")
        return None
    print(f"  Step 1 done: output/{step1_result}")

    if skip_faceswap:
        return step1_result

    # Copy step 1 output to input/ so LoadImage can find it
    src = os.path.join(OUTPUT_DIR, step1_result)
    dst = os.path.join(INPUT_DIR, step1_result)
    shutil.copy2(src, dst)

    # Step 2: ReActor face swap — restore real child's face
    print("  Step 2: Face swap to restore identity...")
    faceswap_wf = build_faceswap_workflow(step1_result, child, prefix=f"{prefix}_final")
    final_result = submit_and_wait(faceswap_wf)
    if final_result:
        print(f"  Step 2 done: output/{final_result}")
    else:
        print(f"  Step 2 FAILED, returning step 1 result")
        return step1_result
    return final_result


def main():
    parser = argparse.ArgumentParser(description="Background replacement via Qwen Image Edit")
    parser.add_argument("--child", type=str, help="Child photo filename (in ComfyUI/input/)")
    parser.add_argument("--bg", type=str, help="Background filename (in ComfyUI/input/)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-faceswap", action="store_true", help="Skip ReActor face swap step")
    parser.add_argument("--all", action="store_true", help="All children x all backgrounds")
    args = parser.parse_args()

    if args.all:
        input_dir = os.path.join(os.path.dirname(__file__), "input")
        children = [f for f in os.listdir(input_dir)
                     if f.lower().startswith(("dsc", "ребенок")) and f.lower().endswith((".jpg", ".jpeg", ".png"))]
        backgrounds = [f for f in os.listdir(input_dir)
                       if f.startswith("фон") and f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"Batch: {len(children)} children x {len(backgrounds)} backgrounds")
        for child in sorted(children):
            for bg in sorted(backgrounds):
                process_one(child, bg, seed=args.seed, skip_faceswap=args.no_faceswap)
        return

    if not args.child or not args.bg:
        parser.print_help()
        sys.exit(1)

    process_one(args.child, args.bg, seed=args.seed, skip_faceswap=args.no_faceswap)


if __name__ == "__main__":
    main()
