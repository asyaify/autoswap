"""
Full Photo Studio Pipeline
===========================
Input:
  - reference_image: Model child in desired pose + outfit + scene (pre-generated or provided)
  - child_photo: Real child's photo from the studio shoot
  - clothing_image: (optional) Specific outfit to transfer

Pipeline:
  Step 1: Qwen Image Edit — composite reference with clothing (if clothing differs)
  Step 2: ReActor Face Swap — put real child's face on the composite
  Step 3: GFPGAN — restore face quality after swap
  Step 4: Save final result

Usage:
  python3 pipeline.py --reference "референс.png" --child "DSC08443.JPG"
  python3 pipeline.py --reference "референс.png" --child "DSC08443.JPG" --clothing "одежда.png"
  python3 pipeline.py --batch  (processes all children against all references)
"""
import json
import urllib.request
import time
import sys
import argparse
import os

SERVER = "http://127.0.0.1:8188"


def submit_prompt(prompt_dict):
    """Submit workflow to ComfyUI and wait for result."""
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
                                fname = img_info["filename"]
                                return fname
                break
        except Exception:
            pass
        if i % 15 == 0 and i > 0:
            print(f"  Waiting... {i*2}s")
    return None


def faceswap_only(reference_image, child_photo, prefix="result"):
    """Simple face swap: reference image + child face."""
    print(f"\n[Face Swap] {reference_image} + face({child_photo})")
    prompt = {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": reference_image}
        },
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": child_photo}
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
    return submit_prompt(prompt)


def clothing_swap_then_face(reference_image, child_photo, clothing_image, prefix="result"):
    """
    Two-step pipeline:
    1. Qwen Image Edit: change clothing on reference image
    2. ReActor: swap face with real child
    """
    print(f"\n[Step 1] Clothing swap on {reference_image} with {clothing_image}")

    # Step 1: Change clothing using Qwen img2img
    step1_prompt = {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": "Qwen-Image-Edit-2509-Q5_K_M.gguf"}
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
            "inputs": {"vae_name": "qwen_image_vae.safetensors"}
        },
        "5": {
            "class_type": "LoadImage",
            "inputs": {"image": reference_image}
        },
        "6": {
            "class_type": "LoadImage",
            "inputs": {"image": clothing_image}
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
        "15": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["7", 0],
                "vae": ["4", 0]
            }
        },
        "9": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": "In <img1>, change only the clothing to match the outfit shown in <img2>. Keep the person's pose, body proportions, face, background, and composition completely unchanged. Only replace the clothing.",
                "clip": ["3", 0],
                "image1": ["7", 0],
                "image2": ["6", 0]
            }
        },
        "10": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "prompt": "different pose, changed composition, distorted, blurry",
                "clip": ["3", 0]
            }
        },
        "12": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "positive": ["9", 0],
                "negative": ["10", 0],
                "latent_image": ["15", 0],
                "seed": 42,
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 0.75
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
                "filename_prefix": f"{prefix}_step1_clothing"
            }
        }
    }

    clothing_result = submit_prompt(step1_prompt)
    if not clothing_result:
        print("  Step 1 failed!")
        return None

    print(f"  Step 1 done: {clothing_result}")

    # Step 2: Face swap
    print(f"\n[Step 2] Face swap with {child_photo}")
    step2_prompt = {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": clothing_result}
        },
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": child_photo}
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
                "filename_prefix": f"{prefix}_final"
            }
        }
    }

    final_result = submit_prompt(step2_prompt)
    if final_result:
        print(f"  Step 2 done: {final_result}")
    return final_result


def main():
    parser = argparse.ArgumentParser(description="Photo Studio AI Pipeline")
    parser.add_argument("--reference", type=str, help="Reference image (pose+outfit+background)")
    parser.add_argument("--child", type=str, help="Child's real photo")
    parser.add_argument("--clothing", type=str, default=None, help="Optional: clothing to transfer")
    parser.add_argument("--prefix", type=str, default="studio", help="Output filename prefix")
    parser.add_argument("--batch", action="store_true", help="Batch mode: all children x all references")

    args = parser.parse_args()

    if args.batch:
        # Batch mode: process all combinations
        references = ["референс.png", "референс_2.png"]
        children = ["DSC08443.JPG", "ребенок1.JPG", "ребенок2.JPG", "ребенок3.JPG"]

        print(f"Batch mode: {len(references)} references x {len(children)} children = {len(references)*len(children)} images")
        results = []
        for ri, ref in enumerate(references):
            for ci, child in enumerate(children):
                child_name = os.path.splitext(child)[0]
                prefix = f"batch_r{ri+1}_{child_name}"
                result = faceswap_only(ref, child, prefix=prefix)
                if result:
                    results.append(result)
                    print(f"  -> {result}")

        print(f"\nDone! Generated {len(results)} images.")
        return

    if not args.reference or not args.child:
        parser.print_help()
        sys.exit(1)

    if args.clothing:
        result = clothing_swap_then_face(
            args.reference, args.child, args.clothing, prefix=args.prefix
        )
    else:
        result = faceswap_only(
            args.reference, args.child, prefix=args.prefix
        )

    if result:
        print(f"\nFinal result: output/{result}")
    else:
        print("\nPipeline failed!")


if __name__ == "__main__":
    main()
