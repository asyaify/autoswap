from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from autoswap.background import save_image
from autoswap.masking import parse_box, validate_sam_source


DEFAULT_NEGATIVE_PROMPT = (
    "adult, mature face, lingerie, underwear, exposed skin, cleavage, cropped top, "
    "swimsuit, low quality, blurry, deformed body, extra fingers, extra limbs"
)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Путь к исходному фото")
    parser.add_argument("--output-dir", required=True, help="Директория для результата")
    parser.add_argument(
        "--sam-model-id",
        default="facebook/sam2.1-hiera-large",
        help="Hugging Face model id для SAM2, например facebook/sam2.1-hiera-large",
    )
    parser.add_argument("--sam-checkpoint", help="Локальный checkpoint SAM2")
    parser.add_argument("--sam-config", help="Hydra config для локального SAM2")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Устройство запуска, обычно cuda или cpu",
    )
    parser.add_argument("--garment-scope", default="upper", choices=["upper", "lower", "full"])
    parser.add_argument("--cloth-box", help="Ручная рамка одежды x1,y1,x2,y2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SAM + inpainting + OpenPose pipeline для смены одежды, позы и фона"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    mask_parser = subparsers.add_parser("mask", help="Только построить subject mask и clothing mask")
    add_common_arguments(mask_parser)

    swap_parser = subparsers.add_parser("swap", help="Построить маски, перенести позу и сменить одежду")
    add_common_arguments(swap_parser)
    swap_parser.add_argument("--pose-ref", help="Путь к референсу позы")
    swap_parser.add_argument("--clothing-prompt", required=True, help="Описание новой одежды")
    swap_parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    swap_parser.add_argument("--steps", type=int, default=35)
    swap_parser.add_argument("--guidance-scale", type=float, default=8.0)
    swap_parser.add_argument("--conditioning-scale", type=float, default=0.95)
    swap_parser.add_argument("--seed", type=int)

    background_parser = subparsers.add_parser("background", help="Только заменить фон")
    add_common_arguments(background_parser)
    background_parser.add_argument("--background-prompt", help="Описание нового фона")
    background_parser.add_argument("--background-image", help="Готовое изображение фона")
    background_parser.add_argument("--steps", type=int, default=35)
    background_parser.add_argument("--seed", type=int)

    run_parser = subparsers.add_parser("run", help="Полный пайплайн за один запуск")
    add_common_arguments(run_parser)
    run_parser.add_argument("--pose-ref", help="Путь к референсу позы")
    run_parser.add_argument("--clothing-prompt", required=True, help="Описание новой одежды")
    run_parser.add_argument("--background-prompt", help="Описание нового фона")
    run_parser.add_argument("--background-image", help="Готовое изображение фона")
    run_parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    run_parser.add_argument("--steps", type=int, default=35)
    run_parser.add_argument("--guidance-scale", type=float, default=8.0)
    run_parser.add_argument("--conditioning-scale", type=float, default=0.95)
    run_parser.add_argument("--seed", type=int)
    return parser


def load_image(path: str | None) -> Image.Image | None:
    if not path:
        return None
    return Image.open(path).convert("RGB")


def main() -> None:
    args = build_parser().parse_args()
    validate_sam_source(args.sam_model_id, args.sam_checkpoint, args.sam_config)
    from autoswap.pipeline import AutoSwapConfig, AutoSwapPipeline

    output_dir = Path(args.output_dir)
    source_image = load_image(args.input)
    pose_image = load_image(getattr(args, "pose_ref", None))
    background_image = load_image(getattr(args, "background_image", None))
    clothing_box = parse_box(args.cloth_box)

    config = AutoSwapConfig(
        sam_model_id=args.sam_model_id,
        sam_checkpoint=args.sam_checkpoint,
        sam_config=args.sam_config,
        device=args.device,
        num_inference_steps=getattr(args, "steps", 35),
        guidance_scale=getattr(args, "guidance_scale", 8.0),
        conditioning_scale=getattr(args, "conditioning_scale", 0.95),
        seed=getattr(args, "seed", None),
    )

    pipeline = AutoSwapPipeline(config)
    if args.command == "mask":
        artifacts = pipeline.build_masks(
            image=source_image,
            garment_scope=args.garment_scope,
            clothing_box=clothing_box,
        )
        save_image(artifacts["subject_mask"], output_dir / "01_subject_mask.png")
        save_image(artifacts["clothing_mask"], output_dir / "02_clothing_mask.png")
        return

    if args.command == "swap":
        artifacts = pipeline.swap_clothing(
            image=source_image,
            clothing_prompt=args.clothing_prompt,
            pose_reference=pose_image,
            garment_scope=args.garment_scope,
            clothing_box=clothing_box,
            negative_prompt=args.negative_prompt,
        )
        save_image(artifacts["subject_mask"], output_dir / "01_subject_mask.png")
        save_image(artifacts["clothing_mask"], output_dir / "02_clothing_mask.png")
        save_image(artifacts["pose_condition"], output_dir / "03_pose_condition.png")
        save_image(artifacts["swapped"], output_dir / "04_swapped.png")
        return

    if args.command == "background":
        artifacts = pipeline.replace_background(
            image=source_image,
            background_prompt=args.background_prompt,
            background_image=background_image,
        )
        save_image(artifacts["subject_mask"], output_dir / "01_subject_mask.png")
        save_image(artifacts["final"], output_dir / "05_final.png")
        return

    pipeline.run(
        image=source_image,
        clothing_prompt=args.clothing_prompt,
        pose_reference=pose_image,
        garment_scope=args.garment_scope,
        output_dir=output_dir,
        background_prompt=args.background_prompt,
        background_image=background_image,
        clothing_box=clothing_box,
        negative_prompt=args.negative_prompt,
    )


if __name__ == "__main__":
    main()