from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from autoswap.background import composite_foreground, mask_to_image, save_image
from autoswap.masking import ClothingBoxEstimator, SamMasker


DEFAULT_NEGATIVE_PROMPT = (
    "adult, mature face, lingerie, underwear, exposed skin, cleavage, cropped top, "
    "swimsuit, low quality, blurry, deformed body, extra fingers, extra limbs"
)


@dataclass(slots=True)
class AutoSwapConfig:
    sam_model_id: str | None = "facebook/sam2.1-hiera-large"
    sam_checkpoint: str | None = None
    sam_config: str | None = None
    device: str = "cuda"
    inpaint_model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    controlnet_model_id: str = "thibaud/controlnet-openpose-sdxl-1.0"
    background_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    openpose_model_id: str = "lllyasviel/Annotators"
    num_inference_steps: int = 35
    guidance_scale: float = 8.0
    conditioning_scale: float = 0.95
    seed: int | None = None


class AutoSwapPipeline:
    def __init__(self, config: AutoSwapConfig) -> None:
        self.config = config
        self.dtype = torch.float16 if config.device.startswith("cuda") else torch.float32
        self.masker = SamMasker(
            device=config.device,
            model_id=config.sam_model_id,
            checkpoint=config.sam_checkpoint,
            config_file=config.sam_config,
        )
        self.box_estimator = ClothingBoxEstimator()
        self.pose_detector: Any | None = None
        self.controlnet: Any | None = None
        self.inpaint_pipe: Any | None = None
        self.background_pipe: Any | None = None

    def _load_swap_models(self) -> None:
        from controlnet_aux import OpenposeDetector
        from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline

        if self.pose_detector is None:
            self.pose_detector = OpenposeDetector.from_pretrained(self.config.openpose_model_id)
        if self.controlnet is None:
            self.controlnet = ControlNetModel.from_pretrained(
                self.config.controlnet_model_id,
                torch_dtype=self.dtype,
            )
        if self.inpaint_pipe is None:
            self.inpaint_pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                self.config.inpaint_model_id,
                controlnet=self.controlnet,
                torch_dtype=self.dtype,
                variant="fp16" if self.dtype == torch.float16 else None,
            )
            self.inpaint_pipe.to(self.config.device)

    def _load_background_model(self) -> None:
        from diffusers import StableDiffusionXLPipeline

        if self.background_pipe is None:
            self.background_pipe = StableDiffusionXLPipeline.from_pretrained(
                self.config.background_model_id,
                torch_dtype=self.dtype,
                variant="fp16" if self.dtype == torch.float16 else None,
            )
            self.background_pipe.to(self.config.device)

    def _generator(self) -> torch.Generator | None:
        if self.config.seed is None:
            return None
        return torch.Generator(device=self.config.device).manual_seed(self.config.seed)

    def _pose_condition(self, pose_reference: Image.Image, output_size: tuple[int, int]) -> Image.Image:
        self._load_swap_models()
        assert self.pose_detector is not None
        conditioning = self.pose_detector(pose_reference.convert("RGB"), hand_and_face=True)
        return conditioning.resize(output_size, Image.Resampling.BILINEAR)

    def _background_image(
        self,
        size: tuple[int, int],
        background_prompt: str | None,
        background_image: Image.Image | None,
    ) -> Image.Image | None:
        if background_image is not None:
            return background_image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        if not background_prompt:
            return None

        self._load_background_model()
        assert self.background_pipe is not None
        prompt = f"clean background plate, no people, {background_prompt}"
        result = self.background_pipe(
            prompt=prompt,
            negative_prompt="people, child, portrait, person, close-up, blurry",
            num_inference_steps=max(20, self.config.num_inference_steps - 10),
            guidance_scale=7.0,
            generator=self._generator(),
            height=size[1],
            width=size[0],
        )
        return result.images[0]

    def build_masks(
        self,
        image: Image.Image,
        garment_scope: str,
        clothing_box: np.ndarray | None = None,
    ) -> dict[str, Image.Image | np.ndarray]:
        source = image.convert("RGB")
        subject_mask, subject_box = self.masker.subject_mask(source)
        if clothing_box is None:
            clothing_box = self.box_estimator.estimate(source, garment_scope, subject_box)
        clothing_mask = self.masker.mask_from_box(source, clothing_box)
        clothing_mask = np.logical_and(clothing_mask, subject_mask)
        return {
            "subject_mask": mask_to_image(subject_mask),
            "clothing_mask": mask_to_image(clothing_mask),
            "subject_box": subject_box,
            "clothing_box": clothing_box,
        }

    def swap_clothing(
        self,
        image: Image.Image,
        clothing_prompt: str,
        pose_reference: Image.Image | None,
        garment_scope: str,
        clothing_box: np.ndarray | None = None,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    ) -> dict[str, Image.Image]:
        source = image.convert("RGB")
        pose_reference = pose_reference.convert("RGB") if pose_reference else source

        masks = self.build_masks(source, garment_scope, clothing_box=clothing_box)
        subject_mask = np.array(masks["subject_mask"].convert("L")) > 0
        clothing_mask = np.array(masks["clothing_mask"].convert("L")) > 0

        pose_condition = self._pose_condition(pose_reference, source.size)
        self._load_swap_models()
        assert self.inpaint_pipe is not None

        prompt = (
            "realistic photo of the same child, preserve identity, age-appropriate styling, "
            f"{clothing_prompt}, natural proportions, realistic fabric folds"
        )

        generated = self.inpaint_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=source,
            mask_image=mask_to_image(clothing_mask),
            control_image=pose_condition,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            generator=self._generator(),
            strength=0.99,
            controlnet_conditioning_scale=self.config.conditioning_scale,
        ).images[0]

        return {
            "subject_mask": mask_to_image(subject_mask),
            "clothing_mask": mask_to_image(clothing_mask),
            "pose_condition": pose_condition,
            "swapped": generated,
        }

    def replace_background(
        self,
        image: Image.Image,
        background_prompt: str | None = None,
        background_image: Image.Image | None = None,
    ) -> dict[str, Image.Image]:
        source = image.convert("RGB")
        subject_mask, _ = self.masker.subject_mask(source)
        bg = self._background_image(source.size, background_prompt, background_image)
        final_image = (
            composite_foreground(source, mask_to_image(subject_mask), bg)
            if bg is not None
            else source
        )
        return {
            "subject_mask": mask_to_image(subject_mask),
            "final": final_image,
        }

    def run(
        self,
        image: Image.Image,
        clothing_prompt: str,
        pose_reference: Image.Image | None,
        garment_scope: str,
        output_dir: Path,
        background_prompt: str | None = None,
        background_image: Image.Image | None = None,
        clothing_box: np.ndarray | None = None,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    ) -> dict[str, Image.Image]:
        artifacts = self.swap_clothing(
            image=image,
            clothing_prompt=clothing_prompt,
            pose_reference=pose_reference,
            garment_scope=garment_scope,
            clothing_box=clothing_box,
            negative_prompt=negative_prompt,
        )
        background_result = self.replace_background(
            image=artifacts["swapped"],
            background_prompt=background_prompt,
            background_image=background_image,
        )
        artifacts["final"] = background_result["final"]

        save_image(artifacts["subject_mask"], output_dir / "01_subject_mask.png")
        save_image(artifacts["clothing_mask"], output_dir / "02_clothing_mask.png")
        save_image(artifacts["pose_condition"], output_dir / "03_pose_condition.png")
        save_image(artifacts["swapped"], output_dir / "04_swapped.png")
        save_image(artifacts["final"], output_dir / "05_final.png")
        return artifacts