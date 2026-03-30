from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2, build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor


@dataclass(slots=True)
class DetectedBoxes:
    subject_box: np.ndarray
    clothing_box: np.ndarray


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def refine_binary_mask(mask: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    refined = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
    return refined.astype(bool)


def _score_mask(mask: dict, image_shape: tuple[int, int, int]) -> float:
    segmentation = mask["segmentation"]
    area = float(mask["area"])
    ys, xs = np.nonzero(segmentation)
    if len(xs) == 0:
        return -1.0
    height, width = image_shape[:2]
    center_x = xs.mean() / width
    center_y = ys.mean() / height
    centrality = 1.0 - ((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2)
    return area * max(centrality, 0.1)


def _mask_box(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def _expand_box(box: np.ndarray, width: int, height: int, pad: float) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.float32)
    box_w = x2 - x1
    box_h = y2 - y1
    x1 = max(0.0, x1 - box_w * pad)
    y1 = max(0.0, y1 - box_h * pad)
    x2 = min(float(width - 1), x2 + box_w * pad)
    y2 = min(float(height - 1), y2 + box_h * pad)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


class SamMasker:
    def __init__(
        self,
        device: str,
        model_id: str | None = None,
        checkpoint: str | None = None,
        config_file: str | None = None,
    ) -> None:
        if model_id:
            sam_model = build_sam2_hf(model_id, device=device)
        elif checkpoint and config_file:
            sam_model = build_sam2(config_file=config_file, ckpt_path=checkpoint, device=device)
        else:
            raise ValueError("Для SAM2 нужен либо --sam-model-id, либо пара --sam-config + --sam-checkpoint")

        self.predictor = SAM2ImagePredictor(sam_model)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            sam_model,
            points_per_side=24,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
        )

    def subject_mask(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        image_bgr = pil_to_bgr(image)
        masks = self.mask_generator.generate(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not masks:
            raise RuntimeError("SAM не нашел ни одной маски на изображении")

        best = max(masks, key=lambda item: _score_mask(item, image_bgr.shape))
        mask = refine_binary_mask(best["segmentation"])
        return mask, _expand_box(_mask_box(mask), image.width, image.height, pad=0.04)

    def mask_from_box(self, image: Image.Image, box: np.ndarray) -> np.ndarray:
        image_rgb = np.array(image.convert("RGB"))
        self.predictor.set_image(image_rgb)
        masks, scores, _ = self.predictor.predict(box=box[None, :], multimask_output=True)
        best_index = int(np.argmax(scores))
        return refine_binary_mask(masks[best_index])


class ClothingBoxEstimator:
    def estimate(self, image: Image.Image, garment_scope: str, subject_box: np.ndarray) -> np.ndarray:
        width, height = image.size
        x1, y1, x2, y2 = subject_box.astype(np.float32)
        body_height = max(1.0, y2 - y1)

        if garment_scope == "upper":
            box = np.array(
                [x1 + 0.12 * (x2 - x1), y1 + 0.16 * body_height, x2 - 0.12 * (x2 - x1), y1 + 0.58 * body_height],
                dtype=np.float32,
            )
        elif garment_scope == "lower":
            box = np.array(
                [x1 + 0.10 * (x2 - x1), y1 + 0.48 * body_height, x2 - 0.10 * (x2 - x1), y2 - 0.02 * body_height],
                dtype=np.float32,
            )
        else:
            box = np.array(
                [x1 + 0.08 * (x2 - x1), y1 + 0.14 * body_height, x2 - 0.08 * (x2 - x1), y2 - 0.05 * body_height],
                dtype=np.float32,
            )

        return _expand_box(box, width, height, pad=0.08)


def parse_box(raw_box: str | None) -> np.ndarray | None:
    if not raw_box:
        return None
    parts = [int(part.strip()) for part in raw_box.split(",")]
    if len(parts) != 4:
        raise ValueError("--cloth-box должен быть в формате x1,y1,x2,y2")
    return np.array(parts, dtype=np.float32)


def validate_sam_source(
    model_id: str | None,
    checkpoint: str | None,
    config_file: str | None,
) -> None:
    if model_id:
        return
    if checkpoint and config_file:
        return
    raise ValueError("Укажите --sam-model-id или одновременно --sam-config и --sam-checkpoint")