"""
Open-vocabulary object detection + segmentation via Grounding DINO + SAM.

Based on the COTNAV/CL_CoTNav pipeline, simplified for single-frame
scene understanding on YCB-M data.
"""

import numpy as np
import torch
from loguru import logger
from mmdet.apis import DetInferencer
from segment_anything import SamPredictor, build_sam
from torchvision.ops import nms

# ---------- default checkpoint / config paths ----------
SAM_CHECKPOINT = (
    "/home/zongtai/Project/Codes/COTNAV/CL_CoTNav/checkpoints/sam_vit_h_4b8939.pth"
)
GROUNDING_DINO_CONFIG = (
    "/home/zongtai/Project/Codes/COTNAV/mmdetection/configs/mm_grounding_dino/"
    "grounding_dino_swin-l_pretrain_obj365_goldg.py"
)
GROUNDING_DINO_CHECKPOINT = (
    "/home/zongtai/Project/Codes/COTNAV/CL_CoTNav/checkpoints/"
    "grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth"
)


class GroundedSAMDetector:
    """Grounding DINO (via mmdet) + SAM for open-vocab detection & segmentation."""

    def __init__(
        self,
        classes: list[str],
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        nms_threshold: float = 0.5,
        device: str = "cuda",
        sam_checkpoint: str = SAM_CHECKPOINT,
        gdino_config: str = GROUNDING_DINO_CONFIG,
        gdino_checkpoint: str = GROUNDING_DINO_CHECKPOINT,
    ):
        self.classes = classes
        self.BOX_THRESHOLD = box_threshold
        self.TEXT_THRESHOLD = text_threshold
        self.NMS_THRESHOLD = nms_threshold
        self.DEVICE = device

        logger.info("Loading SAM model from {}", sam_checkpoint)
        self.sam_model = build_sam(checkpoint=sam_checkpoint).to(self.DEVICE)
        self.sam_predictor = SamPredictor(self.sam_model)

        logger.info("Loading Grounding DINO from {}", gdino_checkpoint)
        self.grounding_model = DetInferencer(
            model=gdino_config, weights=gdino_checkpoint, device=self.DEVICE
        )

        self._build_text_prompts()

    # ------------------------------------------------------------------ #
    #  Text prompt construction – split long vocab into chunks so that
    #  Grounding DINO doesn't truncate the text encoder input.
    # ------------------------------------------------------------------ #
    def _build_text_prompts(self):
        classes = self.classes
        n = len(classes)
        n_splits = max(1, (n + 9) // 10)  # ~10 classes per split
        step = n // n_splits
        ranges = [(i * step, (i + 1) * step) for i in range(n_splits - 1)]
        ranges.append((ranges[-1][1] if ranges else 0, n))

        self.text_prompts: list[str] = []
        self.prompt_labels: list[np.ndarray] = []
        for s, e in ranges:
            chunk = classes[s:e]
            self.prompt_labels.append(np.asarray(chunk))
            self.text_prompts.append(" . ".join(c.lower() for c in chunk) + " .")
        logger.info(
            "Built {} text prompt chunks for {} classes", len(self.text_prompts), n
        )

    # ------------------------------------------------------------------ #
    #  NMS helpers
    # ------------------------------------------------------------------ #
    def _per_class_nms(self, boxes, confs, labels):
        """Apply NMS independently per class."""
        keep_indices = []
        unique_labels = np.unique(labels)
        for lbl in unique_labels:
            mask = labels == lbl
            idx = torch.where(torch.from_numpy(mask))[0]
            cls_keep = nms(boxes[idx], confs[idx], self.NMS_THRESHOLD)
            keep_indices.append(idx[cls_keep])
        if not keep_indices:
            return boxes[:0], confs[:0], labels[:0]
        keep = torch.cat(keep_indices)
        return boxes[keep], confs[keep], labels[keep.cpu().numpy()]

    def _cross_class_dedup(self, boxes, confs, labels, iou_thresh=0.7):
        """Remove lower-confidence box when two different classes overlap heavily."""
        from torchvision.ops import box_iou as _box_iou

        if boxes.shape[0] <= 1:
            return boxes, confs, labels
        iou = _box_iou(boxes, boxes)
        order = torch.argsort(confs, descending=True)
        keep = torch.ones(len(boxes), dtype=torch.bool)
        for i, idx_i in enumerate(order):
            if not keep[idx_i]:
                continue
            for idx_j in order[i + 1 :]:
                if not keep[idx_j]:
                    continue
                if iou[idx_i, idx_j] > iou_thresh and labels[idx_i] != labels[idx_j]:
                    keep[idx_j] = False
        return boxes[keep], confs[keep], labels[keep.cpu().numpy()]

    # ------------------------------------------------------------------ #
    #  Core detect + segment
    # ------------------------------------------------------------------ #
    def detect_and_segment(self, rgb: np.ndarray):
        """
        Run full detection + segmentation on a single RGB image.

        Args:
            rgb: (H, W, 3) uint8 BGR or RGB numpy image.

        Returns:
            boxes:        (N, 4) tensor  – xyxy bounding boxes
            confidences:  (N,)   tensor  – detection scores
            labels:       (N,)   np.array of str – class labels
            masks:        (N, H, W) bool tensor  – per-object masks
        """
        h, w = rgb.shape[:2]
        self.sam_predictor.set_image(rgb)

        all_boxes, all_confs, all_labels = [], [], []

        for prompt, plabels in zip(self.text_prompts, self.prompt_labels):
            res = self.grounding_model(
                inputs=rgb,
                texts=prompt,
                pred_score_thr=self.TEXT_THRESHOLD,
                custom_entities=True,
            )
            preds = res["predictions"][0]
            all_boxes.extend(preds["bboxes"])
            all_confs.extend(preds["scores"])
            all_labels.extend(plabels[preds["labels"]].tolist())

        if len(all_boxes) == 0:
            return (
                torch.zeros((0, 4)),
                torch.zeros(0),
                np.array([], dtype=str),
                torch.zeros((0, h, w), dtype=torch.bool),
            )

        boxes = torch.tensor(all_boxes, dtype=torch.float32)
        confs = torch.tensor(all_confs, dtype=torch.float32)
        labels = np.array(all_labels)

        # confidence filter
        keep = confs > self.BOX_THRESHOLD
        boxes, confs, labels = boxes[keep], confs[keep], labels[keep.numpy()]

        if boxes.shape[0] == 0:
            return boxes, confs, labels, torch.zeros((0, h, w), dtype=torch.bool)

        # filter boxes > 80 % of image area (likely false positives)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        size_ok = areas < 0.8 * h * w
        boxes, confs, labels = boxes[size_ok], confs[size_ok], labels[size_ok.numpy()]

        if boxes.shape[0] == 0:
            return boxes, confs, labels, torch.zeros((0, h, w), dtype=torch.bool)

        # NMS
        nms_idx = nms(boxes, confs, self.NMS_THRESHOLD)
        boxes, confs, labels = boxes[nms_idx], confs[nms_idx], labels[nms_idx.cpu().numpy()]

        logger.info("After NMS: {} detections", boxes.shape[0])

        # SAM mask prediction from boxes
        transformed = self.sam_predictor.transform.apply_boxes_torch(
            boxes, (h, w)
        ).to(self.DEVICE)
        masks, _scores, _logits = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed,
            multimask_output=False,
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        masks = masks.cpu()
        confs = confs.cpu()

        return boxes.cpu(), confs, labels, masks
