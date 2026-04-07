from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence

import numpy as np
import torch
import torchvision

from config import ensure_katacr_environment

ensure_katacr_environment()

from ultralytics.engine.model import Model

from katacr.constants.label_list import idx2unit, unit2idx  # noqa: E402
from katacr.yolov8.custom_model import CRDetectionModel  # noqa: E402
from katacr.yolov8.custom_predict import CRDetectionPredictor  # noqa: E402


@dataclass(slots=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    conf: float
    class_id: int
    class_name: str
    belong: int | None
    track_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["xyxy"] = [round(float(v), 3) for v in self.xyxy]
        payload["conf"] = round(self.conf, 5)
        return payload


class YOLOCRLive(Model):
    def __init__(self, model: str, task: str | None = None, verbose: bool = False) -> None:
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, object]]:
        return {
            "detect": {
                "model": CRDetectionModel,
                "trainer": None,
                "validator": None,
                "predictor": CRDetectionPredictor,
            }
        }


class KataCRDetector:
    def __init__(
        self,
        weights: Sequence[str],
        device: str = "cpu",
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        allowed_class_ids: Sequence[int] | None = None,
    ) -> None:
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.allowed_class_ids = set(allowed_class_ids or ())
        self.models = [YOLOCRLive(path) for path in weights]
        # Pre-build per-model class-ID remap tables (local model ID -> global KataCR ID).
        self._class_remaps: list[np.ndarray] = [self._build_remap(m) for m in self.models]

    def _build_remap(self, model: YOLOCRLive) -> np.ndarray:
        names = getattr(model, "names", None) or {}
        if not names:
            return np.arange(256, dtype=np.int64)
        max_id = max(names.keys())
        remap = np.arange(max_id + 1, dtype=np.int64)
        for local_id, name in names.items():
            remap[local_id] = int(unit2idx.get(str(name), local_id))
        return remap

    def _class_name(self, class_id: int) -> str:
        return str(idx2unit.get(class_id, class_id))

    def predict(self, arena_bgr: np.ndarray) -> list[Detection]:
        # Tell YOLO to infer at the actual input size — not the model's saved default
        # (usually 640).  Without this, a 320-px pre-resized crop gets padded back to
        # 640×640 and inference is 4× slower than necessary.
        imgsz = max(arena_bgr.shape[:2])

        merged_tensors: list[torch.Tensor] = []
        for model, remap in zip(self.models, self._class_remaps):
            result = model.predict(
                arena_bgr,
                device=self.device,
                conf=self.conf_thres,
                iou=self.iou_thres,
                imgsz=imgsz,
                verbose=False,
            )[0]
            boxes = result.orig_boxes
            if boxes is None or len(boxes) == 0:
                continue
            remapped = boxes.clone()
            # Vectorised class-ID remap — avoids a Python loop over every detection.
            local_ids = remapped[:, 5].long().numpy()
            clamped = np.clip(local_ids, 0, len(remap) - 1)
            remapped[:, 5] = torch.from_numpy(remap[clamped]).to(remapped.dtype)
            merged_tensors.append(remapped)

        if not merged_tensors:
            return []

        if len(merged_tensors) == 1:
            # Single model: YOLO already ran NMS — no need for a second pass.
            raw = merged_tensors[0].detach().cpu().numpy()
        else:
            merged = torch.cat(merged_tensors, dim=0)
            keep = torchvision.ops.nms(merged[:, :4], merged[:, 4], iou_threshold=self.iou_thres)
            raw = merged[keep].detach().cpu().numpy()

        detections: list[Detection] = []
        allowed = self.allowed_class_ids
        for row in raw:
            if len(row) < 7:
                continue
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            track_id = int(row[4]) if len(row) == 8 else None
            conf = float(row[-3])
            class_id = int(row[-2])
            if allowed and class_id not in allowed:
                continue
            belong = int(row[-1])
            detections.append(
                Detection(
                    xyxy=(x1, y1, x2, y2),
                    conf=conf,
                    class_id=class_id,
                    class_name=self._class_name(class_id),
                    belong=belong,
                    track_id=track_id,
                )
            )
        return detections
