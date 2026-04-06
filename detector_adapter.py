from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from ultralytics.engine.model import Model

from config import ensure_katacr_environment

ensure_katacr_environment()

from katacr.constants.label_list import idx2unit  # noqa: E402
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
    def __init__(self, weights: str, device: str = "cuda", conf_thres: float = 0.25, iou_thres: float = 0.45) -> None:
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.model = YOLOCRLive(weights)

    def _class_name(self, class_id: int) -> str:
        names = getattr(self.model, "names", None)
        if isinstance(names, dict) and class_id in names:
            return str(names[class_id])
        return str(idx2unit.get(class_id, class_id))

    def predict(self, arena_bgr: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            arena_bgr,
            device=self.device,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
        )[0]
        raw = result.get_data()
        detections: list[Detection] = []
        for row in raw:
            if len(row) < 7:
                continue
            x1, y1, x2, y2 = [float(v) for v in row[:4]]
            track_id = int(row[4]) if len(row) == 8 else None
            conf = float(row[-3])
            class_id = int(row[-2])
            belong = int(row[-1]) if len(row) >= 7 else None
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
