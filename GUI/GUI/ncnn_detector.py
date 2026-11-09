"""Small direct NCNN runtime for the exported YOLO26 detector."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class NCNNDetector:
    CLASS_NAMES = {0: "immature", 1: "mature", 2: "normal"}

    def __init__(self, model_dir: Path, input_size: int = 640, confidence: float = 0.25, iou: float = 0.7):
        import ncnn

        self.ncnn = ncnn
        self.model_dir = Path(model_dir)
        self.input_size = input_size
        self.confidence = confidence
        self.iou = iou

        param_path = self.model_dir / "model.ncnn.param"
        bin_path = self.model_dir / "model.ncnn.bin"
        if not param_path.is_file() or not bin_path.is_file():
            raise FileNotFoundError(f"Model NCNN harus berisi {param_path.name} dan {bin_path.name}: {self.model_dir}")

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        if self.net.load_param(str(param_path)) != 0:
            raise RuntimeError(f"Gagal memuat parameter NCNN: {param_path}")
        if self.net.load_model(str(bin_path)) != 0:
            raise RuntimeError(f"Gagal memuat bobot NCNN: {bin_path}")
        self.input_name = self.net.input_names()[0]
        self.output_names = sorted(self.net.output_names())

    def predict(self, frame_bgr: np.ndarray) -> list[dict]:
        height, width = frame_bgr.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        resized_width = round(width * scale)
        resized_height = round(height * scale)
        pad_left = (self.input_size - resized_width) // 2
        pad_top = (self.input_size - resized_height) // 2

        resized = cv2.resize(frame_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            self.input_size - resized_height - pad_top,
            pad_left,
            self.input_size - resized_width - pad_left,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        blob = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0

        extractor = self.net.create_extractor()
        try:
            extractor.input(self.input_name, self.ncnn.Mat(blob))
            extracted = []
            for output_name in self.output_names:
                status, output = extractor.extract(output_name)
                if status != 0:
                    raise RuntimeError(f"Gagal membaca output NCNN: {output_name}")
                extracted.append(np.asarray(output).copy())
        finally:
            del extractor

        if len(extracted) != 1:
            raise RuntimeError(f"Model deteksi mengeluarkan {len(extracted)} output; dibutuhkan satu tensor.")

        predictions = np.squeeze(extracted[0])
        if predictions.ndim != 2:
            raise RuntimeError(f"Bentuk output YOLO26 NCNN tidak dikenali: {predictions.shape}")
        if predictions.shape[0] == 7 and predictions.shape[1] != 7:
            predictions = predictions.T
        if predictions.shape[1] != 7:
            raise RuntimeError(f"Output YOLO26 NCNN seharusnya berisi 4 koordinat dan 3 skor kelas, didapat {predictions.shape}.")

        class_scores = predictions[:, 4:7]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores.max(axis=1)
        candidate_ids = np.flatnonzero(scores >= self.confidence)
        if not len(candidate_ids):
            return []

        boxes = []
        nms_boxes = []
        kept_scores = []
        kept_class_ids = []
        for index in candidate_ids:
            center_x, center_y, box_width, box_height = predictions[index, :4]
            x1 = (center_x - box_width / 2 - pad_left) / scale
            y1 = (center_y - box_height / 2 - pad_top) / scale
            x2 = (center_x + box_width / 2 - pad_left) / scale
            y2 = (center_y + box_height / 2 - pad_top) / scale
            x1 = float(np.clip(x1, 0, width))
            y1 = float(np.clip(y1, 0, height))
            x2 = float(np.clip(x2, 0, width))
            y2 = float(np.clip(y2, 0, height))
            box = [int(x1), int(y1), max(1, int(x2 - x1)), max(1, int(y2 - y1))]
            boxes.append(box)
            nms_box = box.copy()
            nms_box[0] += int(class_ids[index]) * 7680
            nms_boxes.append(nms_box)
            kept_scores.append(float(scores[index]))
            kept_class_ids.append(int(class_ids[index]))

        indices = cv2.dnn.NMSBoxes(nms_boxes, kept_scores, self.confidence, self.iou)
        if len(indices) == 0:
            return []

        detections = []
        for index in np.asarray(indices).reshape(-1):
            x, y, box_width, box_height = boxes[int(index)]
            class_id = kept_class_ids[int(index)]
            detections.append({
                "class_id": class_id,
                "class_name": self.CLASS_NAMES.get(class_id, str(class_id)),
                "confidence": kept_scores[int(index)],
                "box": (x, y, x + box_width, y + box_height),
            })
        return detections
