from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TrackMemory:
    track_id: int
    class_name: str
    bbox: List[float]
    confidence: float
    stable_count: int = 1
    missing_frames: int = 0
    label_votes: Dict[str, float] = field(default_factory=dict)


class SimpleFrameTracker:
    """本科毕设可解释版本的轻量时序跟踪器。

    功能：
    1. 基于 IoU 将前后帧目标进行关联
    2. 对 bbox 做指数平滑，降低视频框抖动
    3. 对类别做投票，降低类别来回跳变
    4. 通过 stable_count 提供稳定显示依据
    """

    def __init__(self, iou_threshold: float = 0.35, max_missing: int = 5, smooth_alpha: float = 0.55):
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.smooth_alpha = smooth_alpha
        self._tracks: Dict[int, TrackMemory] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    @staticmethod
    def _iou(a: List[float], b: List[float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        iw = max(0.0, inter_x2 - inter_x1)
        ih = max(0.0, inter_y2 - inter_y1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
        return inter / (area_a + area_b - inter)

    def _smooth_bbox(self, old_box: List[float], new_box: List[float]) -> List[float]:
        a = self.smooth_alpha
        return [
            round(old_box[0] * (1 - a) + new_box[0] * a, 2),
            round(old_box[1] * (1 - a) + new_box[1] * a, 2),
            round(old_box[2] * (1 - a) + new_box[2] * a, 2),
            round(old_box[3] * (1 - a) + new_box[3] * a, 2),
        ]

    def _register(self, det: dict) -> dict:
        track_id = self._next_id
        self._next_id += 1
        self._tracks[track_id] = TrackMemory(
            track_id=track_id,
            class_name=det['class_name'],
            bbox=list(det['bbox']),
            confidence=float(det['confidence']),
            stable_count=1,
            label_votes={det['class_name']: float(det['confidence'])}
        )
        det['track_id'] = track_id
        det['stable_count'] = 1
        return det

    def update(self, detections: List[dict]) -> Tuple[List[dict], bool, int]:
        assigned_tracks = set()
        stable_count = 0

        for det in detections:
            best_track_id: Optional[int] = None
            best_iou = 0.0
            for track_id, track in self._tracks.items():
                if track_id in assigned_tracks:
                    continue
                iou = self._iou(track.bbox, det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is None or best_iou < self.iou_threshold:
                self._register(det)
                continue

            track = self._tracks[best_track_id]
            track.bbox = self._smooth_bbox(track.bbox, det['bbox'])
            track.confidence = round(track.confidence * 0.35 + float(det['confidence']) * 0.65, 4)
            track.label_votes[det['class_name']] = track.label_votes.get(det['class_name'], 0.0) + float(det['confidence'])
            track.class_name = max(track.label_votes, key=track.label_votes.get)
            track.stable_count += 1
            track.missing_frames = 0

            det['track_id'] = best_track_id
            det['bbox'] = list(track.bbox)
            det['class_name'] = track.class_name
            det['confidence'] = max(float(det['confidence']), track.confidence)
            det['stable_count'] = track.stable_count
            assigned_tracks.add(best_track_id)

        for track_id in list(self._tracks.keys()):
            if track_id not in assigned_tracks:
                self._tracks[track_id].missing_frames += 1
                if self._tracks[track_id].missing_frames > self.max_missing:
                    self._tracks.pop(track_id, None)

        for det in detections:
            track = self._tracks.get(det['track_id'])
            if track and track.stable_count >= 3:
                stable_count += 1

        detections.sort(key=lambda x: (x.get('stable_count', 0), x.get('confidence', 0)), reverse=True)
        return detections, stable_count > 0, stable_count
