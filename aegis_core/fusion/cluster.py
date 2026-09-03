import numpy as np

class SpatialRegionMerger:
    """Merges overlapping tamper candidates and strictly requires a Primary Anchor signal."""

    @staticmethod
    def compute_iou(boxA: list, boxB: list) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter_area = inter_w * inter_h

        if inter_area == 0:
            return 0.0

        areaA = boxA[2] * boxA[3]
        areaB = boxB[2] * boxB[3]
        denom = float(areaA + areaB - inter_area)
        return inter_area / denom if denom > 0 else 0.0

    @staticmethod
    def is_nearby(boxA: list, boxB: list, max_distance: int = 12) -> bool:
        a_x1, a_y1, a_x2, a_y2 = boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]
        b_x1, b_y1, b_x2, b_y2 = boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]

        x_dist = max(0, max(a_x1, b_x1) - min(a_x2, b_x2))
        y_dist = max(0, max(a_y1, b_y1) - min(a_y2, b_y2))

        return x_dist <= max_distance and y_dist <= max_distance

    @classmethod
    def merge_regions(cls, raw_regions: list, iou_thresh: float = 0.18, max_dist: int = 12, **kwargs) -> list:
        thresh = kwargs.get("iou_threshold", kwargs.get("iou_thresh", iou_thresh))
        dist = kwargs.get("max_distance", kwargs.get("max_dist", max_dist))

        if not raw_regions:
            return []

        sorted_zones = sorted(raw_regions, key=lambda z: z["bbox"][2] * z["bbox"][3], reverse=True)
        merged = []

        while sorted_zones:
            current = sorted_zones.pop(0)
            cluster = [current]
            remaining = []

            for other in sorted_zones:
                iou = cls.compute_iou(current["bbox"], other["bbox"])
                nearby = cls.is_nearby(current["bbox"], other["bbox"], max_distance=dist)

                if iou > thresh or nearby:
                    cluster.append(other)
                else:
                    remaining.append(other)

            all_boxes = [z["bbox"] for z in cluster]
            x_min = min(b[0] for b in all_boxes)
            y_min = min(b[1] for b in all_boxes)
            x_max = max(b[0] + b[2] for b in all_boxes)
            y_max = max(b[1] + b[3] for b in all_boxes)

            signals = list(set([z.get("signal", "Spatial Anomaly") for z in cluster]))

            # PRIMARY ANCHORS: Real physical tampering signatures
            has_digital_ink = any("Digital Ink" in s for s in signals)
            has_strong_ela = any("Compression (ELA)" in s for s in signals)
            has_photo_swap = any("Photo-Swap" in s for s in signals)
            has_primary_anchor = has_digital_ink or has_strong_ela or has_photo_swap

            # CRITICAL SHIELD: If there is no primary anchor (only gradient/noise/font on text edges), REJECT IT!
            if not has_primary_anchor:
                sorted_zones = remaining
                continue

            is_multi_confirmed = len(signals) > 1
            base_score = max(z.get("score", 0.85) for z in cluster)
            final_score = min(0.98, base_score + (0.08 if is_multi_confirmed else 0.0))

            merged.append({
                "region_id": f"ZONE_{len(merged) + 1}",
                "bbox": [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
                "score": round(final_score, 2),
                "signals": signals,
                "multi_signal_verified": is_multi_confirmed
            })

            sorted_zones = remaining

        return merged
