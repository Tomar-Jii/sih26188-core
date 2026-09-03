class SpatialRegionMerger:
    """Fuses multi-signal bounding boxes using Intersection-over-Union (IoU) Non-Maximum Suppression."""

    @staticmethod
    def compute_iou(boxA: list, boxB: list) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        return interArea / float(boxAArea + boxBArea - interArea)

    @classmethod
    def merge_regions(cls, candidate_regions: list, iou_threshold: float = 0.25) -> list:
        if not candidate_regions:
            return []

        # Sort descending by anomaly score
        sorted_regions = sorted(candidate_regions, key=lambda r: r.get("score", 0.5), reverse=True)
        merged = []

        while sorted_regions:
            current = sorted_regions.pop(0)
            overlaps = []
            remaining = []

            for other in sorted_regions:
                if cls.compute_iou(current["bbox"], other["bbox"]) > iou_threshold:
                    overlaps.append(other)
                else:
                    remaining.append(other)

            if overlaps:
                # Merge bounding coordinates to encompass all overlapping signals
                all_boxes = [current["bbox"]] + [o["bbox"] for o in overlaps]
                x_min = min(b[0] for b in all_boxes)
                y_min = min(b[1] for b in all_boxes)
                x_max = max(b[0] + b[2] for b in all_boxes)
                y_max = max(b[1] + b[3] for b in all_boxes)

                all_signals = list(set([current.get("signal", "Spatial Anomaly")] + 
                                       [o.get("signal", "Spatial Anomaly") for o in overlaps]))
                combined_score = min(0.98, max(current.get("score", 0.5), *(o.get("score", 0.5) for o in overlaps)) + 0.05)

                merged.append({
                    "bbox": [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
                    "score": round(combined_score, 2),
                    "signals": all_signals,
                    "signal_count": len(all_signals)
                })
            else:
                merged.append({
                    "bbox": current["bbox"],
                    "score": current.get("score", 0.65),
                    "signals": [current.get("signal", "Spatial Anomaly")],
                    "signal_count": 1
                })

            sorted_regions = remaining

        return merged
