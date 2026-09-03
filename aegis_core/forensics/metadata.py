from PIL import Image
from PIL.ExifTags import TAGS

class MetadataFootprintAnalyzer:
    """Audits container metadata, software tags, and temporal editing stamps."""

    SUSPICIOUS_SOFTWARE = [
        "photoshop", "gimp", "canva", "picsart", "coreldraw",
        "lightroom", "snapseed", "pixelmator", "paint.net"
    ]

    @classmethod
    def inspect(cls, orig_pil: Image.Image) -> dict:
        suspicious_tags = []
        parsed_metadata = {}
        has_exif = False

        try:
            exif_data = orig_pil.getexif()
            if exif_data:
                has_exif = True
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    val_str = str(value)
                    parsed_metadata[tag_name] = val_str[:100]

                    val_lower = val_str.lower()
                    for tool in cls.SUSPICIOUS_SOFTWARE:
                        if tool in val_lower:
                            suspicious_tags.append(
                                f"Digital editor signature in tag '{tag_name}': {tool.upper()}"
                            )
        except Exception:
            pass

        # Check for creation vs modification timestamp discrepancies
        temporal_anomaly = False
        dt_orig = parsed_metadata.get("DateTimeOriginal")
        dt_mod = parsed_metadata.get("DateTime")
        if dt_orig and dt_mod and dt_orig != dt_mod:
            temporal_anomaly = True
            suspicious_tags.append(f"Timestamp divergence: Original ({dt_orig}) vs Modified ({dt_mod})")

        return {
            "has_exif": has_exif,
            "software_traces": suspicious_tags,
            "temporal_anomaly": temporal_anomaly,
            "raw_summary": parsed_metadata
        }
