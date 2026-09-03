from dataclasses import dataclass

@dataclass(frozen=True)
class ForensicConfig:
    APP_VERSION: str = "5.0.0-enterprise"
    MAX_PAYLOAD_BYTES: int = 12 * 1024 * 1024  # 12 MB limit
    ALLOWED_MIME_TYPES: tuple = ("image/jpeg", "image/png", "image/webp", "image/bmp")
    
    # Quality Engine Thresholds
    MIN_SHARPNESS_LAPLACIAN: float = 14.0
    MAX_SPECULAR_GLARE_RATIO: float = 0.20
    MIN_DOCUMENT_WIDTH: int = 200
    MIN_DOCUMENT_HEIGHT: int = 150
    
    # Forensic Sensitivity Bounds
    ELA_JPEG_QUALITY: int = 90
    ELA_DYNAMIC_THRESH_FLOOR: int = 50
    ELA_DYNAMIC_THRESH_CEIL: int = 78
    MIN_TAMPER_CONTOUR_AREA: int = 35
    MAX_TAMPER_CONTOUR_AREA_RATIO: float = 0.12
    MOIRE_PAPR_THRESHOLD: float = 3.90
    
    # Risk Fusion Weights
    WEIGHT_DETERMINISTIC_CHECKSUM: float = 0.30
    WEIGHT_SPATIAL_TAMPERING: float = 0.40
    WEIGHT_SCREEN_SPOOF: float = 0.20
    WEIGHT_METADATA_TRACE: float = 0.10

CONFIG = ForensicConfig()
