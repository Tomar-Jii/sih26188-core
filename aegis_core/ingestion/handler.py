import io
import cv2
import numpy as np
from PIL import Image
from fastapi import UploadFile
from aegis_core.config import CONFIG

class IngestionSecurityError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

class StreamIngestionHandler:
    """Performs deep binary header validation, chunked stream quarantine, and safe matrix decoding."""

    MAGIC_SIGNATURES = {
        "jpeg": (b"\xFF\xD8\xFF",),
        "png": (b"\x89PNG\r\n\x1a\n",),
        "webp": (b"RIFF", b"WEBP"),
        "bmp": (b"BM",)
    }

    @classmethod
    def verify_magic_bytes(cls, header: bytes) -> str:
        """Inspects raw leading bytes to determine legitimate binary MIME identity."""
        if len(header) < 12:
            raise IngestionSecurityError("Truncated binary stream: Insufficient header bytes.", 400)

        # Check JPEG
        if header.startswith(cls.MAGIC_SIGNATURES["jpeg"][0]):
            return "image/jpeg"

        # Check PNG
        if header.startswith(cls.MAGIC_SIGNATURES["png"][0]):
            return "image/png"

        # Check WEBP: starts with RIFF and has WEBP at offset 8
        if header[:4] == cls.MAGIC_SIGNATURES["webp"][0] and header[8:12] == cls.MAGIC_SIGNATURES["webp"][1]:
            return "image/webp"

        # Check BMP
        if header.startswith(cls.MAGIC_SIGNATURES["bmp"][0]):
            return "image/bmp"

        raise IngestionSecurityError(
            "MagicByteVerificationFailed: File signature does not match permissible image binaries.", 
            415
        )

    @classmethod
    async def ingest_and_sanitize(cls, file: UploadFile) -> tuple[bytes, Image.Image, np.ndarray, str]:
        """Reads stream with memory bounding, validates magic bytes, and decodes unperturbed rasters."""
        buffer = bytearray()
        chunk_size = 64 * 1024  # 64 KB chunks
        total_read = 0

        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total_read += len(chunk)
            if total_read > CONFIG.MAX_PAYLOAD_BYTES:
                raise IngestionSecurityError(
                    f"PayloadTooLarge: Stream exceeded ceiling of {CONFIG.MAX_PAYLOAD_BYTES // (1024*1024)}MB.", 
                    413
                )
            buffer.extend(chunk)

        raw_bytes = bytes(buffer)
        if len(raw_bytes) == 0:
            raise IngestionSecurityError("EmptyPayload: Uploaded stream contains 0 bytes.", 400)

        # 1. Inspect Magic Bytes
        verified_mime = cls.verify_magic_bytes(raw_bytes[:16])

        # 2. Defend against PIL Decompression Bombs
        Image.MAX_IMAGE_PIXELS = 40_000_000

        # 3. Decode PIL Object safely
        try:
            pil_img = Image.open(io.BytesIO(raw_bytes))
            pil_img.load()
            pil_rgb = pil_img.convert("RGB")
        except Exception as err:
            raise IngestionSecurityError(f"RasterDecodeFault: PIL cannot process image raster ({str(err)}).", 422)

        # 4. Decode OpenCV BGR Matrix safely
        try:
            img_cv = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img_cv is None or img_cv.size == 0:
                raise ValueError("cv2.imdecode returned an empty buffer.")
        except Exception as err:
            raise IngestionSecurityError(f"PixelMatrixFault: OpenCV cannot decode raster buffer ({str(err)}).", 422)

        return raw_bytes, pil_rgb, img_cv, verified_mime
