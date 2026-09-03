import cv2
import numpy as np
from aegis_core.config import CONFIG

class OpticalMoireAnalyzer:
    @staticmethod
    def inspect(img_cv: np.ndarray) -> dict:
        try:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
            gray = cv2.resize(gray, (512, 512))
            
            dft = np.fft.fft2(gray)
            dft_shift = np.fft.fftshift(dft)
            spectrum = 20 * np.log(np.abs(dft_shift) + 1e-9)
            
            rows, cols = gray.shape
            crow, ccol = rows // 2, cols // 2
            spectrum[crow-25:crow+25, ccol-25:ccol+25] = 0
            
            papr = float(np.percentile(spectrum, 99.8) / (np.mean(spectrum) + 1e-5))
            return {
                "papr_score": round(papr, 3),
                "is_screen_recapture": bool(papr > CONFIG.MOIRE_PAPR_THRESHOLD)
            }
        except Exception:
            return {"papr_score": 1.0, "is_screen_recapture": False}
