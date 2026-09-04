import cv2
import numpy as np

class DocumentPerspectiveWarper:
    """Performs 4-point homography normalisation with auto-bypass for pristine digital documents."""

    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]      # Top-left
        rect[2] = pts[np.argmax(s)]      # Bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]   # Top-right
        rect[3] = pts[np.argmax(diff)]   # Bottom-left
        return rect

    @classmethod
    def extract_and_warp(cls, img_cv: np.ndarray) -> np.ndarray:
        if img_cv is None or img_cv.size == 0:
            return img_cv

        h, w = img_cv.shape[:2]
        total_area = h * w
        aspect = float(w) / max(h, 1)

        # Full-Sheet A4 e-Aadhaar Bypass: Never crop or warp digital scans
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
        white_pixels = np.sum(gray > 235)
        is_pristine_scan = (white_pixels / float(total_area)) > 0.30

        if is_pristine_scan or aspect < 0.92:
            return img_cv

        # Handheld Camera ID Card Boundary Isolation
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 40, 160)
        dilated = cv2.dilate(edged, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        doc_contour = None
        for c in contours:
            area = cv2.contourArea(c)
            if area > (total_area * 0.20):
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    doc_contour = approx
                    break

        if doc_contour is None:
            return img_cv

        pts = doc_contour.reshape(4, 2)
        rect = cls.order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        if maxWidth < 200 or maxHeight < 140:
            return img_cv

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img_cv, M, (maxWidth, maxHeight))
        return warped if (warped is not None and warped.size > 0) else img_cv
