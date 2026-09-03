import io

from fastapi.testclient import TestClient
from PIL import Image

from main import app, MAX_UPLOAD_BYTES


client = TestClient(app)


def _png_bytes(mode: str = "RGB", size=(64, 64)):
    img = Image.new(mode, size, (120, 120, 120) if mode != "L" else 120)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_startup_root():
    res = client.get("/")
    assert res.status_code == 200
    assert "AegisID" in res.text


def test_case_lookup_missing():
    res = client.get("/api/cases/DOES-NOT-EXIST")
    assert res.status_code == 404


def test_audit_valid_upload_and_case_lookup():
    post = client.post(
        "/api/audit",
        files={"file": ("doc.png", _png_bytes(), "image/png")},
        data={"mrz_raw_input": ""},
    )
    assert post.status_code == 200
    payload = post.json()
    assert payload["system_state"] == "COMPLETED"
    assert payload["backend_version"]
    assert len(payload["sha256"]) == 64

    case_id = payload["case_id"]
    get_case = client.get(f"/api/cases/{case_id}")
    assert get_case.status_code == 200
    assert get_case.json()["case_id"] == case_id


def test_audit_rejects_empty_upload():
    res = client.post(
        "/api/audit",
        files={"file": ("empty.png", b"", "image/png")},
        data={"mrz_raw_input": ""},
    )
    assert res.status_code == 400
    payload = res.json()
    assert payload["error"]["code"] == "EMPTY_UPLOAD"
    assert payload["system_state"] == "ERROR"


def test_audit_rejects_unsupported_mime():
    res = client.post(
        "/api/audit",
        files={"file": ("note.txt", b"hello", "text/plain")},
        data={"mrz_raw_input": ""},
    )
    assert res.status_code == 415
    assert res.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_audit_rejects_corrupt_image():
    res = client.post(
        "/api/audit",
        files={"file": ("bad.png", b"not an image", "image/png")},
        data={"mrz_raw_input": ""},
    )
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "MALFORMED_IMAGE"


def test_audit_rejects_oversized_upload():
    huge = b"x" * (MAX_UPLOAD_BYTES + 1)
    res = client.post(
        "/api/audit",
        files={"file": ("huge.png", huge, "image/png")},
        data={"mrz_raw_input": ""},
    )
    assert res.status_code == 413
    assert res.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_audit_supports_grayscale_and_rgba_inputs():
    gray = client.post(
        "/api/audit",
        files={"file": ("gray.png", _png_bytes(mode="L", size=(8, 8)), "image/png")},
        data={"mrz_raw_input": ""},
    )
    rgba_img = Image.new("RGBA", (8, 8), (20, 40, 60, 120))
    b = io.BytesIO()
    rgba_img.save(b, format="PNG")
    rgba = client.post(
        "/api/audit",
        files={"file": ("rgba.png", b.getvalue(), "image/png")},
        data={"mrz_raw_input": ""},
    )
    assert gray.status_code == 200
    assert rgba.status_code == 200
