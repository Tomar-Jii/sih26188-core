import uuid
from datetime import datetime, timezone

class ForensicAuditTrail:
    """Manages high-precision event logging and ephemeral session storage."""

    def __init__(self, case_id: str = None):
        self.case_id = case_id if case_id else f"AEG-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        self.events = []

    def log(self, event_description: str, status: str = "PASS"):
        timestamp_iso = datetime.now(timezone.utc).strftime("%H:%M:%S.%fZ")[:-4] + "Z"
        self.events.append({
            "time": timestamp_iso,
            "timestamp": timestamp_iso,
            "event": event_description,
            "status": status
        })

    def get_timeline(self) -> list:
        return self.events


class EphemeralCaseLedger:
    """In-memory case registry preventing persistent storage of sensitive identity documents."""
    _REGISTRY: dict = {}

    @classmethod
    def register(cls, case_id: str, payload: dict):
        cls._REGISTRY[case_id] = payload

    @classmethod
    def fetch(cls, case_id: str) -> dict:
        return cls._REGISTRY.get(case_id)

    @classmethod
    def purge_all(cls):
        cls._REGISTRY.clear()
