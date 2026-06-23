"""API key and quota primitives for monetized assistant access."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_API_KEY_ROOT = Path(__file__).resolve().parents[2] / "data" / "api_keys"


@dataclass(frozen=True)
class APIKeyRecord:
    """Stored API key metadata. The plaintext key is never persisted."""

    key_id: str
    key_hash: str
    owner: str
    plan: str = "free"
    quota_limit: int = 100
    quota_used: int = 0
    active: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IssuedAPIKey:
    """Plaintext key returned once when a new key is created."""

    key_id: str
    api_key: str
    owner: str
    plan: str
    quota_limit: int


class APIKeyStore:
    """JSON-backed API key store with simple monthly-style quota counters."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else DEFAULT_API_KEY_ROOT
        self.keys_path = self.root / "keys.json"
        self.usage_path = self.root / "usage.jsonl"

    def issue_key(self, owner: str, plan: str = "free", quota_limit: int = 100) -> IssuedAPIKey:
        if not owner.strip():
            raise ValueError("owner must not be empty")
        if quota_limit < 1:
            raise ValueError("quota_limit must be positive")

        api_key = "mfa_" + secrets.token_urlsafe(32)
        key_id = secrets.token_hex(8)
        record = APIKeyRecord(
            key_id=key_id,
            key_hash=_hash_key(api_key),
            owner=owner,
            plan=plan,
            quota_limit=quota_limit,
        )
        records = self._load_records()
        records[key_id] = record
        self._write_records(records)
        return IssuedAPIKey(key_id=key_id, api_key=api_key, owner=owner, plan=plan, quota_limit=quota_limit)

    def validate_and_consume(self, api_key: str, units: int = 1, endpoint: str = "unknown") -> APIKeyRecord:
        if not api_key.strip():
            raise ValueError("missing API key")
        if units < 1:
            raise ValueError("usage units must be positive")

        records = self._load_records()
        key_hash = _hash_key(api_key)
        matched_key_id = None
        matched_record = None
        for key_id, record in records.items():
            if hmac.compare_digest(record.key_hash, key_hash):
                matched_key_id = key_id
                matched_record = record
                break

        if matched_record is None:
            raise ValueError("invalid API key")
        if not matched_record.active:
            raise ValueError("API key is inactive")
        if matched_record.quota_used + units > matched_record.quota_limit:
            raise ValueError("quota exceeded")

        updated = APIKeyRecord(
            key_id=matched_record.key_id,
            key_hash=matched_record.key_hash,
            owner=matched_record.owner,
            plan=matched_record.plan,
            quota_limit=matched_record.quota_limit,
            quota_used=matched_record.quota_used + units,
            active=matched_record.active,
            created_at=matched_record.created_at,
            metadata=matched_record.metadata,
        )
        records[matched_key_id or updated.key_id] = updated
        self._write_records(records)
        self._append_usage(updated, units=units, endpoint=endpoint)
        return updated

    def get_record(self, key_id: str) -> APIKeyRecord | None:
        return self._load_records().get(key_id)

    def _load_records(self) -> dict[str, APIKeyRecord]:
        if not self.keys_path.exists():
            return {}
        try:
            payload = json.loads(self.keys_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        rows = payload.get("keys", {}) if isinstance(payload, dict) else {}
        return {
            str(key_id): _record_from_payload(row)
            for key_id, row in rows.items()
            if isinstance(row, dict)
        }

    def _write_records(self, records: dict[str, APIKeyRecord]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "1.0",
            "keys": {key_id: record.to_record() for key_id, record in records.items()},
        }
        self.keys_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _append_usage(self, record: APIKeyRecord, units: int, endpoint: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "key_id": record.key_id,
            "owner": record.owner,
            "plan": record.plan,
            "endpoint": endpoint,
            "units": units,
            "quota_used": record.quota_used,
            "quota_limit": record.quota_limit,
        }
        with self.usage_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _record_from_payload(payload: dict[str, Any]) -> APIKeyRecord:
    return APIKeyRecord(
        key_id=str(payload.get("key_id", "")),
        key_hash=str(payload.get("key_hash", "")),
        owner=str(payload.get("owner", "")),
        plan=str(payload.get("plan", "free")),
        quota_limit=int(payload.get("quota_limit", 100)),
        quota_used=int(payload.get("quota_used", 0)),
        active=bool(payload.get("active", True)),
        created_at=str(payload.get("created_at", "")) or datetime.now(timezone.utc).isoformat(),
        metadata=payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {},
    )
