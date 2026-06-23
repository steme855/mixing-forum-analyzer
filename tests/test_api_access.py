from __future__ import annotations

import pytest

from api_access import APIKeyStore


def test_api_key_store_issues_key_without_persisting_plaintext(tmp_path) -> None:
    store = APIKeyStore(tmp_path)
    issued = store.issue_key(owner="studio-a", plan="pro", quota_limit=2)
    record = store.get_record(issued.key_id)

    assert issued.api_key.startswith("mfa_")
    assert record is not None
    assert record.owner == "studio-a"
    assert issued.api_key not in store.keys_path.read_text(encoding="utf-8")


def test_api_key_store_consumes_quota(tmp_path) -> None:
    store = APIKeyStore(tmp_path)
    issued = store.issue_key(owner="studio-a", quota_limit=1)

    consumed = store.validate_and_consume(issued.api_key, endpoint="/v1/analyze")

    assert consumed.quota_used == 1
    with pytest.raises(ValueError, match="quota"):
        store.validate_and_consume(issued.api_key, endpoint="/v1/analyze")


def test_api_key_store_rejects_invalid_key(tmp_path) -> None:
    store = APIKeyStore(tmp_path)

    with pytest.raises(ValueError, match="invalid"):
        store.validate_and_consume("mfa_invalid")
