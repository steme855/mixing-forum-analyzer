"""Personal preset libraries for users and future DAW/VST sync."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_PRESET_ROOT = Path(__file__).resolve().parents[2] / "data" / "personal_presets"


@dataclass(frozen=True)
class PersonalPreset:
    """User-owned preset record."""

    user_id: str
    name: str
    plugin: str = ""
    category: str = "custom"
    description: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    preset_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["tags"] = list(self.tags)
        return record


class PersonalPresetLibrary:
    """JSON-backed personal preset library."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else DEFAULT_PRESET_ROOT

    def add_preset(self, preset: PersonalPreset) -> PersonalPreset:
        if not preset.user_id.strip():
            raise ValueError("user_id must not be empty")
        if not preset.name.strip():
            raise ValueError("preset name must not be empty")

        presets = self.list_presets(preset.user_id)
        presets.append(preset)
        self._write_user_presets(preset.user_id, presets)
        return preset

    def list_presets(self, user_id: str, category: str | None = None) -> list[PersonalPreset]:
        if not user_id.strip():
            return []
        path = self._user_path(user_id)
        if not path.exists():
            return []

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

        records = payload.get("presets", []) if isinstance(payload, dict) else []
        presets = [_preset_from_record(record) for record in records if isinstance(record, dict)]
        if category:
            presets = [preset for preset in presets if preset.category == category]
        return presets

    def delete_preset(self, user_id: str, preset_id: str) -> bool:
        presets = self.list_presets(user_id)
        kept = [preset for preset in presets if preset.preset_id != preset_id]
        if len(kept) == len(presets):
            return False
        self._write_user_presets(user_id, kept)
        return True

    def _write_user_presets(self, user_id: str, presets: list[PersonalPreset]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "1.0",
            "user_id": user_id,
            "presets": [preset.to_record() for preset in presets],
        }
        self._user_path(user_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _user_path(self, user_id: str) -> Path:
        safe_user = "".join(char for char in user_id if char.isalnum() or char in {"-", "_"}).strip()
        if not safe_user:
            safe_user = "anonymous"
        return self.root / f"{safe_user}.json"


def _preset_from_record(record: dict[str, Any]) -> PersonalPreset:
    tags = record.get("tags", ())
    if isinstance(tags, list):
        normalized_tags = tuple(str(tag) for tag in tags)
    elif isinstance(tags, str):
        normalized_tags = (tags,)
    else:
        normalized_tags = ()

    return PersonalPreset(
        preset_id=str(record.get("preset_id", "")) or str(uuid.uuid4()),
        user_id=str(record.get("user_id", "")),
        name=str(record.get("name", "")),
        plugin=str(record.get("plugin", "")),
        category=str(record.get("category", "custom")),
        description=str(record.get("description", "")),
        settings=record.get("settings", {}) if isinstance(record.get("settings"), dict) else {},
        tags=normalized_tags,
        created_at=str(record.get("created_at", "")) or datetime.now(timezone.utc).isoformat(),
        updated_at=str(record.get("updated_at", "")) or datetime.now(timezone.utc).isoformat(),
    )
