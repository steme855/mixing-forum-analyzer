from __future__ import annotations

import pytest

from personal_presets import PersonalPreset, PersonalPresetLibrary


def test_personal_preset_library_adds_lists_and_filters(tmp_path) -> None:
    library = PersonalPresetLibrary(tmp_path)

    preset = library.add_preset(
        PersonalPreset(
            user_id="user-1",
            name="Kick Cleanup",
            plugin="Pro-Q",
            category="eq",
            tags=("kick", "cleanup"),
            settings={"freq_hz": 250, "gain_db": -2.0},
        )
    )

    presets = library.list_presets("user-1")
    eq_presets = library.list_presets("user-1", category="eq")

    assert presets[0].preset_id == preset.preset_id
    assert presets[0].settings == {"freq_hz": 250, "gain_db": -2.0}
    assert eq_presets == presets


def test_personal_preset_library_deletes_preset(tmp_path) -> None:
    library = PersonalPresetLibrary(tmp_path)
    preset = library.add_preset(PersonalPreset(user_id="user-1", name="Vocal Air"))

    assert library.delete_preset("user-1", preset.preset_id) is True
    assert library.list_presets("user-1") == []
    assert library.delete_preset("user-1", preset.preset_id) is False


def test_personal_preset_library_rejects_empty_name(tmp_path) -> None:
    library = PersonalPresetLibrary(tmp_path)

    with pytest.raises(ValueError, match="preset name"):
        library.add_preset(PersonalPreset(user_id="user-1", name=" "))
