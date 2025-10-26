from __future__ import annotations

from typing import Iterable

import pytest

from presets.preset_recommender import PresetRecommendation, PresetRecommender


@pytest.fixture(scope="module")
def recommender() -> PresetRecommender:
    return PresetRecommender()


@pytest.mark.unit
def test_preset_library_not_empty(recommender: PresetRecommender) -> None:
    assert recommender.suggest("blechern"), "Library sollte mindestens einen Treffer liefern"


@pytest.mark.unit
def test_recommendation_schema(recommender: PresetRecommender) -> None:
    recommendation = recommender.suggest("s-laute zu scharf")[0]
    assert isinstance(recommendation, PresetRecommendation)
    assert recommendation.type
    assert recommendation.notes


@pytest.mark.unit
def test_recommendation_type_matches_keyword(recommender: PresetRecommender) -> None:
    recommendation = recommender.suggest("zu viel 3kHz, ziemlich blechern")[0]
    assert "EQ" in recommendation.type
    assert recommendation.frequency == 3000


@pytest.mark.unit
def test_severity_adjusts_gain(recommender: PresetRecommender) -> None:
    medium_gain = recommender.suggest("hi hats sind harsch", severity="medium")[0].gain_db
    strong_gain = recommender.suggest("hi hats sind harsch", severity="strong")[0].gain_db
    assert medium_gain is not None and strong_gain is not None
    assert strong_gain < medium_gain
