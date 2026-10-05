"""Tests deterministic target synonyms and fuzzy suggestions."""

import pytest
from surface2anatomy.synonyms import resolve_target
from surface2anatomy.exceptions import UnknownTargetError

def test_exact_resolution():
    canon, idx = resolve_target("spleen")
    assert canon == "spleen"
    assert idx == 0

def test_synonym_resolution():
    canon, idx = resolve_target("left kidney")
    assert canon == "kidney_left"
    assert idx == 2

    canon, idx = resolve_target("ivc")
    assert canon == "inferior_vena_cava"

    canon, idx = resolve_target("lien")
    assert canon == "spleen"

def test_unknown_target_suggestions():
    with pytest.raises(UnknownTargetError) as exc_info:
        resolve_target("splenn")
    assert "spleen" in exc_info.value.suggestions
