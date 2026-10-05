"""Tests target catalog and anatomical categories."""

from surface2anatomy.targets import (
    list_targets, CANONICAL_TARGET_NAMES, PRIMARY_BENCHMARK_TARGETS, TARGET_CATEGORIES
)

def test_target_counts():
    all_t = list_targets()
    assert len(all_t) == 121
    bench_t = list_targets(benchmark_only=True)
    assert len(bench_t) == 104

def test_categories():
    assert "Abdominal & Visceral" in TARGET_CATEGORIES
    assert "Head & Brain" in TARGET_CATEGORIES
    assert "liver" in list_targets(category="Abdominal & Visceral")
    assert "brain" in list_targets(category="Head & Brain")
