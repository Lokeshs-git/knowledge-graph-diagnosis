"""Tests for scorers."""

from evals.scorers import contains, exact_match


def test_exact_match_passes_on_identical_strings() -> None:
    score = exact_match("Paris", "Paris")
    assert score.value == 1.0
    assert score.passed


def test_exact_match_is_case_insensitive() -> None:
    score = exact_match("paris", "PARIS")
    assert score.value == 1.0


def test_exact_match_trims_whitespace() -> None:
    score = exact_match("  Paris  ", "Paris")
    assert score.value == 1.0


def test_exact_match_fails_on_different_strings() -> None:
    score = exact_match("Lyon", "Paris")
    assert score.value == 0.0
    assert not score.passed
    assert "Paris" in score.reason


def test_contains_passes_when_expected_is_substring() -> None:
    score = contains("The capital is Paris.", "Paris")
    assert score.value == 1.0


def test_contains_fails_when_expected_is_missing() -> None:
    score = contains("The capital is Lyon.", "Paris")
    assert score.value == 0.0
