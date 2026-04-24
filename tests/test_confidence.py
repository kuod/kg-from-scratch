"""Tests for the confidence interval module."""
import pytest
from src.confidence import wilson_ci, _validate_confidence


def test_wilson_ci_all_evidence():
    lo, hi = wilson_ci(k=5, n=5)
    assert lo > 0.5
    assert hi == 1.0


def test_wilson_ci_no_evidence():
    lo, hi = wilson_ci(k=0, n=5)
    assert lo == 0.0
    assert hi == 0.0


def test_wilson_ci_zero_total():
    lo, hi = wilson_ci(k=0, n=0)
    assert lo == 0.0
    assert hi == 1.0


def test_wilson_ci_partial_evidence():
    lo, hi = wilson_ci(k=3, n=10)
    assert 0.0 < lo < 0.5
    assert hi < 1.0
    assert lo < hi


def test_wilson_ci_returns_valid_range():
    for k in range(11):
        lo, hi = wilson_ci(k, n=10)
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0
        assert lo <= hi


def test_wilson_ci_single_observation():
    lo, hi = wilson_ci(k=1, n=1)
    assert lo == 1.0
    assert hi == 1.0


def test_validate_confidence_fills_defaults():
    result = _validate_confidence({})
    assert "score" in result
    assert "label" in result
    assert "lower_bound" in result
    assert "upper_bound" in result
    assert result["lower_bound"] <= result["upper_bound"]


def test_validate_confidence_clamps_score():
    result = _validate_confidence({"score": 150})
    assert result["score"] == 100

    result = _validate_confidence({"score": -10})
    assert result["score"] == 0


def test_validate_confidence_swaps_inverted_bounds():
    result = _validate_confidence({"lower_bound": 0.9, "upper_bound": 0.1})
    assert result["lower_bound"] <= result["upper_bound"]


def test_validate_confidence_invalid_label():
    result = _validate_confidence({"label": "Very High"})
    assert result["label"] in ("High", "Medium", "Low")
