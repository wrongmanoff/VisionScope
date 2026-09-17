import numpy as np
import pytest

from src.experiments.benchmark import (
    make_shift_pair, time_callable, disparity_quality
)


def test_make_shift_pair_is_deterministic():
    a1, b1 = make_shift_pair(seed=42)
    a2, b2 = make_shift_pair(seed=42)
    assert np.array_equal(a1, a2)
    assert np.array_equal(b1, b2)


def test_make_shift_pair_shapes():
    a, b = make_shift_pair(height=80, width=100, shift=5)
    assert a.shape == b.shape == (80, 100)


def test_timer():
    result, runtime = time_callable(lambda x: x + 1, 4, repeats=2)
    assert result == 5
    assert runtime >= 0


def test_disparity_quality():
    d = np.full((5, 5), 6.0)
    result = disparity_quality(d, 6)
    assert result["valid_ratio"] == 1.0
    assert result["mae"] == 0.0
    assert result["rmse"] == 0.0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        make_shift_pair(height=20)
    with pytest.raises(ValueError):
        time_callable(lambda: None, repeats=0)
