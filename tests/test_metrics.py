"""Reconstruction metric utilities (sections 4.2, 5.2)."""

import torch

from quaternion import (
    cosine_similarity,
    max_abs_error,
    mse,
    reconstruction_report,
    relative_frobenius_error,
    rmse,
)


def test_perfect_reconstruction():
    a = torch.randn(4, 5)
    assert mse(a, a) == 0.0
    assert rmse(a, a) == 0.0
    assert relative_frobenius_error(a, a) == 0.0
    assert max_abs_error(a, a) == 0.0
    assert abs(cosine_similarity(a, a) - 1.0) < 1e-9


def test_known_values():
    a = torch.zeros(4)
    b = torch.ones(4)
    assert mse(a, b) == 1.0
    assert rmse(a, b) == 1.0
    assert max_abs_error(a, b) == 1.0
    assert abs(relative_frobenius_error(a, b) - 1.0) < 1e-9
    assert abs(cosine_similarity(a, b)) < 1e-9


def test_report_keys_and_types():
    a = torch.randn(3, 3)
    b = a + 0.01 * torch.randn(3, 3)
    report = reconstruction_report(a, b)
    assert set(report) == {
        "mse",
        "rmse",
        "relative_frobenius_error",
        "cosine_similarity",
        "max_abs_error",
    }
    assert all(isinstance(v, float) for v in report.values())


def test_shape_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        mse(torch.randn(2, 2), torch.randn(2, 3))
