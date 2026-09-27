"""Phase 0: Hamilton product correctness tests."""

import pytest
import torch

from quaternion.reference import hamilton_product


def q(*components):
    return torch.tensor(components, dtype=torch.float64)


I = q(0, 1, 0, 0)
J = q(0, 0, 1, 0)
K = q(0, 0, 0, 1)
ONE = q(1, 0, 0, 0)


def test_identity():
    a = q(0.3, -1.2, 4.0, 0.5)
    assert torch.allclose(hamilton_product(a, ONE), a)
    assert torch.allclose(hamilton_product(ONE, a), a)


def test_basis_squares_are_minus_one():
    for basis in (I, J, K):
        assert torch.allclose(hamilton_product(basis, basis), q(-1, 0, 0, 0))


def test_basis_cyclic_products():
    assert torch.allclose(hamilton_product(I, J), K)
    assert torch.allclose(hamilton_product(J, K), I)
    assert torch.allclose(hamilton_product(K, I), J)


def test_basis_anti_cyclic_products():
    assert torch.allclose(hamilton_product(J, I), -K)
    assert torch.allclose(hamilton_product(K, J), -I)
    assert torch.allclose(hamilton_product(I, K), -J)


def test_non_commutative():
    a = q(0.2, 0.9, -0.4, 0.1)
    b = q(-0.7, 0.3, 0.8, -0.2)
    assert not torch.allclose(hamilton_product(a, b), hamilton_product(b, a))


def test_associative():
    a = torch.randn(16, 4, dtype=torch.float64)
    b = torch.randn(16, 4, dtype=torch.float64)
    c = torch.randn(16, 4, dtype=torch.float64)
    left = hamilton_product(hamilton_product(a, b), c)
    right = hamilton_product(a, hamilton_product(b, c))
    assert torch.allclose(left, right, atol=1e-10, rtol=1e-10)


def test_norm_is_multiplicative():
    a = torch.randn(64, 4, dtype=torch.float64)
    b = torch.randn(64, 4, dtype=torch.float64)
    na = a.norm(dim=-1)
    nb = b.norm(dim=-1)
    nab = hamilton_product(a, b).norm(dim=-1)
    assert torch.allclose(nab, na * nb, rtol=1e-10, atol=1e-10)


def test_matches_explicit_formula():
    a = torch.randn(32, 4, dtype=torch.float64)
    b = torch.randn(32, 4, dtype=torch.float64)
    ar, ai, aj, ak = a.unbind(-1)
    br, bi, bj, bk = b.unbind(-1)
    expected = torch.stack(
        (
            ar * br - ai * bi - aj * bj - ak * bk,
            ar * bi + ai * br + aj * bk - ak * bj,
            ar * bj - ai * bk + aj * br + ak * bi,
            ar * bk + ai * bj - aj * bi + ak * br,
        ),
        dim=-1,
    )
    assert torch.allclose(hamilton_product(a, b), expected)


def test_broadcasts_over_leading_axes():
    a = torch.randn(2, 3, 5, 4, dtype=torch.float64)
    b = torch.randn(1, 3, 1, 4, dtype=torch.float64)
    out = hamilton_product(a, b)
    assert out.shape == (2, 3, 5, 4)
    # Check one element against the scalar oracle.
    got = out[1, 2, 4]
    want = hamilton_product(a[1, 2, 4], b[0, 2, 0])
    assert torch.allclose(got, want)


def test_rejects_bad_last_dim():
    with pytest.raises(ValueError):
        hamilton_product(torch.randn(3), torch.randn(3))
