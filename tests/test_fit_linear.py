"""Matrix and activation-aware quaternion reconstruction (Stages 1-2)."""

import torch

from conversion.fit_linear import (
    project_onto_quaternion,
    reconstruct_from_activations,
    reconstruct_matrix,
)
from quaternion import build_block_matrix, quaternion_linear


def true_weights(og, ig, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randn(og, ig, generator=g, dtype=torch.float64) / ig**0.5
        for _ in range(4)
    ]


def test_projection_recovers_exact_block_matrix():
    w = true_weights(3, 4)
    block = build_block_matrix(*w)  # keep float64 for exact recovery
    recovered = project_onto_quaternion(block)
    for got, want in zip(recovered, w):
        assert torch.allclose(got, want, atol=1e-9, rtol=1e-9)


def test_projection_metrics_are_perfect_on_block_matrix():
    w = true_weights(3, 4)
    block = build_block_matrix(*w).float()
    result = reconstruct_matrix(block, method="projection")
    assert result.metrics["relative_frobenius_error"] < 1e-6
    assert result.metrics["cosine_similarity"] > 1 - 1e-6
    assert result.compression_ratio == 4.0
    assert result.parameters == 4 * (3 * 4)
    assert result.dense_parameters == 12 * 16


def test_projection_is_optimal_for_random_matrix():
    g = torch.Generator().manual_seed(1)
    weight = torch.randn(16, 20, generator=g)

    proj = reconstruct_matrix(weight, method="projection")
    adam = reconstruct_matrix(weight, method="adamw", iters=400, lr=5e-3)

    # Closed form cannot be beaten by gradient descent on a linear problem.
    assert (
        proj.metrics["relative_frobenius_error"]
        <= adam.metrics["relative_frobenius_error"] + 1e-6
    )
    assert proj.metrics["relative_frobenius_error"] < 1.0


def test_adamw_reduces_loss_and_converges_on_block_matrix():
    w = true_weights(2, 2)
    block = build_block_matrix(*w).float()
    result = reconstruct_matrix(block, method="adamw", iters=500, lr=1e-2)
    assert len(result.history) == 500
    assert result.history[-1] < result.history[0]
    assert result.metrics["relative_frobenius_error"] < 0.05


def test_random_grouping_records_permutations_and_is_sane():
    g = torch.Generator().manual_seed(1)
    weight = torch.randn(16, 20, generator=g)
    result = reconstruct_matrix(weight, method="projection", grouping="random", seed=7)

    assert result.in_permutation is not None
    assert result.out_permutation is not None
    assert torch.equal(
        result.in_permutation.sort().values, torch.arange(20)
    )
    assert torch.equal(
        result.out_permutation.sort().values, torch.arange(16)
    )
    assert 0.0 <= result.metrics["relative_frobenius_error"] < 1.5
    assert result.metrics["cosine_similarity"] > 0.0


def test_random_grouping_differs_from_contiguous_for_block_matrix():
    # A matrix block-structured under the identity grouping is generally *not*
    # block-structured under an arbitrary channel permutation, so random
    # grouping must be measurably worse on this input.
    w = true_weights(3, 4)
    block = build_block_matrix(*w)
    contiguous = reconstruct_matrix(block, method="projection", grouping="contiguous")
    random = reconstruct_matrix(block, method="projection", grouping="random", seed=7)
    assert contiguous.metrics["relative_frobenius_error"] < 1e-9
    assert random.metrics["relative_frobenius_error"] > 1e-6


def test_activation_aware_reconstruction_fits_synthetic_layer():
    g = torch.Generator().manual_seed(3)
    true = true_weights(2, 2, seed=3)
    x = torch.randn(64, 8, generator=g, dtype=torch.float64)
    y = quaternion_linear(x, *true)

    result = reconstruct_from_activations(x, y, iters=800, lr=1e-2)
    assert result.metrics["relative_frobenius_error"] < 0.02
    assert result.metrics["cosine_similarity"] > 0.999
    assert len(result.history) == 800
