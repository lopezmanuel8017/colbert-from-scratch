"""Tests for maxsim scoring functions."""

import numpy as np
import torch
import pytest

from colbert_from_scratch.maxsim import maxsim_np, maxsim_torch


@pytest.fixture
def identity_case():
    """Q == D (identical embeddings) — MaxSim should equal N_q after normalization."""
    vecs = np.random.randn(4, 16).astype(np.float32)
    return vecs, vecs.copy()


@pytest.fixture
def orthogonal_case():
    """Orthogonal Q and D — MaxSim should be near zero after normalization."""
    Q = np.eye(4, 8, dtype=np.float32)
    D = np.eye(4, 8, k=4, dtype=np.float32)
    return Q, D


@pytest.fixture
def known_score_case():
    """Hand-computed case: 2 query tokens, 3 doc tokens, dim=2."""
    Q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    D = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    return Q, D, 2.0


class TestMaxsimNp:
    def test_returns_scalar(self, identity_case):
        Q, D = identity_case
        score = maxsim_np(Q, D)
        assert isinstance(score, float)

    def test_identical_vectors_score_equals_nq(self, identity_case):
        Q, D = identity_case
        score = maxsim_np(Q, D, normalize=True)
        assert pytest.approx(score, abs=1e-5) == Q.shape[0]

    def test_orthogonal_vectors_score_near_zero(self, orthogonal_case):
        Q, D = orthogonal_case
        score = maxsim_np(Q, D, normalize=True)
        assert pytest.approx(score, abs=1e-5) == 0.0

    def test_known_score(self, known_score_case):
        Q, D, expected = known_score_case
        score = maxsim_np(Q, D, normalize=True)
        assert pytest.approx(score, abs=1e-5) == expected

    def test_unnormalized_differs(self):
        Q = np.array([[3.0, 0.0]], dtype=np.float32)
        D = np.array([[3.0, 0.0]], dtype=np.float32)
        norm_score = maxsim_np(Q, D, normalize=True)
        raw_score = maxsim_np(Q, D, normalize=False)
        assert pytest.approx(norm_score, abs=1e-5) == 1.0
        assert pytest.approx(raw_score, abs=1e-5) == 9.0

    def test_score_is_asymmetric_in_shape(self):
        Q = np.random.randn(3, 8).astype(np.float32)
        D = np.random.randn(10, 8).astype(np.float32)
        score_qd = maxsim_np(Q, D)
        score_dq = maxsim_np(D, Q)
        assert score_qd != pytest.approx(score_dq, abs=1e-5)


class TestMaxsimTorch:
    def test_returns_tensor(self, identity_case):
        Q, D = identity_case
        score = maxsim_torch(torch.from_numpy(Q), torch.from_numpy(D))
        assert isinstance(score, torch.Tensor)

    def test_identical_vectors_score_equals_nq(self, identity_case):
        Q, D = identity_case
        score = maxsim_torch(torch.from_numpy(Q), torch.from_numpy(D), normalize=True)
        assert pytest.approx(score.item(), abs=1e-5) == Q.shape[0]

    def test_orthogonal_vectors_score_near_zero(self, orthogonal_case):
        Q, D = orthogonal_case
        score = maxsim_torch(torch.from_numpy(Q), torch.from_numpy(D), normalize=True)
        assert pytest.approx(score.item(), abs=1e-5) == 0.0

    def test_known_score(self, known_score_case):
        Q, D, expected = known_score_case
        score = maxsim_torch(torch.from_numpy(Q), torch.from_numpy(D), normalize=True)
        assert pytest.approx(score.item(), abs=1e-5) == expected

    def test_unnormalized_differs(self):
        Q = torch.tensor([[3.0, 0.0]])
        D = torch.tensor([[3.0, 0.0]])
        norm_score = maxsim_torch(Q, D, normalize=True)
        raw_score = maxsim_torch(Q, D, normalize=False)
        assert pytest.approx(norm_score.item(), abs=1e-5) == 1.0
        assert pytest.approx(raw_score.item(), abs=1e-5) == 9.0


class TestCrossConsistency:
    def test_np_and_torch_agree(self):
        np.random.seed(42)
        Q = np.random.randn(5, 32).astype(np.float32)
        D = np.random.randn(12, 32).astype(np.float32)
        np_score = maxsim_np(Q, D, normalize=True)
        torch_score = maxsim_torch(
            torch.from_numpy(Q), torch.from_numpy(D), normalize=True
        )
        assert pytest.approx(np_score, abs=1e-4) == torch_score.item()
