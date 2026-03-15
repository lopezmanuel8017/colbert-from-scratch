"""Tests for visualization utilities."""

import numpy as np
import matplotlib
import matplotlib.figure
import pytest

from colbert_from_scratch.viz import plot_maxsim_heatmap


class TestPlotMaxsimHeatmap:
    @pytest.fixture
    def simple_case(self):
        Q = np.eye(3, 4, dtype=np.float32)
        D = np.eye(5, 4, dtype=np.float32)
        q_tokens = ["tok_a", "tok_b", "tok_c"]
        d_tokens = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
        return Q, D, q_tokens, d_tokens

    def test_returns_figure(self, simple_case):
        fig = plot_maxsim_heatmap(*simple_case)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_has_two_axes(self, simple_case):
        fig = plot_maxsim_heatmap(*simple_case)
        axes = fig.get_axes()
        assert len(axes) >= 2

    def test_does_not_call_show(self, simple_case, monkeypatch):
        show_called = []
        monkeypatch.setattr(matplotlib.pyplot, "show", lambda *a, **k: show_called.append(1))
        plot_maxsim_heatmap(*simple_case)
        assert len(show_called) == 0

    def test_title_contains_score(self, simple_case):
        fig = plot_maxsim_heatmap(*simple_case)
        title = fig.texts[0].get_text() if fig.texts else ""
        assert "MaxSim" in title
