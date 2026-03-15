"""Visualization utilities for ColBERT similarity matrices."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns


def plot_maxsim_heatmap(
    Q_embeddings: np.ndarray,
    D_embeddings: np.ndarray,
    query_tokens: list[str],
    doc_tokens: list[str],
) -> matplotlib.figure.Figure:
    """Plot token-level similarity heatmap with MaxSim overlay and score decomposition."""

    Q = Q_embeddings / np.linalg.norm(Q_embeddings, axis=1, keepdims=True)
    D = D_embeddings / np.linalg.norm(D_embeddings, axis=1, keepdims=True)

    sim = Q @ D.T

    max_indices = sim.argmax(axis=1)
    max_values = sim.max(axis=1)
    total_score = max_values.sum()

    fig, (ax_heat, ax_bar) = plt.subplots(
        1, 2, figsize=(16, 6),
        gridspec_kw={"width_ratios": [4, 1]},
    )

    sns.heatmap(
        sim, ax=ax_heat,
        xticklabels=doc_tokens,
        yticklabels=query_tokens,
        cmap="coolwarm", center=0,
        vmin=-1, vmax=1,
    )

    for i, j in enumerate(max_indices):
        ax_heat.add_patch(plt.Rectangle(
            (j, i), 1, 1,
            fill=False, edgecolor="red", linewidth=2,
        ))

    ax_heat.set_title("Token-Level Similarity Matrix")
    ax_heat.set_xlabel("Document Tokens")
    ax_heat.set_ylabel("Query Tokens")

    colors = [
        "#2ecc71" if v > 0.3 else "#e74c3c" if v < 0 else "#95a5a6"
        for v in max_values
    ]
    ax_bar.barh(range(len(query_tokens)), max_values, color=colors)
    ax_bar.set_yticks(range(len(query_tokens)))
    ax_bar.set_yticklabels(query_tokens)
    ax_bar.set_xlabel("MaxSim Contribution")
    ax_bar.set_title("Per-Token Score")
    ax_bar.invert_yaxis()

    fig.suptitle(f"ColBERT MaxSim Score: {total_score:.4f}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig
