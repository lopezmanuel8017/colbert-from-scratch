"""MaxSim scoring — NumPy and PyTorch implementations."""

import numpy as np
import torch


def maxsim_np(Q: np.ndarray, D: np.ndarray, normalize: bool = True) -> float:
    """Compute the MaxSim relevance score between query and document embeddings."""
    if normalize:
        Q_norms = np.linalg.norm(Q, axis=1, keepdims=True)
        D_norms = np.linalg.norm(D, axis=1, keepdims=True)
        Q_norms[Q_norms == 0] = 1.0
        D_norms[D_norms == 0] = 1.0
        Q = Q / Q_norms
        D = D / D_norms
    sim = Q @ D.T
    return float(sim.max(axis=1).sum())


def maxsim_torch(Q: torch.Tensor, D: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Compute the MaxSim relevance score between query and document embeddings."""
    if normalize:
        Q = torch.nn.functional.normalize(Q, dim=1)
        D = torch.nn.functional.normalize(D, dim=1)
    sim = Q @ D.T
    return sim.max(dim=1).values.sum()
