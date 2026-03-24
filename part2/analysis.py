"""Analysis tools: IDF, token contributions, spectral concentration."""

import math
from collections import Counter

import numpy as np
import torch


def compute_idf(documents, tokenizer):
    """Compute IDF for all tokens across a corpus.

    IDF(t) = log(N / (df(t) + 1)) where df(t) = number of docs containing token t.
    """
    N = len(documents)
    doc_freq = Counter()
    for doc in documents:
        for t in set(tokenizer.tokenize(doc)):
            doc_freq[t] += 1
    return {t: math.log(N / (df + 1)) for t, df in doc_freq.items()}


def token_contributions(query_emb, doc_emb, query_tokens):
    """Compute each query token's MaxSim contribution to the total score.

    Returns dict mapping token -> max similarity value.
    """
    if isinstance(query_emb, torch.Tensor):
        query_emb = query_emb.detach().cpu().numpy()
    if isinstance(doc_emb, torch.Tensor):
        doc_emb = doc_emb.detach().cpu().numpy()

    sim = query_emb @ doc_emb.T
    max_sims = sim.max(axis=1)
    return {tok: float(max_sims[i]) for i, tok in enumerate(query_tokens)}


def mask_vs_real_contribution(query_emb, doc_emb, query_tokens):
    """Split MaxSim score into [MASK] token vs real token contributions.

    Returns (real_score, mask_score, total_score).
    """
    if isinstance(query_emb, torch.Tensor):
        query_emb = query_emb.detach().cpu().numpy()
    if isinstance(doc_emb, torch.Tensor):
        doc_emb = doc_emb.detach().cpu().numpy()

    sim = query_emb @ doc_emb.T
    max_sims = sim.max(axis=1)

    real_score = 0.0
    mask_score = 0.0
    for i, tok in enumerate(query_tokens):
        if tok == "[MASK]":
            mask_score += max_sims[i]
        else:
            real_score += max_sims[i]

    return real_score, mask_score, real_score + mask_score


def term_embedding_concentration(term, documents, _tokenizer, encode_fn, max_samples=50):
    """Measure how stable a term's embedding is across documents using SVD.

    High concentration (~0.8-0.9) = stable direction (exact-match behavior).
    Low concentration (~0.3-0.5) = context-dependent (proxy behavior).

    Args:
        term: the token string to analyze
        documents: list of document strings
        tokenizer: HuggingFace tokenizer
        encode_fn: function that takes text and returns (embeddings, tokens)
        max_samples: cap on embeddings to collect (keeps SVD meaningful)

    Returns:
        fraction of variance captured by first singular value, or None if < 3 occurrences.
    """
    embeddings = []
    for doc in documents:
        if len(embeddings) >= max_samples:
            break
        emb, tok_labels = encode_fn(doc)
        if term not in tok_labels:
            continue
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()
        for i, t in enumerate(tok_labels):
            if t == term and i < len(emb):
                embeddings.append(emb[i])
                break

    if len(embeddings) < 3:
        return None

    stacked = np.stack(embeddings)
    _, S, _ = np.linalg.svd(stacked, full_matrices=False)
    return float(S[0]**2 / (S**2).sum())
