# Author's Note

When an analogy to the great work Gödel did, proving that certain systems are fundamentally insufficient, shows up in a new field, it's only natural to make a big deal about it. This time: DeepMind has proven a theoretical upper bound on bi-encoder capacity and shown how retrieval degrades as you approach the limit. Due to this, I have figured that going back to linear algebra is inevitable, and this repo will satisfy the condition that you read linear-algebra-based text once a week. This will be divided into 3 equally delightful pieces, in which we take on the titanic task of trying to make sense of common sense, forcing us to ask why something is better, not just that it is.

Just like the last time, thanks Claude for the docstrings and this beautiful and not-read-by-me README file. Without you, this would be unprofessional, and with you this is immoral.

# ColBERT from Scratch

A ground-up implementation of [ColBERT](https://arxiv.org/abs/2004.12832)'s late interaction retrieval model, built for understanding — not benchmarking.

This project takes ColBERT apart piece by piece. Every design choice is explained. Every tensor gets its shape printed. If you know what a dot product is, you can follow along.

> Most search systems compress an entire document into a single vector. That's like summarizing a 200-page book with a single word. Here's what happens when you don't.

---

## Status

**Part 1 is released.** Parts 2 and 3 are in progress.

| Part | Topic | Status |
|------|-------|--------|
| **1** | What MaxSim does and why it beats single-vector retrieval | **Released** |
| **2** | `[Q]`/`[D]` markers, `[MASK]` query augmentation, the 768→128 projection | Planned |
| **3** | Training from scratch — loss functions, hard negatives, distillation | Planned |

Features that are explicitly **not yet implemented** (and belong in future parts):

- `[Q]` and `[D]` marker tokens
- `[MASK]` query augmentation (32-token padding)
- The 768→128 linear projection layer
- Any training code (loss functions, negatives, distillation)
- ColBERTv2 residual compression
- PLAID / WARP indexing
- Benchmarks against BM25
- ColPali, ColQwen, Jina-ColBERT-v2 extensions

This is intentional. Part 1 focuses on a single question: *what does MaxSim do, and why does it work?* The rest comes later.

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/colbert-from-scratch.git
cd colbert-from-scratch

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -e ".[dev]"
```

Run the notebooks:

```bash
jupyter notebook notebooks/
```

Or open them directly in [Google Colab](https://colab.research.google.com/) — each notebook is self-contained and includes its own dependency installation.

Run the tests:

```bash
pytest tests/ -v
```

---

## What Part 1 Covers

### Notebook 01 — MaxSim in Five Lines

The entire ColBERT scoring function, reduced to its essence:

```python
Q = np.random.randn(8, 128)       # 8 query tokens, 128 dims
D = np.random.randn(20, 128)      # 20 document tokens, 128 dims
similarity = Q @ D.T              # (8, 20) similarity matrix
max_per_query = similarity.max(axis=1)  # best match per query token
score = max_per_query.sum()        # final relevance score
```

No ML, no BERT, no frameworks. Just linear algebra. The notebook walks through each step, introduces L2 normalization (turning dot products into cosine similarities), and builds intuition for why token-level interaction matters.

### Notebook 02 — The Information Bottleneck

Bi-encoders compress every document into a single vector. How far does that get you?

This notebook:
- Builds a toy bi-encoder using `bert-base-uncased` with mean pooling
- Builds token-level MaxSim retrieval with the same model
- Shows both methods scoring 5/5 on a short-document toy dataset — the task is too easy to differentiate them
- Runs a **dilution experiment**: the same answer sentence, buried in increasing amounts of irrelevant filler — scored against a competitive distractor (a document about Python that shares "Python", "programming", and "language" with the query but doesn't answer "who created it"). Both documents are padded with the same filler at each level, so the comparison is fair: any score difference comes from the core content, not document length.
- Shows how the bi-encoder's **margin** between the answer and distractor collapses as filler increases, while MaxSim's margin holds steady

![Dilution experiment](figures/02_dilution_experiment.png)

The math is simple: mean pooling averages all tokens equally, so relevant tokens get diluted and both documents converge toward the same "average filler" vector. MaxSim's $\max$ operator finds the best-matching token regardless of how many irrelevant tokens surround it — the shared filler boosts both scores equally, preserving the discriminative gap. Same model, same embeddings — different architecture, different outcome.

### Notebook 03 — Looking Inside MaxSim

The centerpiece visualization: a token-by-token cosine similarity matrix with MaxSim alignment overlaid.

What the heatmaps reveal:
- **How to read MaxSim** — each red rectangle shows which document token a query token matched. The bar chart decomposes the total score into per-token contributions.

![Heatmap mechanics](figures/03_heatmap_mechanics.png)

- **Disambiguation through token-level voting** — "python" matches strongly in both the programming doc and the snake doc. But "programming", "language", and "created" only find matches in the relevant doc. Those tokens create the entire score gap.

![Disambiguation heatmap](figures/03_heatmap_disambiguation.png)

- **Dilution immunity is visible** — using the same setup from notebook 02's dilution experiment, the padded-document heatmap shows filler tokens as a sea of blue that MaxSim ignores. The red rectangles cluster on the answer sentence regardless of how much noise surrounds it.

![Dilution heatmap](figures/03_heatmap_dilution_padded.png)

The key insight: **MaxSim has zero trainable parameters.** It's just matrix multiplication + max + sum. All the intelligence lives in the embeddings — MaxSim is a lens that makes BERT's token-level knowledge visible and usable for retrieval.

---

## Project Structure

```
colbert-from-scratch/
├── README.md                   ← you are here
├── pyproject.toml
├── requirements.txt
│
├── notebooks/
│   ├── 01_maxsim_in_five_lines.ipynb
│   ├── 02_bow_vs_embeddings.ipynb
│   └── 03_the_heatmap.ipynb
│
├── colbert_from_scratch/       ← importable library
│   ├── __init__.py
│   ├── maxsim.py               ← MaxSim scoring (NumPy + PyTorch)
│   └── viz.py                  ← heatmap and bar chart utilities
│
├── data/
│   └── toy_retrieval.json      ← hand-crafted (lie) 10-document dataset
│
├── figures/                    ← exported PNGs (generated by notebooks)
│
└── tests/
    ├── test_maxsim.py          ← 12 tests for scoring functions
    ├── test_viz.py             ← 4 tests for visualization
    └── test_data.py            ← 11 tests for dataset integrity
```

---

## The Library

The `colbert_from_scratch` package exposes three functions:

### `maxsim_np(Q, D, normalize=True) → float`

NumPy implementation. `Q` is `(N_q, d)`, `D` is `(N_d, d)`. Normalizes by default (cosine similarity). Returns a scalar score.

### `maxsim_torch(Q, D, normalize=True) → torch.Tensor`

PyTorch implementation. Same semantics as the NumPy version. Returns a scalar tensor.

### `plot_maxsim_heatmap(Q_embeddings, D_embeddings, query_tokens, doc_tokens) → Figure`

Produces a two-panel figure:
- **Left:** Token-by-token cosine similarity heatmap (`coolwarm`, centered at 0, range [−1, 1]) with red rectangles marking the MaxSim-selected cell per query token
- **Right:** Per-token score decomposition bar chart

Returns a `matplotlib.figure.Figure` — never calls `plt.show()`.

---

## The Dataset

`data/toy_retrieval.json` contains 10 documents and 5 queries designed to expose specific retrieval phenomena:

| Query | Tests | Bi-encoder struggles? |
|-------|-------|-----------------------|
| q0: "Who created the Python programming language?" | Disambiguation (python language vs. snake) | Yes — "python" appears in both contexts |
| q1: "What is the longest snake species?" | Ranking among multiple snake documents | Sometimes |
| q2: "Where did the creator of Python work?" | Multi-hop reasoning (creator → van Rossum → Google) | Yes |
| q3: "What do machine learning models need?" | Control case — straightforward match | No |
| q4: "Which programming languages are most popular?" | Ranking when multiple docs mention programming | Sometimes |

Each query includes `relevant` document IDs and `hard_negative` IDs — documents that share surface-level keywords but are semantically wrong.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Core array operations |
| `matplotlib` | Plotting |
| `seaborn` | Heatmap rendering |
| `torch` | PyTorch MaxSim implementation |
| `transformers` | BERT tokenizer and model |
| `sentence-transformers` | Optional — cleaner bi-encoder baseline |
| `pytest` | Testing |

All pinned to minor versions. See `requirements.txt` or `pyproject.toml` for exact ranges.

---

## Model

Part 1 uses **`bert-base-uncased`** (110M parameters) exclusively. This is the same base model the original ColBERT was built on. No fine-tuning, no projection layer — just raw BERT token embeddings and MaxSim.

All model inference runs under `torch.no_grad()`. Nothing is trained in Part 1.

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite covers:
- **Scoring correctness** — identity vectors score N_q, orthogonal vectors score 0, hand-computed cases match
- **Normalization behavior** — normalized vs. unnormalized scores differ as expected
- **Cross-implementation consistency** — NumPy and PyTorch produce the same results
- **Visualization contracts** — returns a Figure, has the right axes, never calls `plt.show()`
- **Dataset integrity** — schema validation, referential integrity, no overlap between relevant and hard negative IDs

---

## References

- Khattab & Zaharia, [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832) (SIGIR 2020)
- Formal et al., [A White Box Analysis of ColBERT](https://arxiv.org/abs/2106.00284) (ECIR 2021, Best Short Paper)
- Santhanam et al., [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) (NAACL 2022)
- MacAvaney et al., [Beneath the [MASK]: An Analysis of Structural Query Tokens in ColBERT](https://arxiv.org/abs/2404.08478) (ECIR 2024)
- Weller et al., [On the Theoretical Limitations of Embedding-Based Retrieval](https://arxiv.org/abs/2412.04616) (Google DeepMind, 2025)

---

## License

MIT
