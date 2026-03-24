"""Loss functions and training loop for ColBERT."""

import torch
import torch.nn.functional as F

from colbert_from_scratch.maxsim import maxsim_torch


def colbert_pairwise_loss(query_emb, pos_doc_emb, neg_doc_emb):
    """Pairwise contrastive loss: positive doc should outscore negative doc.

    L = -log( exp(S+) / (exp(S+) + exp(S-)) )
    """
    pos_score = maxsim_torch(query_emb, pos_doc_emb, normalize=False)
    neg_score = maxsim_torch(query_emb, neg_doc_emb, normalize=False)
    scores = torch.stack([pos_score, neg_score]).unsqueeze(0)
    target = torch.tensor([0], device=scores.device)
    return F.cross_entropy(scores, target)


def colbert_in_batch_loss(query_embs, doc_embs):
    """In-batch negatives loss: B queries, B docs, diagonal is positive.

    Builds a B x B MaxSim score matrix and applies cross-entropy
    so that each query's positive doc (same index) scores highest.
    """
    B = len(query_embs)
    device = query_embs[0].device
    scores = torch.zeros(B, B, device=device)
    for i in range(B):
        for j in range(B):
            scores[i, j] = maxsim_torch(query_embs[i], doc_embs[j], normalize=False)
    return F.cross_entropy(scores, torch.arange(B, device=device))


def distillation_loss(student_scores, teacher_scores, temperature=1.0):
    """KL divergence between student and teacher score distributions."""
    student_log_probs = F.log_softmax(student_scores / temperature, dim=0)
    teacher_probs = F.softmax(teacher_scores / temperature, dim=0)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


def train_epoch(model, triples, optimizer, device):
    """Train one epoch over (query, pos_doc, neg_doc) triples.

    Args:
        model: ColBERT model instance
        triples: list of (query_text, pos_doc_text, neg_doc_text)
        optimizer: torch optimizer
        device: torch device

    Returns:
        average loss for the epoch
    """
    model.train()
    total_loss = 0.0

    for query, pos_doc, neg_doc in triples:
        q_emb = model.encode_query(query)
        pos_emb, _ = model.encode_document(pos_doc)
        neg_emb, _ = model.encode_document(neg_doc)

        loss = colbert_pairwise_loss(q_emb, pos_emb, neg_emb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    return total_loss / len(triples)
