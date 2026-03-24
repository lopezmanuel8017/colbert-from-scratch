"""Query and document tokenization with ColBERT special tokens."""

import string
import torch


Q_TOKEN_ID = 1
D_TOKEN_ID = 2
MASK_TOKEN_ID = 103


def tokenize_query(text, tokenizer, max_length=32):
    """Tokenize a query with [Q] marker and [MASK] padding.

    Format: [CLS] [Q] <query tokens> [MASK]...  [SEP]
    Padded to exactly max_length tokens.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)

    max_tokens = max_length - 3
    tokens = tokens[:max_tokens]

    n_masks = max_length - 3 - len(tokens)

    input_ids = (
        [tokenizer.cls_token_id]
        + [Q_TOKEN_ID]
        + tokens
        + [MASK_TOKEN_ID] * n_masks
        + [tokenizer.sep_token_id]
    )

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": torch.tensor([input_ids]),
        "attention_mask": torch.tensor([attention_mask]),
    }


def tokenize_query_with_n_masks(text, tokenizer, n_masks):
    """Tokenize a query with a specific number of [MASK] tokens (for sweeps)."""
    tokens = tokenizer.encode(text, add_special_tokens=False)

    input_ids = (
        [tokenizer.cls_token_id]
        + [Q_TOKEN_ID]
        + tokens
        + [MASK_TOKEN_ID] * n_masks
        + [tokenizer.sep_token_id]
    )

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": torch.tensor([input_ids]),
        "attention_mask": torch.tensor([attention_mask]),
    }


def tokenize_document(text, tokenizer, max_length=180):
    """Tokenize a document with [D] marker.

    Format: [CLS] [D] <doc tokens> [SEP]
    Truncated to max_length.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)

    max_tokens = max_length - 3
    tokens = tokens[:max_tokens]

    input_ids = (
        [tokenizer.cls_token_id]
        + [D_TOKEN_ID]
        + tokens
        + [tokenizer.sep_token_id]
    )

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": torch.tensor([input_ids]),
        "attention_mask": torch.tensor([attention_mask]),
    }


def filter_doc_tokens(embeddings, input_ids, tokenizer):
    """Remove [CLS], [SEP], [PAD], [D], and punctuation token embeddings from a document."""
    skip_ids = {
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
        D_TOKEN_ID,
    }

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    keep = []
    for i, (tid, tok) in enumerate(zip(input_ids, tokens)):
        if tid in skip_ids:
            continue
        if all(c in string.punctuation for c in tok):
            continue
        keep.append(i)

    return embeddings[keep], [tokens[i] for i in keep]


def get_token_labels(text, tokenizer, mode="query", max_length=32, n_masks=None):
    """Get human-readable token labels for a tokenized sequence."""
    if mode == "query":
        if n_masks is not None:
            encoded = tokenize_query_with_n_masks(text, tokenizer, n_masks)
        else:
            encoded = tokenize_query(text, tokenizer, max_length)
    else:
        encoded = tokenize_document(text, tokenizer, max_length)

    ids = encoded["input_ids"].squeeze(0).tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)

    labels = []
    for t in tokens:
        if t == "[unused0]":
            labels.append("[Q]")
        elif t == "[unused1]":
            labels.append("[D]")
        else:
            labels.append(t)
    return labels
