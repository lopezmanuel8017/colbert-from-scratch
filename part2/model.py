"""ColBERT model: wraps encoder with query/document tokenization."""

import torch.nn as nn

from part2.encoder import ColBERTEncoder
from part2.tokenize import (
    tokenize_query,
    tokenize_document,
    filter_doc_tokens,
)


class ColBERT(nn.Module):
    """Full ColBERT model with tokenization, encoding, and doc filtering."""

    def __init__(self, bert_model, tokenizer, dim=128, max_query_len=32, max_doc_len=180):
        super().__init__()
        self.encoder = ColBERTEncoder(bert_model, dim=dim)
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len

    def encode_query(self, text):
        """Encode a single query string. Returns (seq_len, dim) tensor."""
        encoded = tokenize_query(text, self.tokenizer, self.max_query_len)
        encoded = {k: v.to(next(self.parameters()).device) for k, v in encoded.items()}
        return self.encoder(**encoded).squeeze(0)

    def encode_document(self, text):
        """Encode a single document string with punctuation filtering. Returns (filtered_len, dim) tensor and token labels."""
        encoded = tokenize_document(text, self.tokenizer, self.max_doc_len)
        encoded = {k: v.to(next(self.parameters()).device) for k, v in encoded.items()}
        emb = self.encoder(**encoded).squeeze(0)
        input_ids = encoded["input_ids"].squeeze(0).tolist()
        filtered_emb, filtered_tokens = filter_doc_tokens(emb, input_ids, self.tokenizer)
        return filtered_emb, filtered_tokens

    def encode_queries_batch(self, texts):
        """Encode a list of queries. Returns list of (seq_len, dim) tensors."""
        return [self.encode_query(t) for t in texts]

    def encode_documents_batch(self, texts):
        """Encode a list of documents. Returns list of (filtered_len, dim) tensors."""
        return [self.encode_document(t) for t in texts]

    def freeze_bert(self, n_unfreeze_layers=4):
        """Freeze all BERT layers except the top n_unfreeze_layers and the projection."""
        for param in self.encoder.bert.parameters():
            param.requires_grad = False

        layers = self.encoder.bert.encoder.layer
        for layer in layers[-n_unfreeze_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        for param in self.encoder.projection.parameters():
            param.requires_grad = True

        frozen = sum(1 for p in self.parameters() if not p.requires_grad)
        trainable = sum(1 for p in self.parameters() if p.requires_grad)
        print(f"Frozen: {frozen} params, Trainable: {trainable} params")
