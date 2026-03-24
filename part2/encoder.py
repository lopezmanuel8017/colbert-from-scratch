"""ColBERT encoder: BERT + linear projection + L2 normalization."""

import torch.nn as nn
import torch.nn.functional as F


class ColBERTEncoder(nn.Module):
    """BERT backbone with a no-bias, no-activation linear projection to dim dimensions."""

    def __init__(self, bert_model, dim=128):
        super().__init__()
        self.bert = bert_model
        self.projection = nn.Linear(768, dim, bias=False)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        projected = self.projection(outputs.last_hidden_state)
        return F.normalize(projected, p=2, dim=-1)
