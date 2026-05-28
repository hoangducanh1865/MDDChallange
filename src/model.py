from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Local path overrides for pretrained models
# ---------------------------------------------------------------------------

BACKBONE_PATHS = {
    "wav2vec2-base-100h":            "facebook/wav2vec2-base-100h",
    "wav2vec2-base-vietnamese-250h": "vinai/wav2vec2-base-vietnamese-250h",
    "hubert-base-ls960":             "./models/hubert-base-ls960",
}


# ---------------------------------------------------------------------------
# Sub-modules copied verbatim from reference (operate on 768-dim features)
# ---------------------------------------------------------------------------

class PhoneCNNStack(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.Conv2d  = nn.Conv2d(1, 1, 3, 1, 1)
        self.reLU    = nn.ReLU()
        self.drop_out= nn.Dropout(p=0.2)
        self.bn      = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.Conv2d(x)
        x = x.squeeze(1)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.reLU(x)
        x = self.drop_out(x)
        return x


class PhoneRNNStack(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.reLU    = nn.ReLU()
        self.drop_out= nn.Dropout(p=0.2)
        self.bn      = nn.BatchNorm1d(hidden_dim)
        self.bilstm  = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim // 2,
            bidirectional=True, batch_first=True,
        )

    def forward(self, x):
        x, _ = self.bilstm(x)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.drop_out(x)
        return x


class PhoneticEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.CNN = PhoneCNNStack(hidden_dim=hidden_dim)
        self.RNN = PhoneRNNStack(hidden_dim=hidden_dim)

    def forward(self, x):
        return self.RNN(self.CNN(x))


class LinguisticEncoder(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        emb_size = max(vocab_size + 1, 256)
        self.embedding = nn.Embedding(emb_size, 64, padding_idx=0)
        self.bilstm    = nn.LSTM(input_size=64, hidden_size=64, bidirectional=True, batch_first=True)
        self.fc_key    = nn.Linear(128, 2304)
        self.fc_val    = nn.Linear(128, 2304)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.long()
        x = self.embedding(x)
        o, _ = self.bilstm(x)
        return self.fc_key(o), self.fc_val(o)  # key, value for attention


# ---------------------------------------------------------------------------
# Unified MDDModel (supports Wav2Vec2 and HuBERT backbones)
# ---------------------------------------------------------------------------

class MDDModel(nn.Module):
    def __init__(self, backbone_name: str, vocab_size: int, hidden_dim: int = 768):
        super().__init__()
        self.vocab_size   = vocab_size
        self.hidden_dim   = hidden_dim
        self.backbone_key = backbone_name

        pretrained_path = BACKBONE_PATHS.get(backbone_name, backbone_name)
        self._is_hubert  = "hubert" in backbone_name.lower()

        if self._is_hubert:
            from transformers import HubertModel
            self.backbone = HubertModel.from_pretrained(pretrained_path)
        else:
            from transformers import Wav2Vec2Model
            self.backbone = Wav2Vec2Model.from_pretrained(pretrained_path)

        self.phonetic_enc = PhoneticEncoder(hidden_dim=hidden_dim)
        self.linguistic_enc = LinguisticEncoder(vocab_size=vocab_size)
        self.multihead_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True, kdim=2304, vdim=2304,
        )
        self.ctc_head    = nn.Linear(hidden_dim * 2, vocab_size)
        self.binary_head = nn.Linear(hidden_dim, 2)

    def freeze_feature_extractor(self):
        if self._is_hubert:
            self.backbone.feature_extractor._freeze_parameters()
        else:
            self.backbone.feature_extractor._freeze_parameters()

    def get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        # Wav2Vec2 / HuBERT CNN feature extractor: 7 conv layers, strides [5,2,2,2,2,2,2]
        lengths = input_lengths.clone().float()
        for stride in [5, 2, 2, 2, 2, 2, 2]:
            lengths = torch.floor((lengths - 1) / stride + 1)
        return lengths.long()

    def forward(self, input_values: torch.Tensor, linguistic: torch.Tensor):
        # Backbone
        backbone_out = self.backbone(input_values)
        phonetic = backbone_out[0]  # (B, T, 768)

        # Phonetic encoder (CNN + BiLSTM)
        phonetic = self.phonetic_enc(phonetic)  # (B, T, 768)

        # Linguistic encoder
        h_key, h_val = self.linguistic_enc(linguistic)  # each (B, L, 2304)

        # Cross-attention: query=phonetic, key/value=linguistic
        attn_out, _ = self.multihead_attn(phonetic, h_key, h_val)  # (B, T, 768)

        # CTC head
        fused     = torch.cat([attn_out, phonetic], dim=2)  # (B, T, 1536)
        ctc_logits= self.ctc_head(fused)                    # (B, T, vocab_size)

        # Auxiliary utterance-level binary head (mean-pool over time)
        pooled       = phonetic.mean(dim=1)                 # (B, 768)
        binary_logits= self.binary_head(pooled)             # (B, 2)

        return ctc_logits, binary_logits


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class EnsembleModel(nn.Module):
    def __init__(self, models: List[MDDModel], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        total = sum(weights)
        self.weights = [w / total for w in weights]

    def forward(self, input_values: torch.Tensor, linguistic: torch.Tensor):
        weighted_logits = None
        for model, w in zip(self.models, self.weights):
            ctc_logits, _ = model(input_values, linguistic)
            if weighted_logits is None:
                weighted_logits = w * ctc_logits
            else:
                weighted_logits = weighted_logits + w * ctc_logits
        return weighted_logits


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_model(backbone_name: str, vocab_size: int, device: torch.device) -> MDDModel:
    model = MDDModel(backbone_name=backbone_name, vocab_size=vocab_size)
    model.freeze_feature_extractor()
    model = model.to(device)
    return model
