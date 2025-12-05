import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class ToyTextDataset(Dataset):
    """
    Toy tokenized text dataset for classification.
    input_ids: (seq_len,)
    """
    def __init__(self, num_samples=1000, seq_len=64, vocab_size=5000, num_classes=4):
        super().__init__()
        self.inputs = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


class TokenEmbedding(nn.Module):
    """
    Token + positional embedding layer.
    """
    def __init__(self, vocab_size=5000, d_model=128, max_len=256):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        tok = self.token_embed(input_ids)
        pos = self.pos_embed(positions)
        return tok + pos


class TransformerBackbone(nn.Module):
    """
    Transformer encoder backbone that produces a sequence of hidden states.
    """
    def __init__(
        self,
        vocab_size=5000,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        max_len=256,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size=vocab_size, d_model=d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (batch, seq_len)
        x = self.embedding(input_ids)  # (batch, seq_len, d_model)
        if attention_mask is not None:
            # attention_mask: (batch, seq_len) with 1 for valid tokens
            mask = attention_mask == 0  # True where padding
        else:
            mask = None
        h = self.encoder(x, src_key_padding_mask=mask)
        # simple [CLS]-style pooling: take first token
        cls_hidden = h[:, 0, :]
        return cls_hidden


class TextClassificationHead(nn.Module):
    """
    Classification head on top of Transformer backbone.
    """
    def __init__(self, d_model=128, num_classes=4, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, features):
        return self.mlp(features)


class TextClassifier(nn.Module):
    """
    Full model = TransformerBackbone + TextClassificationHead.
    """
    def __init__(
        self,
        vocab_size=5000,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        max_len=256,
        num_classes=4,
        dropout=0.1,
    ):
        super().__init__()
        self.backbone = TransformerBackbone(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            max_len=max_len,
            dropout=dropout,
        )
        self.head = TextClassificationHead(
            d_model=d_model,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, input_ids, attention_mask=None):
        features = self.backbone(input_ids, attention_mask=attention_mask)
        logits = self.head(features)
        return logits


# ------------------------
# Training configuration
# ------------------------
vocab_size = 5000
d_model = 128
num_classes = 4
batch_size = 16
num_epochs = 8
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = ToyTextDataset(
    num_samples=512,
    seq_len=64,
    vocab_size=vocab_size,
    num_classes=num_classes,
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = TextClassifier(
    vocab_size=vocab_size,
    d_model=d_model,
    num_layers=2,
    num_classes=num_classes,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
