import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class ToySequenceDataset(Dataset):
    """
    Simple toy dataset for sequence classification.
    Each sample is (seq_len, input_dim).
    """
    def __init__(self, num_samples=1000, seq_len=50, input_dim=16, num_classes=3):
        super().__init__()
        self.inputs = torch.randn(num_samples, seq_len, input_dim)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


class RNNBackbone(nn.Module):
    """
    Backbone that encodes a variable-length sequence into a fixed-size vector.
    """
    def __init__(self, input_dim=16, hidden_dim=64, num_layers=2, bidirectional=False):
        super().__init__()
        self.rnn = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.hidden_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, h_n = self.rnn(x)  # h_n: (num_layers * num_directions, batch, hidden_dim)
        last_layer = h_n[-1]  # (batch, hidden_dim * num_directions)
        return last_layer


class ClassificationHead(nn.Module):
    """
    Simple MLP head that maps backbone features to class logits.
    """
    def __init__(self, in_dim=64, num_classes=3, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, features):
        return self.net(features)


class SequenceClassifier(nn.Module):
    """
    Full model that connects backbone and head.
    """
    def __init__(
        self,
        input_dim=16,
        hidden_dim=64,
        num_layers=2,
        num_classes=3,
        bidirectional=False,
    ):
        super().__init__()
        self.backbone = RNNBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        self.head = ClassificationHead(
            in_dim=self.backbone.hidden_dim,
            num_classes=num_classes,
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits


# ------------------------
# Training configuration
# ------------------------
input_dim = 16
hidden_dim = 64
num_layers = 2
num_classes = 3
batch_size = 32
num_epochs = 10
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = ToySequenceDataset(
    num_samples=512,
    seq_len=50,
    input_dim=input_dim,
    num_classes=num_classes,
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = SequenceClassifier(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_classes=num_classes,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
