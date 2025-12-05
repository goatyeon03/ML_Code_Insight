import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class ToyTimeSeriesDataset(Dataset):
    """
    Toy 1D signal regression dataset.
    Each sample is (channels, length).
    """
    def __init__(self, num_samples=512, channels=3, length=128):
        super().__init__()
        self.inputs = torch.randn(num_samples, channels, length)
        # Scalar regression target
        self.targets = torch.randn(num_samples, 1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class ConvBackbone1D(nn.Module):
    """
    1D CNN backbone that encodes (C, L) into a feature vector.
    """
    def __init__(self, in_channels=3, hidden_channels=32, kernel_size=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.out_dim = hidden_channels * 2

    def forward(self, x):
        # x: (batch, C, L)
        h = self.net(x)          # (batch, out_dim, 1)
        h = h.squeeze(-1)        # (batch, out_dim)
        return h


class RegressionHead(nn.Module):
    """
    Simple regression head on top of ConvBackbone1D.
    """
    def __init__(self, in_dim=64, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.mlp(features)


class TimeSeriesRegressor(nn.Module):
    """
    Full model = ConvBackbone1D + RegressionHead.
    """
    def __init__(
        self,
        in_channels=3,
        backbone_channels=32,
        head_hidden_dim=64,
    ):
        super().__init__()
        self.backbone = ConvBackbone1D(
            in_channels=in_channels,
            hidden_channels=backbone_channels,
        )
        self.head = RegressionHead(
            in_dim=self.backbone.out_dim,
            hidden_dim=head_hidden_dim,
        )

    def forward(self, x):
        features = self.backbone(x)
        pred = self.head(features)
        return pred


# ------------------------
# Training configuration
# ------------------------
in_channels = 3
batch_size = 64
num_epochs = 15
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = ToyTimeSeriesDataset(
    num_samples=1024,
    channels=in_channels,
    length=128,
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = TimeSeriesRegressor(
    in_channels=in_channels,
    backbone_channels=32,
    head_hidden_dim=64,
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
