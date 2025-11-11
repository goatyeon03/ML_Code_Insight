import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------
# Dataset / Model Config
# -----------------------------
INPUT_DIM = 128
OUTPUT_DIM = 1
LR = 1e-3
EPOCHS = 10
BATCH_SIZE = 32

# -----------------------------
# Model Definition
# -----------------------------
class EEGRegressor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )

    def forward(self, x):
        return self.net(x)

model = EEGRegressor(INPUT_DIM, OUTPUT_DIM)

# -----------------------------
# Optimizer / Loss
# -----------------------------
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# -----------------------------
# DataLoader (Dummy)
# -----------------------------
x = torch.randn(200, INPUT_DIM)
y = torch.randn(200, OUTPUT_DIM)
loader = DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss / len(loader):.4f}")
