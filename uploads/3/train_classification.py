import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score

# -----------------------------
# Dataset / Model Config
# -----------------------------
INPUT_DIM = 128
OUTPUT_DIM = 2
LR = 1e-3
EPOCHS = 10
BATCH_SIZE = 32

# -----------------------------
# Model Definition
# -----------------------------
class EEGClassifier(nn.Module):
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

model = EEGClassifier(INPUT_DIM, OUTPUT_DIM)

# -----------------------------
# Optimizer / Loss
# -----------------------------
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# -----------------------------
# DataLoader (Dummy)
# -----------------------------
x = torch.randn(300, INPUT_DIM)
y = torch.randint(0, OUTPUT_DIM, (300,))
loader = DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    preds, trues = [], []
    for xb, yb in loader:
        out = model(xb)
        loss = criterion(out, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds.extend(torch.argmax(out, dim=1).tolist())
        trues.extend(yb.tolist())
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="macro")
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f} | Acc: {acc:.3f} | F1: {f1:.3f}")
