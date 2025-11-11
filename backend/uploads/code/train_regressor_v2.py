"""
train_regressor_v2.py
-------------------------------------------------------
Version 2: 개선된 회귀 학습 파이프라인
변경 사항:
- 모델 구조 변경 (Dropout 추가)
- 학습률 스케줄러 추가
- validation loss 로그 추가
- random seed 고정
-------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random

# ---------------------------
# 1. 설정 (변경됨)
# ---------------------------
EPOCHS = 30        # v1: 20
BATCH_SIZE = 32     # 동일
LEARNING_RATE = 1e-3  # v1: 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42  # 추가됨

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(SEED)


# ---------------------------
# 2. 데이터 로더
# ---------------------------
def create_dataset(n=200):
    x = np.random.randn(n, 10).astype(np.float32)
    y = (x.sum(axis=1) + np.random.randn(n) * 0.1).astype(np.float32)
    return TensorDataset(torch.tensor(x), torch.tensor(y))

train_ds = create_dataset(200)
val_ds = create_dataset(50)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)


# ---------------------------
# 3. 모델 정의 (변경됨)
# ---------------------------
class Regressor(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # 🔹추가됨
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------
# 4. 학습 함수 (변경됨)
# ---------------------------
def train_model(model, train_loader, val_loader, epochs=EPOCHS):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 🔹추가됨
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x).squeeze()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 🔹 validation 추가
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).squeeze()
                loss = criterion(pred, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"[Epoch {epoch:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        scheduler.step()


# ---------------------------
# 5. 실행
# ---------------------------
if __name__ == "__main__":
    model = Regressor()
    train_model(model, train_loader, val_loader, epochs=EPOCHS)
