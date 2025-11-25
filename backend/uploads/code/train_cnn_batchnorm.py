import torch
import torch.nn as nn
import torch.optim as optim

class BatchNormCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.flatten(1)
        return self.classifier(x)

def train():
    model = BatchNormCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.0008)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 5

if __name__ == "__main__":
    train()
