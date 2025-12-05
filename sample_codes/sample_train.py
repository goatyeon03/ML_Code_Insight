import torch, torch.nn as nn, torch.optim as optim
class TinyNet(nn.Module):
    def __init__(self): 
        super().__init__()
        self.fc = nn.Linear(10, 1)
    def forward(self, x): 
        return self.fc(x)
model = TinyNet()
optimizer = optim.Adam(model.parameters(), lr=3e-4)
batch_size = 8
epochs = 5
