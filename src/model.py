
import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitNet(nn.Module):
    def __init__(self):
        super(DigitNet, self).__init__()
        # Input: 1 channel (grayscale), 28x28
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) # 10 digits

    def forward(self, x):
        # x shape: [batch, 1, 28, 28]
        x = self.pool(F.relu(self.conv1(x))) # -> [batch, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x))) # -> [batch, 64, 7, 7]
        x = self.dropout(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
