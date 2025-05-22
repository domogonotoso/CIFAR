import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Unlike MNIST, CIFAR-10 starts with 3 channels, RGB.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  
        self.dropout = nn.Dropout(0.25) 
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))       # (batch, 32, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))       # (batch, 64, 8, 8)
        x = self.dropout(x)
        x = x.view(x.size(0), -1) # Make code more generally. Automatically multiply x.shape except mini batch size.   
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x