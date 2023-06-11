from torch import nn, optim
import numpy as np

from datasets import train_data

class simpleDCNN(nn.Modules):
    def __init__(self, num_classes, max_length=50):
        super(simpleDCNN, self).__innit__()
        self.conv1 = nn.Conv1d(max_length, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        self.fc - nn.Linear(64 * 25, num_classes)
    
    def forward(self, x):
        x = x.transpose(1, 2) # swap frewuency and time dimensions
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
num_classes = len(np.unique(train_data.keys()))
model = simpleDCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)