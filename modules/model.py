from torch import nn

from datasets import train_data, train_dataloader

class simpleDCNN(nn.Module):
    def __init__(self, num_classes, max_length=50):
        super(simpleDCNN, self).__init__()
        self.conv1 = nn.Conv1d(max_length, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 96, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(384, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = x.transpose(1, 2) # swap frequency and time dimensions
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(self.relu4(x))
        return x
