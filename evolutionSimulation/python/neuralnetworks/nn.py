# conda_env: evolution

import torch.nn as nn 
import torch.nn.functional as F


# Simple neural network to model brain of sheep, 2 classes: predator or not
class Brain(nn.Module):
    def __init__(self, name = "brain", in_channels = 1, num_classes=2):
        self.name = name
        super(Brain, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = self.pool(x)           
        x = F.relu(self.conv2(x))  
        x = self.pool(x)           
        x = x.reshape(x.shape[0], -1)  
        x = self.fc1(x)            
        return x
