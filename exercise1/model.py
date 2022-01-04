import torch.nn.functional as F
from torch import nn

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128,64)
        self.output = nn.Linear(64,10)
        
    def forward(self, input):
        # Hidden layer with sigmoid activation
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        
        # Output layer with softmax activation
        x = F.softmax(self.output(x))
        
        return x