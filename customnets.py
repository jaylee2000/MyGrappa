import torch
import torch.nn as nn


class CustomNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()  # You can choose a different activation function if needed
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Ensure input has the correct shape (Sx, Sy)
        x = x.view(x.size(0), -1)
        
        # Apply first linear layer and activation function
        x = self.fc1(x)
        x = self.relu(x)
        
        # Apply second linear layer
        output = self.fc2(x)

        # Ensure output has the correct shape (4, Sx)
        output = output.view(4, -1)

        return output
