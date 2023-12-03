import torch
import torch.nn as nn

class ComplexLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(ComplexLinear, self).__init__()
        self.real_fc = nn.Linear(input_size, output_size)
        self.imag_fc = nn.Linear(input_size, output_size)

    def forward(self, x_real, x_imag):
        real_part = self.real_fc(x_real) - self.imag_fc(x_imag)
        imag_part = self.real_fc(x_imag) + self.imag_fc(x_real)
        return real_part, imag_part

class CustomNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomNet, self).__init__()
        self.fc1 = ComplexLinear(input_size, hidden_size)
        self.relu = nn.ReLU()  # You can choose a different activation function if needed
        self.fc2 = ComplexLinear(hidden_size, output_size)

    def forward(self, x_real, x_imag):
        # Ensure input has the correct shape (Sx, Sy)
        x_real = x_real.view(x_real.size(0), -1)
        x_imag = x_imag.view(x_imag.size(0), -1)

        # Apply first complex linear layer and activation function
        x1_real, x1_imag = self.fc1(x_real, x_imag)
        x1_real = self.relu(x1_real)
        x1_imag = self.relu(x1_imag)

        # Apply second complex linear layer
        output_real, output_imag = self.fc2(x1_real, x1_imag)

        # Ensure output has the correct shape (4, Sx)
        output_real = output_real.view(4, -1)
        output_imag = output_imag.view(4, -1)

        return output_real, output_imag
