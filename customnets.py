import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ComplexMSELoss(nn.Module):
    def __init__(self):
        super(ComplexMSELoss, self).__init__()

    def forward(self, input_complex, target_complex):
        # Calculate MSE for real and imaginary parts separately
        mse_real = nn.functional.mse_loss(input_complex.real.float(), target_complex.real.float())
        mse_imag = nn.functional.mse_loss(input_complex.imag.float(), target_complex.imag.float())

        # Combine real and imaginary MSE
        mse_combined = mse_real + mse_imag

        return mse_combined.float()

class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 5), stride=2, padding=2):
        super(ComplexConv2d, self).__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_realimag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_imagreal = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x_real, x_imag):
        real_output = self.conv_real(x_real) + self.conv_realimag(x_imag)
        imag_output = self.conv_imagreal(x_real) + self.conv_imag(x_imag)
        return real_output, imag_output

class ComplexConvNet(nn.Module):
    def __init__(self, output_size):
        super(ComplexConvNet, self).__init__()
        # Input size: 128x128x4
        # Output size: 4xoutput_size # btw 4x12 and 4x40
        self.output_size = output_size
        self.conv1 = ComplexConv2d(4, 4, kernel_size=(5, 5), stride=2, padding=2) # Output size: 64x64x4
        self.conv2 = ComplexConv2d(4, 4, kernel_size=(5, 5), stride=2, padding=2) # Output size: 32x32x4
        self.conv3 = ComplexConv2d(4, 4, kernel_size=(5, 5), stride=2, padding=2) # Output size: 16x16x4
        self.conv4 = ComplexConv2d(4, 4, kernel_size=(5, 5), stride=2, padding=2) # Output size: 8x8x4
        self.fc_real = nn.Linear(8 * 8 * 4, 4 * output_size)
        self.fc_imag = nn.Linear(8 * 8 * 4, 4 * output_size)

    def forward(self, x_real, x_imag):
        x_real1, x_imag1 = self.conv1(x_real, x_imag)
        x_real1, x_imag1 = nn.functional.relu(x_real1), nn.functional.relu(x_imag1)
        x_real2, x_imag2 = self.conv2(x_real1, x_imag1)
        x_real2, x_imag2 = nn.functional.relu(x_real2), nn.functional.relu(x_imag2)
        x_real3, x_imag3 = self.conv3(x_real2, x_imag2)
        x_real3, x_imag3 = nn.functional.relu(x_real3), nn.functional.relu(x_imag3)
        x_real4, x_imag4 = self.conv4(x_real3, x_imag3)
        x_real4, x_imag4 = nn.functional.relu(x_real4), nn.functional.relu(x_imag4)

        x_real4, x_imag4 = x_real4.reshape(-1, 8 * 8 * 4), x_imag4.reshape(-1, 8 * 8 * 4)
        x_real = self.fc_real(x_real4)
        x_imag = self.fc_imag(x_imag4)

        x_real = x_real.view(-1, 4, self.output_size)
        x_imag = x_imag.view(-1, 4, self.output_size)
        return x_real, x_imag

if __name__ == '__main__':
    # Initialize the model
    model = ComplexConvNet(12)

    # Loss function
    criterion = ComplexMSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Dummy input and target
    train_data = np.load('/storage/jeongjae/128x128/landmark/train/kspace_input_nn/type1.npy') # 1000 X 128 X 128 X 4
    train_data = np.transpose(train_data, (0, 3, 1, 2))
    my_input = torch.from_numpy(train_data)
    my_input_real, my_input_imag = torch.real(my_input).float(), torch.imag(my_input).float()

    target_data = np.load('/storage/jeongjae/128x128/landmark/train/groundtruth_nn/type1/W_1.npy') # 4 X 12
    my_target = torch.from_numpy(target_data)
    my_target_real, my_target_imag = torch.real(my_target).float(), torch.imag(my_target).float()

    # Training loop
    num_epochs = 100
    batch_size = 50
    for epoch in range(num_epochs):
        for i in range(0, 1000, batch_size):
            # Batch data
            inputs = my_input[i:i+batch_size]
            inputs_real, inputs_imag = my_input_real[i:i+batch_size], my_input_imag[i:i+batch_size]
            targets = my_target[i:i+batch_size]
            targets_real, targets_imag = my_target_real[i:i+batch_size], my_target_imag[i:i+batch_size]

            # Forward pass
            outputs = model(inputs_real, inputs_imag)

            output = outputs[0] + 1j * outputs[1]
            # Calculate loss
            loss = criterion(output, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
