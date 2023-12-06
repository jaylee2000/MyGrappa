import torch.nn as nn

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
        self.conv5 = ComplexConv2d(4, 4, kernel_size=(5, 5), stride=2, padding=2) # Output size: 4x4x4
        self.fc_real = nn.Linear(4 * 4 * 4, 4 * output_size)
        self.fc_imag = nn.Linear(4 * 4 * 4, 4 * output_size)

    def forward(self, x_real, x_imag):
        x_real1, x_imag1 = self.conv1(x_real, x_imag)
        x_real1, x_imag1 = nn.functional.relu(x_real1), nn.functional.relu(x_imag1)
        x_real2, x_imag2 = self.conv2(x_real1, x_imag1)
        x_real2, x_imag2 = nn.functional.relu(x_real2), nn.functional.relu(x_imag2)
        x_real3, x_imag3 = self.conv3(x_real2, x_imag2)
        x_real3, x_imag3 = nn.functional.relu(x_real3), nn.functional.relu(x_imag3)
        x_real4, x_imag4 = self.conv4(x_real3, x_imag3)
        x_real4, x_imag4 = nn.functional.relu(x_real4), nn.functional.relu(x_imag4)
        x_real5, x_imag5 = self.conv5(x_real4, x_imag4)
        x_real5, x_imag5 = nn.functional.relu(x_real5), nn.functional.relu(x_imag5)

        x_real5, x_imag5 = x_real5.reshape(-1, 4 * 4 * 4), x_imag5.reshape(-1, 4 * 4 * 4)
        x_real = self.fc_real(x_real5)
        x_imag = self.fc_imag(x_imag5)

        x_real = x_real.view(-1, 4, self.output_size)
        x_imag = x_imag.view(-1, 4, self.output_size)
        return x_real, x_imag
