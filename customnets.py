import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from configargparse import ArgumentParser
from utils import ii_to_Sx

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

def load_train_data(args):
    train_data = np.load(os.path.join(args.root_dir, args.module, 'kspace_input_nn', args.type + '.npy')) # 1000 x 128 x 128 x 4
    train_data = np.transpose(train_data, (0, 3, 1, 2))
    nn_input = torch.from_numpy(train_data)
    target_data = np.load(os.path.join(args.root_dir, args.module, 'groundtruth_nn', args.type, f"W_{args.ii}.npy")) # 4 x 12
    nn_target = torch.from_numpy(target_data)
    return nn_input, nn_target


def main(args):
    logfilepath = os.path.join(args.root_dir, f'training_log_type_{args.type}_ii_{args.ii}.txt')
    log_file = open(logfilepath, 'w')
    sys.stdout = log_file

    nn_input, nn_target = load_train_data(args)
    nn_input_real, nn_input_imag = torch.real(nn_input).float(), torch.imag(nn_input).float()
    nn_target_real, nn_target_imag = torch.real(nn_target).float(), torch.imag(nn_target).float()

    model = ComplexConvNet(ii_to_Sx[args.ii]) # init model
    criterion = ComplexMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    batch_size = args.batch_size
    for epoch in range(args.n_epochs):
        for i in range(0, 1000, batch_size):
            # Batch data
            inputs = nn_input[i:i+batch_size]
            inputs_real, inputs_imag = nn_input_real[i:i+batch_size], nn_input_imag[i:i+batch_size]
            targets = nn_target[i:i+batch_size]
            targets_real, targets_imag = nn_target_real[i:i+batch_size], nn_target_imag[i:i+batch_size]

            outputs = model(inputs_real, inputs_imag) # forward pass
            output = outputs[0] + 1j * outputs[1]

            loss = criterion(output, targets) # calculate loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % args.print_loss_per == 0:
            print(f'Epoch {epoch+1}/{args.n_epochs}, Loss: {loss.item()}')
        # save model
        if (epoch+1) % args.save_model_per == 0:
            savepath = os.path.join(args.root_dir, 'trained_nn_models', args.type, f"model_{args.ii}", f"epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), savepath)
            print(f"Saved model epoch: {epoch+1}")
    log_file.close()
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--module", type=str, default='train', choices=['train'])
    parser.add_argument("--root_dir", type=str, default="/storage/jeongjae/128x128/landmark")
    parser.add_argument("--batch_size", type=int, default=50, help="size of the batches")

    parser.add_argument("--print_loss_per", type=int, default=10, help="print loss per n epochs")
    parser.add_argument("--save_model_per", type=int, default=20, help="print loss per n epochs")

    parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")

    parser.add_argument("--ii", type=int, default=1, choices=[1, 2, 6, 7, 13, 14, 15, 16, 17, 18],
                        help="which kernel to train")
    parser.add_argument("--type", type=str, default='type1', choices=['type1', 'type2', 'type3'])
    args = parser.parse_args()
    main(args)