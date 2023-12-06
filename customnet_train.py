import torch
import torch.optim as optim
import numpy as np
import os
import sys
from configargparse import ArgumentParser
from utils import ii_to_Sx
from customnets import ComplexMSELoss, ComplexConvNet

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
    parser.add_argument("--save_model_per", type=int, default=10, help="print loss per n epochs")

    parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")

    parser.add_argument("--ii", type=int, default=1, choices=[1, 2, 6, 7, 13, 14, 15, 16, 17, 18],
                        help="which kernel to train")
    parser.add_argument("--type", type=str, default='type1', choices=['type1', 'type2', 'type3', 'type4', 'type5'])
    args = parser.parse_args()
    main(args)