import torch
import numpy as np
import os
from configargparse import ArgumentParser
from customnets import ComplexMSELoss, ComplexConvNet
from utils import ii_to_Sx

def load_modelpath(args, epoch, chooseBest=False):
    if chooseBest:
        modelpath = os.path.join(args.root_dir, 'trained_nn_models', args.type, f"model_{args.ii}", "best.pth")
    else:
        modelpath = os.path.join(args.root_dir, 'trained_nn_models', args.type, f"model_{args.ii}", f"epoch_{epoch}.pth")
    return modelpath

def load_test_data(args):
    test_data = np.load(os.path.join(args.root_dir, args.module, 'kspace_input_nn', args.type + '.npy')) # 1000 x 128 x 128 x 4
    test_data = np.transpose(test_data, (0, 3, 1, 2))
    nn_input = torch.from_numpy(test_data)
    return nn_input

def load_val_data(args):
    test_data = np.load(os.path.join(args.root_dir, args.module, 'kspace_input_nn', args.type + '.npy')) # 1000 x 128 x 128 x 4
    test_data = np.transpose(test_data, (0, 3, 1, 2))
    nn_input = torch.from_numpy(test_data)
    target_data = np.load(os.path.join(args.root_dir, args.module, 'groundtruth_nn', args.type, f"W_{args.ii}.npy")) # 4 x 12
    nn_target = torch.from_numpy(target_data)
    return nn_input, nn_target

def main(args):
    model = ComplexConvNet(ii_to_Sx[args.ii])
    criterion = ComplexMSELoss()

    if args.module == "val":
        best_epoch = 20
        best_loss = 100
        for epoch in range(20, 1001, 20):
            modelpath = load_modelpath(args, epoch)
            model.load_state_dict(torch.load(modelpath))
            model.eval()

            nn_input, nn_target = load_val_data(args)
            nn_input_real, nn_input_imag = torch.real(nn_input).float(), torch.imag(nn_input).float()
            nn_target_real, nn_target_imag = torch.real(nn_target).float(), torch.imag(nn_target).float()

            # Make predictions
            with torch.no_grad():
                outputs = model(nn_input_real, nn_input_imag)
                output = outputs[0] + 1j * outputs[1]
                loss = criterion(output, nn_target)
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_epoch = epoch
                print(f"Epoch: {epoch}, Loss: {loss.item()}")
        print(f"Best epoch: {best_epoch}, Best loss: {best_loss}")
        # make a copy of the best model named best.pth
        modelpath = load_modelpath(args, best_epoch)
        model.load_state_dict(torch.load(modelpath))
        best_path = os.path.join(args.root_dir, 'trained_nn_models', args.type, f"model_{args.ii}", "best.pth")
        torch.save(model.state_dict(), best_path)

    elif args.module == "test":
        modelpath = load_modelpath(args, -1, chooseBest=True)
        model.load_state_dict(torch.load(modelpath))
        model.eval()

        nn_input = load_test_data(args)
        nn_input_real, nn_input_imag = torch.real(nn_input).float(), torch.imag(nn_input).float()

        # Make predictions
        with torch.no_grad():
            outputs = model(nn_input_real, nn_input_imag)
            output = outputs[0] + 1j * outputs[1]
            # TODO: Save the output == W == GRAPPA weights for recon
            np.save(os.path.join(args.root_dir, args.module, 'w_output_nn', args.type, f"W_{args.ii}.npy"), output.numpy())

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/storage/jeongjae/128x128/landmark")

    parser.add_argument("--module", type=str, default='val', choices=['val', 'test'])
    parser.add_argument("--ii", type=int, default=1, choices=[1, 2, 6, 7, 13, 14, 15, 16, 17, 18],
                        help="which kernel to train")
    parser.add_argument("--type", type=str, default='type1', choices=['type1', 'type2', 'type3'])
    args = parser.parse_args()
    main(args)
