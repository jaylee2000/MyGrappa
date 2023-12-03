import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import glob

from configargparse import ArgumentParser
from utils import root_dir, add_args, ii_to_Sx, ii_to_hidden, ii_to_Sy
from customnets import CustomNet

SUBSET_FACTOR = 20

def get_model(ii):
    return CustomNet(ii_to_Sx(ii), ii_to_hidden(ii), 4 * ii_to_Sx(ii))

def get_npy_filenames_C(args):
    directory_path = os.path.join(os.path.join(os.path.join(root_dir, args.data_dir), args.sub_dir), args.module)
    pattern = f"image*_C_{args.ii}.npy"
    matching_filenames = glob.glob(os.path.join(directory_path, pattern))
    matching_filenames = sorted(matching_filenames)
    return matching_filenames

def get_subset_npy_filenames_C(npy_filenames, t):
    subset_npy_filenames = []
    for i, npy_filename in enumerate(npy_filenames):
        if i % SUBSET_FACTOR == t:
            subset_npy_filenames.append(npy_filename)
    return subset_npy_filenames

def load_npy_files(npy_filenames_C):
    Ss, Cs = [], []
    for npy_filename_C in npy_filenames_C:
        C = np.load(npy_filename_C)
        Cs.append(C)
        S = np.load(npy_filename_C.replace('C', 'S'))
        Ss.append(S)
    return Ss, Cs

def main(args):
    # Create the model, loss function, and optimizer
    model = get_model(args.ii)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    npy_filenames_C = get_npy_filenames_C(args)
    for epoch in range(args.num_epochs):
        for t in range(SUBSET_FACTOR):
            subset_npy_filenames_C = get_subset_npy_filenames_C(npy_filenames_C, t)
            Ss, Cs = load_npy_files(subset_npy_filenames_C)

            for S, C in zip(Ss, Cs):
                # Forward pass
                W = model(S)

                # Compute the loss & do backward pass
                loss = criterion(torch.matmul(W, S) - C, torch.zeros_like(C))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Print the loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{args.num_epochs}], Loss: {loss.item()}')

    # Save the trained model
    torch.save(model.state_dict(), f'trained_model_{args.sub_dir}_{args.ii}.pth')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_args(parser)
    parser.add_argument("--ii", type=int, choices=[1, 2, 6, 7, 13, 14, 15, 16, 17, 18])
    parser.add_argument("--module", type=str, default='train',
                        choices=['train'])
    parser.add_argument("--data_dir", type=str, default='128x128', choices=['128x128'])
    parser.add_argument("--sub_dir", type=str, default='landmark',
                        choices=['apparel', 'artwork', 'cars', 'dishes', 'furniture',
                        'illustrations', 'landmark', 'meme', 'packaged', 'storefronts', 'toys'],
                        help='subdir to load image npy')
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=1000,)
    args = parser.parse_args()
    main(args)
