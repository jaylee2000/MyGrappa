"""
Extract 'groundtruth Ws' from entire kspace data
'Groundtruth Ws' are used as groundtruth for training neural network
that learns to map
input: undersampled kspace --> output: W
"""


from tempfile import NamedTemporaryFile as NTF
from skimage.util import view_as_windows
from masks_precompute import masks
from utils import generate_fake_sensitivity_maps
from pygrappa_multicoil import load_image
from configargparse import ArgumentParser
import numpy as np
import os

def extract_W_groundtruth(calib, kernel_size=(5, 5), coil_axis=-1, lamda=0.01):
    # Put the coil dimension at the end
    calib = np.moveaxis(calib, coil_axis, -1)

    # Get shape of kernel
    kx, ky = kernel_size[:]
    kx2, ky2 = int(kx/2), int(ky/2)
    nc = calib.shape[-1]

    # Pad calib kspace data
    calib = np.pad(
        calib, ((kx2, kx2), (ky2, ky2), (0, 0)), mode='constant')

    Ws = {}
    # Store windows in temporary files so we don't overwhelm memory
    with NTF() as fA:
        # Get all overlapping patches of ACS
        try:
            A = np.memmap(fA, dtype=calib.dtype, mode='w+', shape=(
                calib.shape[0]-2*kx, calib.shape[1]-2*ky, 1, kx, ky, nc))
            A[:] = view_as_windows(
                calib, (kx, ky, nc)).reshape((-1, kx, ky, nc))
        except ValueError:
            A = view_as_windows(
                calib, (kx, ky, nc)).reshape((-1, kx, ky, nc))
        for ii in [1, 2, 6, 7, 13, 14, 15, 16, 17, 18]:
            S = A[:, masks[ii]]
            T = A[:, kx2, ky2, :]
            ShS = S.conj().T @ S
            ShT = S.conj().T @ T
            lamda0 = lamda*np.linalg.norm(ShS)/ShS.shape[0]
            W = np.linalg.solve(
                ShS + lamda0*np.eye(ShS.shape[0]), ShT).T
            Ws[ii] = W

    return Ws

def main(args):
    xx = np.linspace(0, 1, 128)
    yy = np.linspace(0, 1, 128)
    x, y = np.meshgrid(xx, yy)
    mps = generate_fake_sensitivity_maps(x, y, 4)

    base_dir = '/storage/jeongjae/128x128/landmark'
    module = args.module # 'val' or 'test'
    npy_folder = 'kspace_fullsampled'
    thistype = args.thistype

    kspace_tot = np.load(os.path.join(base_dir, module, npy_folder, thistype + '.npy'))
    W_dict = {}
    for i in range(kspace_tot.shape[0]):
        kspace = kspace_tot[i, ...]
        Ws = extract_W_groundtruth(kspace)
        for ii in [1, 2, 6, 7, 13, 14, 15, 16, 17, 18]:
            W = Ws[ii]
            if ii not in W_dict:
                W_dict[ii] = [W]
            else:
                W_dict[ii].append(W)
    
    for ii in [1, 2, 6, 7, 13, 14, 15, 16, 17, 18]:
        W = np.stack(W_dict[ii], axis=0)
        save_folder = os.path.join(base_dir, module, 'groundtruth_nn', thistype)
        np.save(os.path.join(save_folder, f'W_{ii}.npy'), W)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--module", type=str, default='train', choices=['train', 'val'])
    parser.add_argument("--thistype", type=str, default='type4',
                        choices=['type1', 'type2', 'type3', 'type4', 'type5'])
    args = parser.parse_args()
    main(args)