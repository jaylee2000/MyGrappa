"""
Used to generate 'learned' Ws (b-2)
"""

from tempfile import NamedTemporaryFile as NTF
from skimage.util import view_as_windows
from masks_precompute import masks
from utils import generate_fake_sensitivity_maps
import numpy as np
import os

def extract_W_groundtruth(calibs, kernel_size=(5, 5), coil_axis=-1, lamda=0.01):
    S_dict = {}
    T_dict = {}
    Ws = {}
    for i in range(calibs.shape[0]):
        calib = calibs[i, :, :, :]
        # Put the coil dimension at the end
        calib = np.moveaxis(calib, coil_axis, -1)

        # Get shape of kernel
        kx, ky = kernel_size[:]
        kx2, ky2 = int(kx/2), int(ky/2)
        nc = calib.shape[-1]

        # Pad calib kspace data
        calib = np.pad(
            calib, ((kx2, kx2), (ky2, ky2), (0, 0)), mode='constant')

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
                if ii not in S_dict:
                    S_dict[ii] = S
                else:
                    S_dict[ii] = np.concatenate((S_dict[ii], S), axis=0)
                if ii not in T_dict:
                    T_dict[ii] = T
                else:
                    T_dict[ii] = np.concatenate((T_dict[ii], T), axis=0)
    for ii in [1, 2, 6, 7, 13, 14, 15, 16, 17, 18]:
        S = S_dict[ii]
        T = T_dict[ii]
        ShS = S.conj().T @ S
        ShT = S.conj().T @ T
        lamda0 = lamda*np.linalg.norm(ShS)/ShS.shape[0]
        W = np.linalg.solve(
            ShS + lamda0*np.eye(ShS.shape[0]), ShT).T
        Ws[ii] = W

    return Ws

if __name__ == "__main__":
    # datatype = 1 # 2, 3
    # lsstype = 100 # 1, 10, 100

    # for lsstype in [1, 10, 100]:
    for lsstype in [1, 10, 100]:
        for datatype in [1, 2, 3]:
            root_directory = '/storage/jeongjae/128x128/landmark'
            module = 'train' # 'train', 'val', 'test'
            under = 'kspace_temp'
            typek = f'type{datatype}_{lsstype}.npy'
            folder = os.path.join(root_directory, module, under, typek)
            kspace = np.load(folder)

            Ws = extract_W_groundtruth(kspace)
            save_folder = os.path.join(root_directory, module, 'trained_weights', f'type{datatype}', f'lss_{lsstype}')
            for ii in [1, 2, 6, 7, 13, 14, 15, 16, 17, 18]:
                np.save(os.path.join(save_folder, 'W_{}.npy'.format(ii)), Ws[ii])
            print(f"complete lss_{lsstype} data_{datatype}")
