from configargparse import ArgumentParser
import numpy as np

NUM_COIL_MAX = 4
root_dir = '/Users/jaylee/Documents/2023_2/DSP/FinalProject/MyGrappa'

ii_to_Sx = {
    1: 12,
    2: 24,
    6: 16,
    7: 32,
    13: 12,
    14: 16,
    15: 20,
    16: 24,
    17: 32,
    18: 40
}

# need experimenting
ii_to_hidden = {
    1: 12,
    2: 24,
    6: 16,
    7: 32,
    13: 12,
    14: 16,
    15: 20,
    16: 24,
    17: 32,
    18: 40,
}

ii_to_Sy = {
    1: 1,
    2: 63,
    6: 1,
    7: 63,
    13: 1,
    14: 1,
    15: 124,
    16: 63,
    17: 63,
    18: 7812
}

def generate_fake_sensitivity_maps(x, y, Nc):
    mps = np.zeros((x.shape[0], y.shape[0], Nc))
    if Nc > 0:
        mps[..., 0] = x**2
    if Nc > 1:
        mps[..., 1] = 1 - x**2
    if Nc > 2:
        mps[..., 2] = y**2
    if Nc > 3:
        mps[..., 3] = 1 - y**2
    if Nc > NUM_COIL_MAX:
        raise NotImplementedError(
            f"Only up to {NUM_COIL_MAX} coils supported for now.")
    return mps

def undersample(kspace, R):
    """
    input: kspace (i.e. 128x128 numpy array)
    R: acceleration factor (2, 4, or 8)
    """
    for j in range(kspace.shape[1]):
        if j % R != 0:
            # if j < ctr_y-pd or j >= ctr_y+pd:
            kspace[:, j] = 0
    return kspace

def multicoil_recon(res, mps, Nx, Ny):
    res0 = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            mhat = np.ndarray.flatten(res[i, j, :])
            mhat = mhat.reshape(len(mhat), 1)
            cvec = np.ndarray.flatten(mps[i, j, :])
            cvec = cvec.reshape(len(cvec), 1)
            res0[i, j] = np.linalg.pinv(cvec.T @ cvec) @ cvec.T @ mhat
    return res0

def restore_center(res, calib, ctr_y, pd):
    res[:, ctr_y-pd:ctr_y+pd, :] = calib
    return res

def add_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--Nx", type=int, default=128, choices=[128],
                        help="Number of pixels in x-direction")
    parser.add_argument("--Ny", type=int, default=128, choices=[128],
                        help="Number of pixels in x-direction")
    parser.add_argument("--Nc", type=int, default=4, choices=[4],
                        help="Number of coils")
    parser.add_argument("--image", type=str, default=None,
                        help="path to image used for recon")
    parser.add_argument("--acs_width", type=int, default=20, choices=[20],
                        help="width of Auto Calibration Signal")
    parser.add_argument("--k", type=int, default=5, choices=[3, 5, 7],
                        help="size of GRAPPA kernel")
    parser.add_argument("--R", type=int, default=2, choices=[2, 4, 8],
                        help="acceleration factor")
    parser.add_argument("--data_consistency", type=bool, default=False,
                        help="restore ACS region")
    return parser
