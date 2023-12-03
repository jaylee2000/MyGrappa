from phantominator import shepp_logan
from pygrappa import grappa
from configargparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

NUM_COIL_MAX = 4

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

def load_image(image_path, Nx):
    if image_path is None:
        return shepp_logan(Nx)
    else:
        raise NotImplementedError("Image loading not implemented yet.")

def undersample(kspace, R):
    """
    input: kspace (i.e. 128x128 numpy array)
    R: acceleration factor (2, 4, or 8)
    """
    for j in range(kspace.shape[1]):
        if j % R != 0:
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

def main(args):
    xx = np.linspace(0, 1, args.Nx)
    yy = np.linspace(0, 1, args.Ny)
    x, y = np.meshgrid(xx, yy)
    mps = generate_fake_sensitivity_maps(x, y, args.Nc)
    image = load_image(args.image, args.Nx) # ground-truth
    imspace = image[..., None]*mps
    imspace = imspace.astype('complex')
    ax = (0, 1)
    kspace = 1/np.sqrt(args.Nx * args.Ny)*np.fft.fftshift(np.fft.fft2(
        np.fft.ifftshift(imspace, axes=ax), axes=ax), axes=ax)

    # TODO: Add noise to kspace
    # Random Gaussian Noise, Noise at Low frequency, Noise at High frequency etc.

    # crop acs_width//2 * Nx window from the center of k-space for calibration
    pd = args.acs_width // 2
    ctr_y = args.Ny // 2
    calib = kspace[:, ctr_y-pd:ctr_y+pd, :].copy()

    undersampled_kspace = undersample(kspace.copy(), args.R)
    kernel_size = (args.k, args.k)

    res = grappa(undersampled_kspace, calib, kernel_size, coil_axis=-1, lamda=0.01,
                 memmap=False)
    if args.data_consistency:
        res = restore_center(res, calib, ctr_y, pd)
        # TODO: Compute sum of L1 distance at ACS region
    res = np.abs(np.sqrt(args.Nx*args.Ny)*np.fft.fftshift(np.fft.ifft2(
        np.fft.ifftshift(res, axes=ax), axes=ax), axes=ax)) # IFFT
    res0 = multicoil_recon(res, mps, args.Nx, args.Ny)
    plt.imshow(res0, cmap='gray')
    plt.savefig('output_multirecon.png')
    # TODO: Compute SSIM of image and res0

    return

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--Nx", type=int, default=128,
                        help="Number of pixels in x-direction")
    parser.add_argument("--Ny", type=int, default=128,
                        help="Number of pixels in x-direction")
    parser.add_argument("--Nc", type=int, default=4,
                        help="Number of coils")
    parser.add_argument("--image", type=str, default=None,
                        help="path to image used for recon")
    parser.add_argument("--acs_width", type=int, default=20,
                        help="width of Auto Calibration Signal")
    parser.add_argument("--k", type=int, default=5, choices=[3, 5, 7],
                        help="size of GRAPPA kernel")
    parser.add_argument("--R", type=int, default=2, choices=[2, 4, 8],
                        help="acceleration factor")
    parser.add_argument("--data_consistency", type=bool, default=False,
                        help="restore ACS region")
    args = parser.parse_args()
    main(args)
