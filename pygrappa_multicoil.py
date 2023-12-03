from phantominator import shepp_logan
from pygrappa import grappa
from configargparse import ArgumentParser
from utils import add_args, generate_fake_sensitivity_maps, undersample, \
                    multicoil_recon, restore_center

import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path, Nx):
    if image_path is None:
        return shepp_logan(Nx)
    else:
        raise NotImplementedError("Image loading not implemented yet.")

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
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)
