from time import time
from tempfile import NamedTemporaryFile as NTF
from skimage.util import view_as_windows
from phantominator import shepp_logan
from pygrappa import grappa

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Generate fake sensitivity maps: mps
    N = 128
    ncoils = 4
    xx = np.linspace(0, 1, N)
    x, y = np.meshgrid(xx, xx)
    mps = np.zeros((N, N, ncoils))
    mps[..., 0] = x**2
    mps[..., 1] = 1 - x**2
    mps[..., 2] = y**2
    mps[..., 3] = 1 - y**2

    # generate 4 coil phantom
    ph = shepp_logan(N)
    imspace = ph[..., None]*mps
    # save imspace as png
    plt.imshow(np.abs(imspace[..., 0]), cmap='gray')
    plt.savefig('imspace.png')
    imspace = imspace.astype('complex')
    ax = (0, 1)
    kspace = 1/np.sqrt(N**2)*np.fft.fftshift(np.fft.fft2(
        np.fft.ifftshift(imspace, axes=ax), axes=ax), axes=ax)
    # kspace = np.floor(1000*np.real(kspace)) # temp

    # crop 20x20 window from the center of k-space for calibration
    pd = 10
    ctr = int(N/2)
    calib = kspace[ctr-pd:ctr+pd, ctr-pd:ctr+pd, :].copy()

    # calibrate a kernel
    kernel_size = (5, 5)

    # undersample by a factor of 2 in both kx and ky
    kspace[::2, 1::2, :] = 0
    kspace[1::2, ::2, :] = 0

    # reconstruct:
    res = grappa(
        kspace, calib, kernel_size, coil_axis=-1, lamda=0.01,
        memmap=False)

    # Take a look
    res = np.abs(np.sqrt(N**2)*np.fft.fftshift(np.fft.ifft2(
        np.fft.ifftshift(res, axes=ax), axes=ax), axes=ax))

    res0 = np.zeros((N, N))
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            mhat = np.ndarray.flatten(res[i, j, :])
            mhat = mhat.reshape(len(mhat), 1)
            cvec = np.ndarray.flatten(mps[i, j, :])
            cvec = cvec.reshape(len(cvec), 1)
            res0[i, j] = np.linalg.pinv(cvec.T @ cvec) @ cvec.T @ mhat

    plt.imshow(res0, cmap='gray')
    plt.savefig('output_multirecon_legacy.png')

    res1 = np.zeros((2*N, 2*N))
    kk = 0
    for idx in np.ndindex((2, 2)):
        ii, jj = idx[:]
        res1[ii*N:(ii+1)*N, jj*N:(jj+1)*N] = res[..., kk]
        kk += 1
    plt.imshow(res1, cmap='gray')
    plt.savefig('output_each_coil_recon.png')
