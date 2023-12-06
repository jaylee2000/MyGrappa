"""
From curated image dataset (grayscale, .npy files),
1) generate kspace input for neural network. (undersampled, complex-valued)
2) generate ground-truth kspace for other usage. (full-sampled, complex-valued)
"""

from phantominator import shepp_logan
from pygrappa import grappa
from configargparse import ArgumentParser
from utils import add_args, generate_fake_sensitivity_maps, undersample, \
                    multicoil_recon, restore_center
from scipy.ndimage import convolve
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    module = 'test'
    xx = np.linspace(0, 1, 128)
    yy = np.linspace(0, 1, 128)
    x, y = np.meshgrid(xx, yy)
    mps = generate_fake_sensitivity_maps(x, y, 4)

    root_directory = f'/storage/jeongjae/128x128/landmark/{module}/npy_bw'
    file_list = [file for file in os.listdir(root_directory) if file.endswith('.npy')]

    tot_kspaces_type1 = []
    tot_kspaces_type2 = []
    tot_kspaces_type3 = []
    tot_u_kspaces_type1 = []
    tot_u_kspaces_type2 = []
    tot_u_kspaces_type3 = []
    for file in file_list:
        image = np.load(os.path.join(root_directory, file))
        imspace = image[..., None]*mps
        imspace = imspace.astype('complex')
        ax = (0, 1)
        kspace = 1/np.sqrt(128 * 128)*np.fft.fftshift(np.fft.fft2(
            np.fft.ifftshift(imspace, axes=ax), axes=ax), axes=ax)

        # Add noise to kspace
        neighborhood_size = (5, 5)

        # Extract the real part of the complex array for magnitude calculations
        magnitude = np.abs(kspace)

        # Use scipy.ndimage.convolve to compute the local average amplitude
        filter_weights = np.ones(neighborhood_size) / np.prod(neighborhood_size)
        local_average_amplitude = convolve(magnitude, filter_weights[:, :, np.newaxis], mode='reflect')

        # Normalize the Gaussian noise magnitude to the local average amplitude
        normalized_noise_magnitude = 0.1  # Adjust this factor as needed
        real_noise = np.random.normal(scale=normalized_noise_magnitude * local_average_amplitude)
        imag_noise = np.random.normal(scale=normalized_noise_magnitude * local_average_amplitude) * 1j
        noise = real_noise + imag_noise

        kspace_type2 = kspace + noise
        noise_range = slice(74, 105)
        kspace_type3 = kspace.copy()
        kspace_type3[noise_range, :, :] += noise[noise_range, :, :]

        # crop acs_width//2 * Nx window from the center of k-space for calibration
        pd = 10
        ctr_y = 64

        kspace_us_type1 = kspace.copy()
        kspace_us_type2 = kspace_type2.copy()
        kspace_us_type3 = kspace_type3.copy()

        for j in range(kspace.shape[1]):
            if j % 2 != 0:
                if j < ctr_y - pd or j > ctr_y + pd:
                    kspace_us_type1[:, j, :] = 0
                    kspace_us_type2[:, j, :] = 0
                    kspace_us_type3[:, j, :] = 0
        tot_kspaces_type1.append(kspace)
        tot_kspaces_type2.append(kspace_type2)
        tot_kspaces_type3.append(kspace_type3)
        tot_u_kspaces_type1.append(kspace_us_type1)
        tot_u_kspaces_type2.append(kspace_us_type2)
        tot_u_kspaces_type3.append(kspace_us_type3)
    
    kspace_u_stacked1 = np.stack(tot_u_kspaces_type1, axis=0)
    save_directory = f'/storage/jeongjae/128x128/landmark/{module}/kspace_input_nn'
    np.save(os.path.join(save_directory, 'type1.npy'), kspace_u_stacked1)
    kspace_u_stacked2 = np.stack(tot_u_kspaces_type2, axis=0)
    save_directory = f'/storage/jeongjae/128x128/landmark/{module}/kspace_input_nn'
    np.save(os.path.join(save_directory, 'type2.npy'), kspace_u_stacked2)
    kspace_u_stacked3 = np.stack(tot_u_kspaces_type3, axis=0)
    save_directory = f'/storage/jeongjae/128x128/landmark/{module}/kspace_input_nn'
    np.save(os.path.join(save_directory, 'type3.npy'), kspace_u_stacked3)

    kspace_stacked1 = np.stack(tot_kspaces_type1, axis=0)
    save_directory = f'/storage/jeongjae/128x128/landmark/{module}/kspace_fullsampled'
    np.save(os.path.join(save_directory, 'type1.npy'), kspace_stacked1)
    kspace_stacked2 = np.stack(tot_kspaces_type2, axis=0)
    save_directory = f'/storage/jeongjae/128x128/landmark/{module}/kspace_fullsampled'
    np.save(os.path.join(save_directory, 'type2.npy'), kspace_stacked2)
    kspace_stacked3 = np.stack(tot_kspaces_type3, axis=0)
    save_directory = f'/storage/jeongjae/128x128/landmark/{module}/kspace_fullsampled'
    np.save(os.path.join(save_directory, 'type3.npy'), kspace_stacked3)

if __name__ == '__main__':
    main() 