import os
import numpy as np

from configargparse import ArgumentParser
from skimage.util import view_as_windows
from utils import add_args, generate_fake_sensitivity_maps, undersample
from tempfile import NamedTemporaryFile as NTF

def load_imgs(module):
    file_paths, imgs = [], []
    root_directory = './128x128'
    for sub_dir in os.listdir(root_directory):
        if (not sub_dir.endswith(".csv")) and ('DS_Store' not in sub_dir):
            sub_dir_path = os.path.join(os.path.join(root_directory, sub_dir), module)
            npy_files = [f for f in os.listdir(sub_dir_path) if f.endswith(".npy")]
            for filename in npy_files:
                file_path = os.path.join(sub_dir_path, filename)
                img = np.load(file_path)
                file_paths.append(file_path)
                imgs.append(img)
    return file_paths, imgs 

def generate_W_train_data(img, mps, args):
    Ss, Cs = [], []
    imspace = img[..., None]*mps
    imspace = imspace.astype('complex')
    ax = (0, 1)
    kspace = 1/np.sqrt(args.Nx*args.Ny)*np.fft.fftshift(np.fft.fft2(
        np.fft.ifftshift(imspace, axes=ax), axes=ax), axes=ax)
    # TODO: Add noise to kspace
    # Random Gaussian noise, noise at low frequency, noise at high frequency, etc.

    # crop acs_width//2 * Nx window from the center of k-space for calibration
    pd = args.acs_width // 2
    ctr_y = args.Ny // 2
    calib = kspace[:, ctr_y-pd:ctr_y+pd, :].copy()

    undersampled_kspace = undersample(kspace.copy(), args.R)
    kernel_size = (args.k, args.k)

    kx, ky = kernel_size[:]
    kx2, ky2 = kx//2, ky//2
    nc = calib.shape[-1]
    adjx = np.mod(kx, 2)
    adjy = np.mod(ky, 2)
    undersampled_kspace  = np.pad(
        undersampled_kspace, ((kx2, kx2), (ky2, ky2), (0, 0)), mode='constant')
    calib = np.pad(
        calib, ((kx2, kx2), (ky2, ky2), (0, 0)), mode='constant')
    
    mask = np.ascontiguousarray(np.abs(undersampled_kspace[..., 0]) > 0)
    with NTF() as fP:
        P = np.memmap(fP, dtype=mask.dtype, mode='w+', shape=(
            mask.shape[0]-2*kx2, mask.shape[1]-2*ky2, 1, kx, ky))
        P = view_as_windows(mask, (kx, ky))
        Psh = P.shape[:]
        P = P.reshape((-1, kx, ky))

        P, iidx = np.unique(P, return_inverse=True, axis=0)

        validP = np.argwhere(~P[:, kx2, ky2]).squeeze()
        invalidP = np.argwhere(np.all(P == 0, axis=(1, 2)))
        validP = np.setdiff1d(validP, invalidP, assume_unique=True)
        validP = np.atleast_1d(validP)
        P = np.tile(P[..., None], (1, 1, 1, nc))

        for ii in validP:
            idx = np.unravel_index(np.argwhere(iidx == ii), Psh[:2])
            x, y = idx[0]+kx2, idx[1]+ky2
            x = np.atleast_1d(x.squeeze())
            y = np.atleast_1d(y.squeeze())
            S_temp, C_temp = [], []
            for xx, yy in zip(x, y):
                S = undersampled_kspace[xx-kx2:xx+kx2+adjx, yy-ky2:yy+ky2+adjy, :][P[ii, ...]]
                S_temp.append(S)
                C = kspace[xx-kx2, yy-ky2, :]
                C_temp.append(C)
            S = np.vstack(S_temp).T 
            C = np.vstack(C_temp).T
            Ss.append(S)
            Cs.append(C)
    return Ss, Cs, validP.tolist()

def main(args):
    xx = np.linspace(0, 1, args.Nx)
    yy = np.linspace(0, 1, args.Ny)
    x, y = np.meshgrid(xx, yy)
    mps = generate_fake_sensitivity_maps(x, y, args.Nc)

    file_paths, imgs = load_imgs(args.module)
    for filepath, img in zip(file_paths, imgs):
        img_id = filepath.split('.npy')[0]
        # S: undersampled k-space for each ii (12x1 ... 40x7812)
        # C: ground-truth k-space for each ii (4x1 ... 4x7812)
        # ii: [1, 2, 6, 7, 13, 14, 15, 16, 17, 18]
        Ss, Cs, iis = generate_W_train_data(img, mps, args)
        # save Ss and Cs using variant of filepath as filename
        for S, C, ii in zip(Ss, Cs, iis):
            np.save(f"{img_id}_S_{ii}.npy", S)
            np.save(f"{img_id}_C_{ii}.npy", C)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_args(parser)
    parser.add_argument("--module", type=str, default='train',
                        choices=['train', 'val', 'test'])
    parser.add_argument("--n_epochs", type=int, default=1000,
                        help='number of epochs of training')
    args = parser.parse_args()
    main(args)
