from configargparse import ArgumentParser
from utils import generate_fake_sensitivity_maps, multicoil_recon, ii_to_Sx, undersample
from tempfile import NamedTemporaryFile as NTF
from skimage.util import view_as_windows
from customnets import ComplexConvNet
import numpy as np
import os
import torch
from scipy.stats import rankdata
import time
import matplotlib.pyplot as plt
import sys


# read kspace undersampled input
N = 128
ax = (0, 1)
kernel_size = (5, 5)
ctr_y = 64
pd = 10
lamda = 0.01

def get_masks(ii: int):
    if ii == 1:
        return np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0]
        ])
    elif ii == 13:
        return np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ])
    elif ii == 6:
        return np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0]
        ])
    elif ii == 14:
        return np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ])
    elif ii == 15:
        return np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0]
        ])
    elif ii == 2:
        return np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0]
        ])
    elif ii == 16:
        return np.array([
            [0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ])
    elif ii == 7:
        return np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0]
        ])
    elif ii == 17:
        return np.array([
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ])
    elif ii == 18:
        return np.array([
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0]
        ])
    else:
        raise ValueError('Invalid mask ii')

def do_grappa(kspace, calib, Ws):
    """
    Return: 
        1) Ws = GRAPPA weights
        2) reconstructed kspace
    """
    if Ws is None:
        Ws = {}
    # Get shape of kernel
    kx, ky = kernel_size[:]
    kx2, ky2 = int(kx/2), int(ky/2)
    nc = calib.shape[-1]
    adjx = np.mod(kx, 2)
    adjy = np.mod(ky, 2)

    # Pad kspace data
    kspace = np.pad(
        kspace, ((kx2, kx2), (ky2, ky2), (0, 0)), mode='constant')
    calib = np.pad(
        calib, ((kx2, kx2), (ky2, ky2), (0, 0)), mode='constant')

    # All coils have same sampling pattern, choose the 0th one arbitrarily for mask
    mask = np.ascontiguousarray(np.abs(kspace[..., 0]) > 0)

    # Store windows in temporary files so we don't overwhelm memory
    with NTF() as fP, NTF() as fA, NTF() as frecon:
        # Get all overlapping patches from the mask
        P = np.memmap(fP, dtype=mask.dtype, mode='w+', shape=(
            mask.shape[0]-2*kx2, mask.shape[1]-2*ky2, 1, kx, ky))
        P = view_as_windows(mask, (kx, ky))
        Psh = P.shape[:]  # save shape for unflattening indices later
        P = P.reshape((-1, kx, ky))

        # Find the unique patches and associate them with indices
        P, iidx = np.unique(P, return_inverse=True, axis=0)

        # Filter out geometries that don't have a hole at the center.
        # These are all the kernel geometries we actually need to
        # compute weights for.
        validP = np.argwhere(~P[:, kx2, ky2]).squeeze()

        # We also want to ignore empty patches
        invalidP = np.argwhere(np.all(P == 0, axis=(1, 2)))
        validP = np.setdiff1d(validP, invalidP, assume_unique=True)

        # Make sure validP is iterable
        validP = np.atleast_1d(validP)

        # Give P back its coil dimension
        P = np.tile(P[..., None], (1, 1, 1, nc))

        # Get all overlapping patches of ACS
        try:
            A = np.memmap(fA, dtype=calib.dtype, mode='w+', shape=(
                calib.shape[0]-2*kx, calib.shape[1]-2*ky, 1, kx, ky, nc))
            A[:] = view_as_windows(
                calib, (kx, ky, nc)).reshape((-1, kx, ky, nc))
        except ValueError:
            A = view_as_windows(
                calib, (kx, ky, nc)).reshape((-1, kx, ky, nc))

        # Initialize recon array
        recon = np.memmap(
            frecon, dtype=kspace.dtype, mode='w+',
            shape=kspace.shape)

        # Train weights and apply them for each valid hole we have in
        # kspace data:
        for ii in validP:
            # Formulate the problem so we avoid
            # computing the inverse, use numpy.linalg.solve, and
            # Tikhonov regularization for better conditioning:
            #     SW = T
            #     S^HSW = S^HT
            #     W = (S^HS)^-1 S^HT
            #  -> W = (S^HS + lamda I)^-1 S^HT
            S = A[:, P[ii, ...]]
            T = A[:, kx2, ky2, :]
            ShS = S.conj().T @ S
            ShT = S.conj().T @ T
            lamda0 = lamda*np.linalg.norm(ShS)/ShS.shape[0]
            if ii not in Ws.keys():
                W = np.linalg.solve(
                    ShS + lamda0*np.eye(ShS.shape[0]), ShT).T
                Ws[ii] = W
            else:
                W = Ws[ii]

            # Apply the weights
            # Find all holes corresponding to current geometry.
            # x, y define where top left corner is, so move to ctr.
            # make sure they are iterable by enforcing atleast_1d
            idx = np.unravel_index(
                np.argwhere(iidx == ii), Psh[:2])
            x, y = idx[0]+kx2, idx[1]+ky2
            x = np.atleast_1d(x.squeeze())
            y = np.atleast_1d(y.squeeze())
            for xx, yy in zip(x, y):
                # Collect sources for this hole and apply weights
                S = kspace[xx-kx2:xx+kx2+adjx, yy-ky2:yy+ky2+adjy, :]
                S = S[P[ii, ...]]
                recon[xx, yy, :] = (W @ S[:, None]).squeeze()
        return Ws, (recon[:] + kspace)[kx2:-kx2, ky2:-ky2, :]

def do_recon(kspace_input, calib_input, Ws, mps):
    """
    Return: W, res
    W = GRAPPA weights
    res = reconstructed image (after multicoil_recon)
    """
    Ws, res_k = do_grappa(kspace_input, calib_input, Ws)
    res = np.abs(np.sqrt(N*N)*np.fft.fftshift(np.fft.ifft2(
        np.fft.ifftshift(res_k, axes=ax), axes=ax), axes=ax)) # IFFT
    res0 = multicoil_recon(res, mps, N, N)
    return Ws, res0

def load_b2_Ws(args, lss_type):
    lss_root_dir = os.path.join(args.root_dir, 'train', 'trained_weights', args.type)
    lss_dir = os.path.join(lss_root_dir, f'lss_{lss_type}')
    Ws = {}
    for ii in [1, 2, 6, 7, 13, 14, 15, 16, 17, 18]:
        Ws[ii] = np.load(os.path.join(lss_dir, f'W_{ii}.npy'))
    return Ws

def infer_from_model(model, kspace_input):
    test_data = np.transpose(kspace_input , (2, 0, 1))
    nn_input = torch.from_numpy(test_data)
    nn_input_real, nn_input_imag = torch.real(nn_input).float(), torch.imag(nn_input).float()
    outputs = model(nn_input_real, nn_input_imag)
    output = outputs[0] + 1j * outputs[1]
    return output.detach().numpy()

def load_b3_Ws(b3_nns, kspace_input):
    Ws = {}
    for ii in [1, 2, 6, 7, 13, 14, 15, 16, 17, 18]:
        model = b3_nns[ii]
        Ws[ii] = infer_from_model(model, kspace_input)
    return Ws

def get_psnr(original, reconstructed, max_val=1.0):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')  # PSNR is infinite if the images are identical
    psnr_value = 10 * np.log10((max_val ** 2) / mse)
    return psnr_value

def print_statistics(b1_psnrs, b21_psnrs, b22_psnrs, b23_psnrs, b3_psnrs, meta_psnrs):
    # print mean, median
    print(f'b1_psnrs: mean {np.mean(b1_psnrs)}, median {np.median(b1_psnrs)}')
    print(f'b21_psnrs: mean {np.mean(b21_psnrs)}, median {np.median(b21_psnrs)}')
    print(f'b22_psnrs: mean {np.mean(b22_psnrs)}, median {np.median(b22_psnrs)}')
    print(f'b23_psnrs: mean {np.mean(b23_psnrs)}, median {np.median(b23_psnrs)}')
    print(f'b3_psnrs: mean {np.mean(b3_psnrs)}, median {np.median(b3_psnrs)}')
    print(f'meta_psnrs: mean {np.mean(meta_psnrs)}, median {np.median(meta_psnrs)}')
    return

def print_avg_rank(b1_psnrs, b21_psnrs, b22_psnrs, b23_psnrs, b3_psnrs, meta_psnrs):
    # Generate 6 x len(array) array
    mtx = np.array([b1_psnrs, b21_psnrs, b22_psnrs, b23_psnrs, b3_psnrs, meta_psnrs])
    mtx[np.isinf(mtx)] = np.nanmax(mtx) + 1 # replace inf with max + 1

    ranks = np.apply_along_axis(lambda x: rankdata(-x, method='ordinal'), axis=0, arr=mtx)
    average_ranks = np.mean(ranks, axis=1)
    print(f'average_ranks: {average_ranks}')

def get_consistency(kspace_input, calib_input, Ws):
    # Using the 5 sets of 'Ws', compute data consistency in the ACS (calib_input)
    # return the Ws that are most consistent with the ACS
    consistency = 0
    for ii in [1, 2, 6, 7, 13, 14, 15, 16, 17, 18]:
        W = Ws[ii] # 4x12
        mask = get_masks(ii)
        mask = np.tile(mask[:, :, np.newaxis], (1, 1, 4))
        for i in range(2, 126):
            for j in range(2, 18):
                tmp = calib_input[i-2:i+3, j-2:j+3, :] * mask
                tmp_flatten = tmp.reshape((25, 4))
                zero_indices = np.where(tmp_flatten[:,0] == 0)[0]
                zeromask = np.ones(25, dtype=bool)
                zeromask[zero_indices] = False
                tmp_trunc = tmp_flatten[zeromask]
                # make it 1D
                tmp_trunc = tmp_trunc.reshape(-1)
                pred = W @ tmp_trunc # 4x1
                consistency += np.sum(np.abs(pred - calib_input[i, j, :]) ) # 불일치도
    return -consistency # 음의 부호 가해줘서 consistency가 클수록 좋도록

def get_Ws_most_consistent(kspace_input, calib_input, b1_Ws, b21_Ws, b22_Ws, b23_Ws, b3_Ws):
    # Using the 5 sets of 'Ws', compute data consistency in the ACS (calib_input)
    # return the Ws that are most consistent with the ACS
    b1_consistency = get_consistency(kspace_input, calib_input, b1_Ws)
    b21_consistency = get_consistency(kspace_input, calib_input, b21_Ws)
    b22_consistency = get_consistency(kspace_input, calib_input, b22_Ws)
    b23_consistency = get_consistency(kspace_input, calib_input, b23_Ws)
    b3_consistency = get_consistency(kspace_input, calib_input, b3_Ws)
    idx = np.argmax([b1_consistency, b21_consistency, b22_consistency, b23_consistency, b3_consistency])
    if idx == 0:
        return b1_Ws
    elif idx == 1:
        return b21_Ws
    elif idx == 2:
        return b22_Ws
    elif idx == 3:
        return b23_Ws
    elif idx == 4:
        return b3_Ws

def load_b3_nns(args):
    models = {}
    b3_nns_base_dir = os.path.join(args.root_dir, 'trained_nn_models', args.type)
    for ii in [1, 2, 6, 7, 13, 14, 15, 16, 17, 18]:
        b3_nns_dir = os.path.join(b3_nns_base_dir, f'model_{ii}')
        model = ComplexConvNet(ii_to_Sx[ii])
        model.load_state_dict(torch.load(os.path.join(b3_nns_dir, 'best.pth')))
        model.eval()
        models[ii] = model
    return models

def save_sample(b1_res0, b21_res0, b22_res0, b23_res0, b3_res0, meta_res0, image_groundtruth, args):
    plt.imsave(os.path.join(args.root_dir, 'sample', args.type, 'b1_res0.png'), b1_res0, cmap='gray')
    plt.imsave(os.path.join(args.root_dir, 'sample', args.type, 'b21_res0.png'), b21_res0, cmap='gray')
    plt.imsave(os.path.join(args.root_dir, 'sample', args.type, 'b22_res0.png'), b22_res0, cmap='gray')
    plt.imsave(os.path.join(args.root_dir, 'sample', args.type, 'b23_res0.png'), b23_res0, cmap='gray')
    plt.imsave(os.path.join(args.root_dir, 'sample', args.type, 'b3_res0.png'), b3_res0, cmap='gray')
    plt.imsave(os.path.join(args.root_dir, 'sample', args.type, 'meta_res0.png'), meta_res0, cmap='gray')
    plt.imsave(os.path.join(args.root_dir, 'sample', args.type, 'image_groundtruth.png'), image_groundtruth, cmap='gray')


def main(args):
    logfilepath = os.path.join(args.root_dir, f'finaleval_log_type_{args.type}.txt')
    log_file = open(logfilepath, 'w')
    sys.stdout = log_file

    # sensitivity mapping
    xx = np.linspace(0, 1, N)
    yy = np.linspace(0, 1, N)
    x, y = np.meshgrid(xx, yy)
    mps = generate_fake_sensitivity_maps(x, y, 4)


    # read kspace undersampled input
    kspace_inputs = np.load(os.path.join(args.input_file, args.type + '.npy'))
    groundtruth_kspaces = np.load(os.path.join(args.groundtruth_file, args.type + '.npy'))
    b1_psnrs, b21_psnrs, b22_psnrs, b23_psnrs, b3_psnrs, meta_psnrs = \
        [], [], [], [], [], []

    b21_Ws = load_b2_Ws(args, 1)
    b22_Ws = load_b2_Ws(args, 10)
    b23_Ws = load_b2_Ws(args, 100)

    b3_nns = load_b3_nns(args)

    for i in range(kspace_inputs.shape[0]):
        # if i % 10 != 0: # COMMENT THIS OUT FOR FINAL RESULT
        #     continue
        kspace_input = kspace_inputs[i, :, :, :]
        u_kspace_input = undersample(kspace_input.copy(), 2)
        calib_input = kspace_input[:, ctr_y-pd:ctr_y+pd, :].copy()

        groundtruth_kspace = groundtruth_kspaces[i, :, :, :]
        image_groundtruth = np.abs(np.sqrt(N * N)*np.fft.fftshift(np.fft.ifft2(
        np.fft.ifftshift(groundtruth_kspace, axes=ax), axes=ax), axes=ax))
        image_groundtruth = multicoil_recon(image_groundtruth, mps, N, N)

        b1_Ws, b1_res0 = do_recon(u_kspace_input, calib_input, None, mps)
        b21_Ws, b21_res0 = do_recon(u_kspace_input, calib_input, b21_Ws, mps)
        b22_Ws, b22_res0 = do_recon(u_kspace_input, calib_input, b22_Ws, mps)
        b23_Ws, b23_res0 = do_recon(u_kspace_input, calib_input, b23_Ws, mps)
        b3_Ws, b3_res0 = do_recon(u_kspace_input, calib_input, load_b3_Ws(b3_nns, kspace_input), mps)
        meta_Ws = get_Ws_most_consistent(u_kspace_input, calib_input, b1_Ws, b21_Ws, b22_Ws, b23_Ws, b3_Ws)
        meta_Ws, meta_res0 = do_recon(u_kspace_input, calib_input, meta_Ws, mps)

        b1_psnr = get_psnr(image_groundtruth, b1_res0)
        b21_psnr = get_psnr(image_groundtruth, b21_res0)
        b22_psnr = get_psnr(image_groundtruth, b22_res0)
        b23_psnr = get_psnr(image_groundtruth, b23_res0)
        b3_psnr = get_psnr(image_groundtruth, b3_res0)
        meta_psnr = get_psnr(image_groundtruth, meta_res0)

        b1_psnrs.append(b1_psnr)
        b21_psnrs.append(b21_psnr)
        b22_psnrs.append(b22_psnr)
        b23_psnrs.append(b23_psnr)
        b3_psnrs.append(b3_psnr)
        meta_psnrs.append(meta_psnr)

    print_statistics(b1_psnrs, b21_psnrs, b22_psnrs, b23_psnrs, b3_psnrs, meta_psnrs)
    print_avg_rank(b1_psnrs, b21_psnrs, b22_psnrs, b23_psnrs, b3_psnrs, meta_psnrs)
    save_sample(b1_res0, b21_res0, b22_res0, b23_res0, b3_res0, meta_res0, image_groundtruth, args)
    log_file.close()
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root_dir", type=str, default='/storage/jeongjae/128x128/landmark')
    parser.add_argument("--input_file", type=str, default='/storage/jeongjae/128x128/landmark/test/kspace_input_nn')
    parser.add_argument("--groundtruth_file", type=str, default='/storage/jeongjae/128x128/landmark/test/kspace_fullsampled')

    parser.add_argument("--type", type=str, default='type1', choices=['type1', 'type2', 'type3'])

    parser.add_argument("--module", type=str, default="test", choices=["test"])
    args = parser.parse_args()
    main(args)