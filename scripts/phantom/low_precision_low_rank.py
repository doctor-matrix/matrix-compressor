# This is a script to generate low-precision low-rank approximations for matrices

import math
import pathlib
from typing import Tuple
import numpy as np
from numpy import random
from phantominator import shepp_logan
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# plt.rcParams["figure.figsize"] = [20, 18]
plt.rcParams.update({"font.size": 10})

SEED = int(datetime.now().timestamp())


def sign_1bit(X: np.ndarray = None):
    """
    :param X: Input matrix
    :return: Element-wise signs (1 if 0)
    """
    return np.where(X > 0, 1, -1)


def quantize(X: np.ndarray = None, B: int = 16):
    """
    Element-wise matrix quantization for general bit-budget
    :param X: Matrix to be quantized with entries in [-1,1]
    :param B: Bit-budget per coordinate for quantization
    :return: Quantized matrix
    """

    M = 2**B  # No. of quantization points per dimension
    res = 2 / (M - 1)  # Resolution

    # Quantize each coordinate with a scalar quantizer of unit dynamic range
    fst_pt = -1  # First quantization point
    L_idx = np.floor((X - fst_pt) / res)  # Lower index for each entry
    L = fst_pt + L_idx * res  # Matrix of lower quantization points
    U = fst_pt + (L_idx + 1) * res  # Matrix of upper quantization points

    # Nearest neighbor quantization
    Q = np.zeros_like(X)
    Q[X < -1] = -1  # Value less than lower limit of dynamic range
    Q[X > 1] = 1  # Value more than upper limit of dynamic range
    mask0 = np.abs(X) <= 1  # Value within dynamic range
    mask = np.abs(X - L) <= res / 2
    Q[mask * mask0] = L[mask * mask0]
    mask = np.abs(U - X) <= res / 2
    Q[mask * mask0] = U[mask * mask0]

    return Q


def normalize_wrt_inner_prod(X: np.ndarray = None, Y: np.ndarray = None):
    """
    :param X: Target matrix
    :param Y: Approximation matrix
    :return: Best approximation of X up to scaling of Y
    """
    scale = np.sum(np.multiply(Y, X)) / np.sum(np.multiply(Y, Y))
    return scale * Y


def normalize_and_shift_wrt_inner_prod(X: np.ndarray = None, Y: np.ndarray = None):
    """
    :param X: Target matrix
    :param Y: Approximation matrix
    :return: Best approximation of X up to scaling of Y
    """

    assert X.shape == Y.shape, "Dimension mismatch!"

    n = X.shape[0]
    d = X.shape[1]
    M = np.ones_like(X)

    t1 = 1 / (np.linalg.norm(Y, ord="fro") ** 2 - np.sum(np.multiply(Y, M)) / (n * d))
    t2 = np.sum(np.multiply(X, Y)) - np.sum(np.multiply(Y, M)) * np.sum(
        np.multiply(X, M)
    ) / (n * d)
    alpha = t1 * t2

    beta = (np.sum(np.multiply(X, M)) - np.sum(np.multiply(Y, M)) * alpha) / (n * d)

    return alpha * Y + beta * M


def direct_svd_quant(X: np.ndarray = None, r: int = None, B1: int = 8, B2: int = 8):
    """
    Compute the full SVD and naively quantize each low rank factor
    :param X: Target matrix in [0, 1]^{n x d}
    :param r: Target rank (intrinsic rank)
    :param B1: Bit-budget for the first low-rank factor
    :param B2: Bit-budget for the second low-rank factor
    :return: A (naive) low-precision low-rank approximation of X
    """

    # Compute full SVD
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    U = U[:, 0:r]
    S = S[0:r]
    VT = VT[0:r, :]

    # Normalize and quantize the first low-rank factor
    Z = U @ np.diag(S)
    Z = quantize(Z, B=B1)

    # Normalize and quantize the second low-rank factor
    W = VT
    W = quantize(W, B=B2)

    return normalize_and_shift_wrt_inner_prod(X, Z @ W)


def lplr(X: np.ndarray = None, r: int = None, B1: int = 8, B2: int = 8):
    """
    :param X: Target matrix in [0, 1]^{n x d}
    :param r: Inherent rank
    :param B1: Bit-budget for first low-rank factor
    :param B2: Bit-budget for second low-rank factor
    :return: Low-precision Low-rank approximation
    """

    random.seed(SEED)

    # Sketch the column space of X with S matrix
    S = np.random.randn(X.shape[1], r) / np.sqrt(r)  # Gaussian sketching matrix

    # Quantize the sketched matrix and get the first low-rank factor
    Z = quantize(X=X @ S, B=B1)

    # Get the second low-rank factor
    W = np.linalg.pinv(Z) @ X
    W = quantize(W, B=B2)

    # Return the scaled and shifted output
    return normalize_and_shift_wrt_inner_prod(X, Z @ W)


def lplr_noshift(X: np.ndarray = None, r: int = None, B1: int = 8, B2: int = 8):
    """
    :param X: Target matrix in [0, 1]^{n x d}
    :param r: Inherent rank
    :param B1: Bit-budget for first low-rank factor
    :param B2: Bit-budget for second low-rank factor
    :return: Low-precision Low-rank approximation
    """

    random.seed(SEED)

    # Sketch the column space of X with S matrix
    S = np.random.randn(X.shape[1], r) / np.sqrt(r)  # Gaussian sketching matrix

    # Quantize the sketched matrix and get the first low-rank factor
    Z = quantize(X=X @ S, B=B1)

    # Get the second low-rank factor
    W = np.linalg.pinv(Z) @ X
    W = quantize(W, B=B2)

    # Return the scaled and shifted output
    return normalize_wrt_inner_prod(X, Z @ W)


def lplr_sign(X: np.ndarray = None, r: int = None):
    """
    This function computes the low-precision low-rank approximation.
    :param X: Input matrix
    :param r: Target rank (Intrinsic rank)
    :return: The low-precision low-rank approximation of X
    """

    random.seed(SEED)

    # Sketch the column space of X with S matrix
    S = np.random.randn(X.shape[1], r) / np.sqrt(r)  # Gaussian sketching matrix

    # Quantize the sketched matrix and get the first low-rank factor
    Z = sign_1bit(X @ S)

    # Get the second low-rank factor
    W = sign_1bit(np.linalg.pinv(Z) @ X)

    # Return the scaled output
    return normalize_wrt_inner_prod(X, Z @ W)


def lplr_sign_shift(X: np.ndarray = None, r: int = None):
    """
    This function computes the low-precision low-rank approximation.
    :param X: Input matrix
    :param r: Target rank (Intrinsic rank)
    :return: The low-precision low-rank approximation of X
    """

    random.seed(SEED)

    # Sketch the column space of X with S matrix
    S = np.random.randn(X.shape[1], r) / np.sqrt(r)  # Gaussian sketching matrix

    # Normalize the matrix to the range [-1, 1]
    XS = X @ S
    XS = np.interp(XS, (XS.min(), XS.max()), (-1, 1))

    # Quantize the sketched matrix and get the first low-rank factor
    Z = sign_1bit(XS)

    # Get the second low-rank factor
    W = sign_1bit(np.linalg.pinv(Z) @ X)

    # Return the scaled and shifted output
    return normalize_and_shift_wrt_inner_prod(X, Z @ W)


def lplr_semi_quant(X: np.ndarray = None, r: int = None, p: int = 0):
    """
    This function computes the low-precision low-rank approximation but does not quantize the second factor
    :param X: Input matrix
    :param r: Target rank (Intrinsic rank)
    :param p: Oversampling factor
    :return: The low-precision low-rank approximation of X
    """

    random.seed(10)

    # Sketch the column space of X with S matrix
    ny = X.shape[1]
    S = np.random.randn(ny, r + p) / np.sqrt(r + p)  # Gaussian sketching matrix

    # Quantize the sketched matrix and get the first low-rank factor
    Z = sign_1bit(X @ S)

    # Get the second low-rank factor
    W_opt = np.linalg.pinv(Z) @ X

    X_app = Z @ W_opt
    X_app = normalize_and_shift_wrt_inner_prod(X, X_app)

    return X_app


def rsvd(X: np.ndarray = None, r: int = None, p: int = 0):
    """
    This function computes the low-rank approximation using randomized SVD
    :param X: Input matrix
    :param r: Target rank (Intrinsic rank)
    :param p: Oversampling factor
    :return: The randomized low-rank approximation of X
    """

    # random.seed(10)

    # Sketch the column space of X with S matrix
    ny = X.shape[1]
    S = np.random.randn(ny, r + p) / np.sqrt(r + p)  # Gaussian sketching matrix

    # Quantize the sketched matrix and get the first low-rank factor
    Z = X @ S

    # Get the second low-rank factor
    W_opt = np.linalg.pinv(Z) @ X

    X_app = Z @ W_opt
    X_app = normalize_wrt_inner_prod(X, X_app)

    return X_app


def iterative_lplr(
    X: np.ndarray = None, r: int = None, K: int = None, B1: int = 8, B2: int = 8
):
    """
    :param X: Target matrix in [0, 1]^{n x d}
    :param r: Target rank (per-iteration)
    :param K: Total number of iterations (Target rank = r * K)
    :param B1: Bit-budget for first low-rank factor
    :param B2: Bit-budget for second low-rank factor
    :return: Low-precision Low-rank approximation
    """

    Xres = np.copy(X)
    X_app = np.zeros_like(X)

    for k in range(K):
        Xq = lplr(X=Xres, r=r, B1=B1, B2=B2)
        X_app += Xq
        Xres -= Xq

        res_err = np.linalg.norm(X_app - X, ord="fro") / np.linalg.norm(X, ord="fro")
        print("Residual error after iteration {}: {}".format(k, res_err))

    return X_app


def iterative_vs_oneshot_lplr_comparison():
    # Load phantom image
    P = shepp_logan(1000)
    P = np.interp(P, (P.min(), P.max()), (0, 1))
    r = 500

    P_lplr = lplr(X=P, r=r, B1=5, B2=5)
    err_lplr = np.linalg.norm(P_lplr - P, ord="fro") / np.linalg.norm(P, ord="fro")
    print("Error (LPLR): {}".format(err_lplr))

    P_iter = iterative_lplr(X=P, r=int(r / 5), K=5, B1=1, B2=1)
    err_iterative_lplr = np.linalg.norm(P_iter - P, ord="fro") / np.linalg.norm(
        P, ord="fro"
    )
    print("Error (iterative LPLR): {}".format(err_iterative_lplr))


def testbench_direct_svd_vs_lplr():
    # Load phantom image
    P = shepp_logan(1000)
    P = np.interp(P, (P.min(), P.max()), (0, 1))
    r = 200

    P_direct = direct_svd_quant(X=P, r=r, B1=8, B2=8)
    err_direct_svd = np.linalg.norm(P_direct - P, ord="fro") / np.linalg.norm(
        P, ord="fro"
    )
    print("Error (Direct SVD): {}".format(err_direct_svd))

    P_lplr = lplr(X=P, r=r, B1=8, B2=8)
    err_lplr = np.linalg.norm(P_lplr - P, ord="fro") / np.linalg.norm(P, ord="fro")
    print("Error (LPLR): {}".format(err_lplr))

    # Plot the images side by side
    fig, axs = plt.subplots(1, 3)
    plt.set_cmap("gray")
    axs[0].imshow(P)
    axs[0].axis("off")
    axs[0].title.set_text("Original image")
    axs[1].imshow(P_direct)
    axs[1].axis("off")
    axs[1].title.set_text("Direct SVD - multibit")
    axs[2].imshow(P_lplr)
    axs[2].axis("off")
    axs[2].title.set_text("LPLR - multibit")
    plt.tight_layout()
    plt.show()



def testbench_sign_vs_quantize():
    # Load phantom image
    P = shepp_logan(1000)
    P = np.interp(P, (P.min(), P.max()), (0, 1))
    r = 200

    P_sign = lplr(X=P, r=r, B1=1, B2=1)
    err_sign = np.linalg.norm(P_sign - P, ord="fro") / np.linalg.norm(P, ord="fro")
    print("Error (Sign): {}".format(err_sign))

    P_app = lplr(X=P, r=r, B1=1, B2=6)
    err_app = np.linalg.norm(P_app - P, ord="fro") / np.linalg.norm(P, ord="fro")
    print("Error (Multi-bit): {}".format(err_app))

    # Plot the images side by side
    fig, axs = plt.subplots(1, 3)
    plt.set_cmap("gray")
    axs[0].imshow(P)
    axs[0].axis("off")
    axs[0].title.set_text("Original image")
    axs[1].imshow(P_sign)
    axs[1].axis("off")
    axs[1].title.set_text("Sign")
    axs[2].imshow(P_app)
    axs[2].axis("off")
    axs[2].title.set_text("Quant. Multi-bit")
    plt.tight_layout()

    plt.show()


def testbench_shift():
    # Load phantom image
    P = shepp_logan(1000)
    P = np.interp(P, (P.min(), P.max()), (0, 1))
    r = 200

    P_noshift = lplr_noshift(X=P, r=r, B1=1, B2=1)
    err_sign = np.linalg.norm(P_noshift - P, ord="fro") / np.linalg.norm(P, ord="fro")
    print("Error (Scaled): {}".format(err_sign))

    P_shift = lplr(X=P, r=r, B1=1, B2=1)
    err_sign_shift = np.linalg.norm(P_shift - P, ord="fro") / np.linalg.norm(
        P, ord="fro"
    )
    print("Error (Scaled and Shifted): {}".format(err_sign_shift))

    # Plot the images side by side
    fig, axs = plt.subplots(1, 3)
    plt.set_cmap("gray")
    axs[0].imshow(P)
    axs[0].axis("off")
    axs[0].title.set_text("Original image")
    axs[1].imshow(P_noshift)
    axs[1].axis("off")
    axs[1].title.set_text("Sign (scaling)")
    axs[2].imshow(P_shift)
    axs[2].axis("off")
    axs[2].title.set_text("Sign (scaling and shifting)")
    plt.tight_layout()

    plt.show()


def lplr_testbench():
    random.seed(10)

    # Load phantom image
    P = shepp_logan(1000)
    P = np.interp(P, (P.min(), P.max()), (0, 1))
    r = 400  # Intrinsic rank

    P_app_dir_svd = direct_svd_quant(X=P, r=r, B1=1, B2=1)
    err_dir_svd = np.linalg.norm(P_app_dir_svd - P, ord="fro") / np.linalg.norm(
        P, ord="fro"
    )
    print("Error - sign quantizing direct SVD factors: {}".format(err_dir_svd))

    P_app_lplr_sign = lplr_sign(X=P, r=r)
    err_lplr_sign = np.linalg.norm(P_app_lplr_sign - P, ord="fro") / np.linalg.norm(
        P, ord="fro"
    )
    print("Error - LPLR (Sign): {}".format(err_lplr_sign))

    P_app_rsvd = rsvd(X=P, r=r)
    err_rsvd = np.linalg.norm(P_app_rsvd - P, ord="fro") / np.linalg.norm(P, ord="fro")
    print("Error - Full-precision RSVD: {}".format(err_rsvd))

    P_app_semi_quant = lplr_semi_quant(X=P, r=r)
    err_semi_quant = np.linalg.norm(P_app_semi_quant - P, ord="fro") / np.linalg.norm(
        P, ord="fro"
    )
    print("Error - Semi-Quantized RSVD: {}".format(err_semi_quant))

    P_app_lplr_shift = lplr_sign_shift(X=P, r=r)
    err_lplr_shift = np.linalg.norm(P_app_lplr_shift - P, ord="fro") / np.linalg.norm(
        P, ord="fro"
    )
    print("Error - LPLR with normalization and shift: {}".format(err_lplr_shift))

    P_app_lplr_multibit = lplr(X=P, r=r, B1=1, B2=16)
    err_lplr_multibit = np.linalg.norm(
        P_app_lplr_multibit - P, ord="fro"
    ) / np.linalg.norm(P, ord="fro")
    print("Error - LPLR (multibit): {}".format(err_lplr_multibit))

    # Plot the images side by side
    fig, axs = plt.subplots(1, 7)
    plt.set_cmap("gray")
    axs[0].imshow(P)
    axs[0].axis("off")
    axs[0].title.set_text("Original image")
    axs[1].imshow(P_app_dir_svd)
    axs[1].axis("off")
    axs[1].title.set_text("Direct SVD quant.")
    axs[2].imshow(P_app_lplr_sign)
    axs[2].axis("off")
    axs[2].title.set_text("LPLR with scaling")
    axs[3].imshow(P_app_rsvd)
    axs[3].axis("off")
    axs[3].title.set_text("Full-precision RSVD")
    axs[4].imshow(P_app_semi_quant)
    axs[4].axis("off")
    axs[4].title.set_text("Semi-quant RSVD")
    axs[5].imshow(P_app_lplr_shift)
    axs[5].axis("off")
    axs[5].title.set_text("LPLR - scaling and shift")
    axs[6].imshow(P_app_lplr_multibit)
    axs[6].axis("off")
    axs[6].title.set_text("LPLR - multibit")
    plt.tight_layout()

    plt.show()

def quantize_v2(
    X: torch.Tensor = None,
    B: int = 16,
    full_range: bool = False,
    simulate: bool = False,
    preserve_original_dtype=False,
    force=False
) -> torch.Tensor:
    """
    Element-wise matrix quantization for general bit-budget
    :param X (torch.Tensor): Matrix to be quantized
    :param B (int): Bit-budget per coordinate for quantization
    :param full_range (bool): If true, use bfloat16 when B=16, ignored otherwise
    :param simulate (bool): If true, simulate quantization using 64bits
    :param preserve_original_dtype (bool): If true, retains original dtype after quantization
    :return: Quantized matrix
    """
    from loguru import logger
    orig_dtype = X.dtype
    device = X.device
    if B == 16 and device == torch.device("cpu"):
        logger.warning(
            f"Setting simulate = True as Half() dtype is not supported on CPU"
        )
        simulate = True
    
    match B:
        case 64 if not simulate:
            out = X.double()
        case 32 if not simulate:
            out = X.float()
        case 16 if not simulate:
            out = X.half() if not full_range else X.bfloat16()
        case _:
            if simulate:
                logger.warning(f"Forced quantization simulation to {B} bits")
            else:
                logger.warning(f"Using simulation to quantize to {B} bits")

            M = 2**B  # No. of quantization points per dimension
            res = 2 / (M - 1)  # Resolution

            # Normalize the coordinates of the quantizer input to [-1,1]
            X_min = X.min().item()
            X_max = X.max().item()
            # logger.trace(f"X_min = {X_min}, X_min.dtype = {X_min.dtype}, X_min.device = {X_min.device}")
            # logger.trace(f"X = {X}, X.dtype = {X.dtype}, X.device = {X.device}")
            X = torch.from_numpy(
                np.interp(X.to("cpu").numpy(), (X_min, X_max), (-1, 1))
            ).to(X.device)
            # logger.trace(f"X = {X}, X.dtype = {X.dtype}, X.device = {X.device}")

            # Quantize each coordinate with a scalar quantizer of unit dynamic range
            fst_pt = -1  # First quantization point
            L_idx = torch.floor((X - fst_pt) / res)  # Lower index for each entry
            L = fst_pt + L_idx * res  # Matrix of lower quantization points
            U = fst_pt + (L_idx + 1) * res  # Matrix of upper quantization points

            # Nearest neighbor quantization
            Q = torch.zeros_like(X)
            Q[X < -1] = -1  # Value less than lower limit of dynamic range
            Q[X > 1] = 1  # Value more than upper limit of dynamic range
            mask0 = torch.abs(X) <= 1  # Value within dynamic range
            mask = torch.abs(X - L) <= res / 2
            Q[mask * mask0] = L[mask * mask0]
            mask = torch.abs(U - X) <= res / 2
            Q[mask * mask0] = U[mask * mask0]

            # Re-normalize the quantized matrix back to its input scale
            Qr = torch.from_numpy(
                np.interp(
                    Q.to("cpu").numpy(),
                    (Q.min().item(), Q.max().item()),
                    (X_min, X_max),
                )
            ).to(X.device)
            logger.trace(f"Qr.dtype = {Qr.dtype}, Qr.device = {Qr.device}")
            out = Qr
    if preserve_original_dtype:
        out = out.to(orig_dtype)
    return out

def maximum_output_rank(
    compression_ratio: float,
    b1: int,
    b2: int,
    b_nq: float,
    input_shape: Tuple[int, int],
):
    numerator = compression_ratio * math.prod(input_shape) * b_nq
    denominator = np.dot(input_shape, (b1, b2))
    return math.floor(numerator / denominator)

def paper_output_data():
    
    def save_image(mat, bp, name):
        plt.clf()
        plt.set_cmap("gray")
        im = plt.imshow(mat)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(bp/name,  bbox_inches='tight', pad_inches=0)
        
    # Load phantom image
    P = shepp_logan(1000)
    P = np.interp(P, (P.min(), P.max()), (0, 1))
    r = 200
    n = P.shape[0]
    d = P.shape[1]
    
    base_output_path = pathlib.Path(f"artifacts/paper")
    base_output_path.mkdir(parents=True, exist_ok=True)
    
    b_lplr_range = [8, 16, 32]
    b_nq_range = [1, 2, 4, 8]
    
    save_image(P, base_output_path, "original.png")
    # for rr in [100,200,400,800,1600]:
    for b1 in b_lplr_range:
        for b_nq in b_nq_range:
            rr = maximum_output_rank(1, b1, b1, b_nq, P.shape)
            output_dir = base_output_path / f"rank-{rr}_b1-{b1}_b0-{b_nq}"
            output_dir.mkdir(parents=True, exist_ok=True)
            log_file = output_dir / "eval.log"
            print(f"processing b1 = {b1} b_nq = {b_nq} rank = {rr}")
            with open(log_file, "w") as f:
                P_direct = direct_svd_quant(X=P, r=r, B1=b1, B2=b1)
                err_direct_svd = np.linalg.norm(P_direct - P, ord="fro") / np.linalg.norm(
                    P, ord="fro"
                )
                print(f"Error (Direct SVD): {err_direct_svd}", file=f)
                save_image(P_direct, output_dir, "dsvd.png")

                P_lplr = lplr(X=P, r=r, B1=b1, B2=b1)
                err_lplr = np.linalg.norm(P_lplr - P, ord="fro") / np.linalg.norm(P, ord="fro")
                print(f"Error (LPLR): {err_lplr}", file=f)
                save_image(P_lplr, output_dir, "lplr.png")
                
                P_nq = quantize_v2(torch.from_numpy(P), b_nq).numpy()
                err_nq = np.linalg.norm(P_nq - P, ord="fro") / np.linalg.norm(P, ord="fro")
                print(f"Error (NQ): {err_nq}", file=f)
                save_image(P_nq, output_dir, "nq.png")
                
                cr = ((n + d) * rr * b1)/(n * d * 64)
                # cr = 1.0
                print(f"Compression Ratio: {cr}", file=f)
                

if __name__ == "__main__":
    # iterative_vs_oneshot_lplr_comparison()

    paper_output_data()
    
    # testbench_direct_svd_vs_lplr()

    # testbench_sign_vs_quantize()

    # testbench_shift()

    # lplr_testbench()
