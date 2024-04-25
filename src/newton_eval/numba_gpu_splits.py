import numba
import numpy as np

from math import ceil
from numba import cuda

INT_DTYPE = np.int_
FLOAT_DTYPE = np.float_

FLOAT = numba.from_dtype(FLOAT_DTYPE)
F_1D = FLOAT[:]
F_2D = FLOAT[:, :]

INT = numba.from_dtype(INT_DTYPE)
I_1D = INT[:]
I_2D = INT[:, :]

__all__ = ["eval_driver_numba_gpu_splits"]


@cuda.jit(device=True)
def compute_newton_monomial(
    x_single: np.ndarray,
    exponent_single: np.ndarray,
    generating_points: np.ndarray,
):
    """Precompute the value of a Newton monomial at a single point.

    Notations used below:

    - ``m``: spatial dimension
    - ``n``: the polynomial degree
    - ``N``: the number of monomials (the cardinality of the multi-index set)

    Parameters
    ----------
    x_single : np.ndarray
        A single point, an array of length ``m``.
    exponent_single : np.ndarray
        Array single multidimensional exponent of length ``m``.
    generating_points : np.ndarray
        Array of points to generate the grid;
        the array is of shape ``(n-by-m)``.
    """
    m = len(exponent_single)
    tmp = 1.0
    for i in range(m):
        for j in range(exponent_single[i]):
            p_ij = generating_points[j, i]
            tmp *= x_single[i] - p_ij

    return tmp


@cuda.jit
def eval_newton_polynomial(
    xx: np.ndarray,
    exponents: np.ndarray,
    generating_points: np.ndarray,
    newton_coefficients: np.ndarray,
    output_placeholder: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> None:
    """Evaluate the Newton polynomial multiple query points (CUDA kernel)."""
    row = cuda.grid(1)
    N = exponents.shape[0]
    start_0 = start[0]
    end_0 = end[0]
    if row < len(output_placeholder):
        tmp = 0.0
        x_single = xx[row]

        upper_bound = end_0 - start_0

        for i in range(upper_bound):
            i = np.int32(i)
            idx = start_0 + i
            newton_monomials = compute_newton_monomial(x_single, exponents[idx], generating_points)
            tmp += newton_monomials * newton_coefficients[idx]

        output_placeholder[row] += tmp


def eval_driver_numba_gpu_splits(
    xx: np.ndarray,
    coefficients: np.ndarray,
    exponents: np.ndarray,
    generating_points: np.ndarray,
    threads_per_block: int = 256,
    splits: int = 5,
):
    """Numba CUDA driver function to evaluate a polynomial in Newton form."""
    # Common data
    nr_points = xx.shape[0]

    # Copy arrays to the device
    xx_global_mem = cuda.to_device(xx)
    exponents_global_mem = cuda.to_device(exponents)
    generating_points_global_mem = cuda.to_device(generating_points)
    coefficients_global_mem = cuda.to_device(coefficients)

    # Allocate memory on the device for results
    output_global_mem = cuda.device_array((nr_points))

    # Configure the blocks
    blocks_per_grid = int(ceil(xx.shape[0] / threads_per_block))

    N = len(exponents)
    chunk = int(N / splits)

    # Start the kernel

    for i in range(splits):
        start = np.array([i*chunk], dtype=np.int32)
        end = np.array([(i + 1)*chunk], dtype=np.int32)

        if end > N:
            end = N

        start_global_mem = cuda.to_device(start)
        end_global_mem = cuda.to_device(end)

        eval_newton_polynomial[blocks_per_grid, threads_per_block](
            xx_global_mem,
            exponents_global_mem,
            generating_points_global_mem,
            coefficients_global_mem,
            output_global_mem,
            start_global_mem,
            end_global_mem,
        )

    # Copy the result back to the host
    output_placeholder = output_global_mem.copy_to_host()

    return output_placeholder
