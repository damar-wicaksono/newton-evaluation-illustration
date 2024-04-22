import numpy as np
import minterpy as mp

from utils_numba_cuda import eval_newton_driver_numba_gpu


if __name__ == "__main__":
    
    spatial_dimension = 6
    mi = mp.MultiIndexSet.from_degree(
        spatial_dimension=spatial_dimension,
        poly_degree=8,
        lp_degree=2.0,
    )
    
    xx_test = -1 + 2 * np.random.rand(1000000, spatial_dimension)
    nwt_coeffs = np.random.rand(len(mi))
    exponents = mi.exponents
    grd = mp.Grid(mi)
    gen_points = grd.generating_points
    
    yy_numba_gpu = eval_newton_driver_numba_gpu(
        xx_test,
        nwt_coeffs,
        exponents,
        gen_points,
    )
