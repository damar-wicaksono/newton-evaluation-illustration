import numpy as np
import minterpy as mp
import timeit
import datetime

if __name__ == "__main__":

    SETUP_CODE = \
        f"import minterpy as mp\n" \
        f"import numpy as np\n" \
        f"from newton_eval import eval_driver_numba_cpu_single\n" \
        f"spatial_dimension = 4\n" \
        f"mi = mp.MultiIndexSet.from_degree(spatial_dimension, 45, 2.0)\n" \
        f"xx_test = -1 + 2 * np.random.rand(1000000, spatial_dimension)\n" \
        f"nwt_coeffs = np.random.rand(len(mi))\n" \
        f"exponents = mi.exponents\n" \
        f"grd = mp.Grid(mi)\n" \
        f"gen_points = grd.generating_points\n" \
        f"yy_numba_gpu = eval_driver_numba_cpu_single(xx_test[:5], nwt_coeffs, exponents, gen_points)\n" \

    n_reps = 1
    n_loops = 2
    t = timeit.Timer(f"eval_driver_numba_cpu_single(xx_test, nwt_coeffs, exponents, gen_points)", setup=SETUP_CODE).repeat(n_reps, n_loops)

    t_best = np.min(t) / n_loops
    t_worst = np.max(t) / n_loops
    t_avg = np.mean(t) / n_loops
    t_summary = np.array([t_worst, t_avg, t_best])

    current_datetime = datetime.datetime.now()
    timestamp = current_datetime.timestamp()

    np.savetxt(f"/p/home/jusers/wicaksono1/jureca/cpu-par-single-timing-{timestamp}.txt", t_summary)
