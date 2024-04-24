import numba
import numpy as np

from numba import njit, void

INT_DTYPE = np.int_
FLOAT_DTYPE = np.float_

FLOAT = numba.from_dtype(FLOAT_DTYPE)
F_1D = FLOAT[:]
F_2D = FLOAT[:, :]

INT = numba.from_dtype(INT_DTYPE)
I_1D = INT[:]
I_2D = INT[:, :]

__all__ = ["eval_driver_numba_cpu_single"]


@njit(void(F_1D, I_2D, F_2D, I_1D, F_2D, F_1D), cache=True)  # O(Nm)
def eval_newton_monomials_single(
    x_single: np.ndarray,
    exponents: np.ndarray,
    generating_points: np.ndarray,
    max_exponents: np.ndarray,
    products_placeholder: np.ndarray,
    monomials_placeholder: np.ndarray,
) -> None:
    """Precompute the value of all Newton monomials at a single point.

    Notations used below:

    - ``m``: spatial dimension
    - ``n``: the polynomial degree
    - ``N``: the number of monomials (the cardinality of the multi-index set)

    Parameters
    ----------
    x_single : np.ndarray
        A single point, an array of length ``m``.
    exponents : np.ndarray
        Array with exponents for the polynomials of shape ``(N-by-m)``.
    generating_points : np.ndarray
        Array of points to generate the grid;
        the array is of shape ``(n-by-m)``.
    max_exponents : np.ndarray
        Array with the maximum exponent in each spatial dimension;
        the array is of length ``m``.
    products_placeholder : np.ndarray
        Array for storing the chained products;
        the array is if shape ``(n-by-m)``.
    monomials_placeholder : np.ndarray
        Array for storing the values of all Newton monomials
        at the given point; the array is of length ``N``.
    
    Notes
    -----
    - The function precomputes all the (chained) products required
      during a Newton polynomial evaluation for a single query point
      with complexity of ``O(mN)`` (``n`` < ``N``) or ``O(mn)`` (otherwise).
    - The (pre-)computation of Newton monomials is coefficient agnostic.
    - Results are stored in the placeholder arrays. The function returns None.
    """
    # NOTE: the maximal exponent might be different in every dimension,
    #    in this case the matrix becomes sparse (towards the end)
    # NOTE: avoid index shifting during evaluation (has larger complexity than pre-computation!)
    #    by just adding one empty row in front. ATTENTION: these values must not be accessed!
    #    -> the exponents of each monomial ("alpha") then match the indices of the required products

    # Create the products matrix, O(nm)
    m = exponents.shape[1]
    for i in range(m):
        max_exp_in_dim = max_exponents[i]
        x_i = x_single[i]
        prod = 1.0
        for j in range(max_exp_in_dim):  # O(n)
            # TODO there are n+1 1D grid values, the last one will never be used!?
            p_ij = generating_points[j, i]
            prod *= x_i - p_ij
            # NOTE: shift index by one
            exponent = j + 1  # NOTE: otherwise the result type is float
            products_placeholder[exponent, i] = prod

    # evaluate all Newton polynomials. O(Nm)
    N = exponents.shape[0]
    for j in range(N):
        # the exponents of each monomial ("alpha")
        # are the indices of the products which need to be multiplied
        newt_mon_val = 1.0  # required as multiplicative identity
        for i in range(m):
            exp = exponents[j, i]
            # NOTE: an exponent of 0 should not cause a multiplication
            # (inefficient, numerical instabilities)
            if exp > 0:
                newt_mon_val *= products_placeholder[exp, i]
        monomials_placeholder[j] = newt_mon_val
    #NOTE: results have been stored in the numpy arrays. no need to return anything.


@njit(void(F_2D, I_2D, F_2D, F_1D, I_1D, F_1D), cache=True)
def eval_newton_polynomial(
    xx: np.ndarray,
    exponents: np.ndarray,
    generating_points: np.ndarray,
    newton_coefficients: np.ndarray,
    max_exponents: np.ndarray,
    output_placeholder: np.ndarray,
) -> None:
    """Evaluate Newton polynomial at multiple query points.

    The following notations are used below:

    - :math:`m`: the spatial dimension of the polynomial
    - :math:`p`: the (maximum) degree of the polynomial in any dimension
    - :math:`n`: the number of elements in the multi-index set (i.e., monomials)
    - :math:`N`: the number of query (evaluation) points
    - :math:`\mathrm{nr_polynomials}`: the number of polynomials with different
      coefficient sets of the same multi-index set
    """
    # Total number of query points (N)
    n_points = xx.shape[0]
    N = exponents.shape[0]
    m = exponents.shape[1]

    newton_coefficients = np.ascontiguousarray(newton_coefficients)
    # Iterate each query points and evaluate the Newton monomials
    for idx in range(n_points):

        # To avoid race condition each parallel computation
        # must have an independent data placeholders.
        monomials_placeholder = np.empty(N, dtype=FLOAT_DTYPE)
        products_placeholder = np.empty(
            (np.max(max_exponents) + 1, m), dtype=FLOAT_DTYPE
        )
        x_single = xx[idx, :]

        # Evaluate the Newton monomials on a single query point
        eval_newton_monomials_single(
            x_single,
            exponents,
            generating_points,
            max_exponents,
            products_placeholder,
            monomials_placeholder,
        )

        output_placeholder[idx] = np.dot(
            np.ascontiguousarray(monomials_placeholder),
            newton_coefficients,
        )


def eval_driver_numba_cpu_single(
    xx: np.ndarray,
    coefficients: np.ndarray,
    exponents: np.ndarray,
    generating_points: np.ndarray,
) -> np.ndarray:
    """Numba parallel driver function to Evaluate a polynomial in Newton form.

    Here we use the notations:
        - ``n`` = polynomial degree
        - ``N`` = amount of coefficients
        - ``k`` = amount of points
        - ``p`` = amount of polynomials
    """

    _, m_1 = exponents.shape
    nr_points, m_2 = xx.shape
    assert m_1 == m_2
    
    # NOTE: the downstream numba-accelerated function does not support kwargs,
    # so the maximum exponent per dimension must be computed here
    max_exponents = np.max(exponents, axis=0)

    # Create placeholders for the final and intermediate results
    output_placeholder = np.empty(nr_points, dtype=FLOAT_DTYPE)

    eval_newton_polynomial(
        xx,
        exponents,
        generating_points,
        coefficients,
        max_exponents,
        output_placeholder,
    )

    return output_placeholder
