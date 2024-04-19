import numba
import numpy as np

from numba import njit, void

INT_DTYPE = np.int_
FLOAT_DTYPE = np.float_


def eval_newton_monomials_single(
    x_single: np.ndarray,
    exponents: np.ndarray,
    generating_points: np.ndarray,
    max_exponents: np.ndarray,
    products_placeholder: np.ndarray,
    monomials_placeholder: np.ndarray,
) -> None:
    """Precompute the value of all Newton monomials at a single point (O(Nm)).

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


def eval_newton_polynomial(
    xx: np.ndarray,
    exponents: np.ndarray,
    generating_points: np.ndarray,
    newton_coefficients: np.ndarray,
    max_exponents: np.ndarray,
    products_placeholder: np.ndarray,
    monomials_placeholder: np.ndarray,
    output_placeholder: np.ndarray,
) -> None:
    """Evaluate the Newton monomials at multiple query points.

    The following notations are used below:

    - :math:`m`: the spatial dimension of the polynomial
    - :math:`p`: the (maximum) degree of the polynomial in any dimension
    - :math:`n`: the number of elements in the multi-index set (i.e., monomials)
    - :math:`N`: the number of query (evaluation) points
    - :math:`\mathrm{nr_polynomials}`: the number of polynomials with different
      coefficient sets of the same multi-index set

    :param xx: numpy array with coordinates of points where polynomial is to be evaluated.
              The shape has to be ``(k x m)``.
    :param exponents: numpy array with exponents for the polynomial. The shape has to be ``(N x m)``.
    :param generating_points: generating points used to generate the grid. The shape is ``(n x m)``.
    :param max_exponents: array with maximum exponent in each dimension. The shape has to be ``m``.
    :param products_placeholder: a numpy array for storing the (chained) products.
    :param monomials_placeholder: placeholder numpy array where the results of evaluation are stored.
                               The shape has to be ``(k x p)``.
    :param triangular: whether the output will be of lower triangular form or not.
                       -> will skip the evaluation of some values
    :return: the value of each Newton polynomial at each point. The shape will be ``(k x N)``.

    Notes
    -----
    - This is a Numba-accelerated function.
    - The memory footprint for evaluating the Newton monomials iteratively
       with a single query point at a time is smaller than evaluating all
       the Newton monomials on all query points.
       However, when multiplied with multiple coefficient sets,
       this approach will be faster.
    - Results are stored in the placeholder arrays. The function returns None.
    """
    # Total number of query points (N)
    n_points = xx.shape[0]

    # Iterate each query points and evaluate the Newton monomials
    for idx in range(n_points):

        x_single = xx[idx, :]

        # Evaluate the Newton monomials on a single query point
        # NOTE: Due to "view" access,
        # the whole 'monomials_placeholder' will be modified in place
        eval_newton_monomials_single(
            x_single,
            exponents,
            generating_points,
            max_exponents,
            products_placeholder,
            monomials_placeholder,
        )

        output_placeholder[idx] = np.dot(
            monomials_placeholder,
            newton_coefficients,
        )


def eval_newton_driver_base(
    xx: np.ndarray,
    coefficients: np.ndarray,
    exponents: np.ndarray,
    generating_points: np.ndarray,
):
    """Evaluate the polynomial(s) in Newton form at multiple query points.

    Iterative implementation of polynomial evaluation in Newton form

    This version able to handle both:
        - list of input points x (2D input)
        - list of input coefficients (2D input)

    Here we use the notations:
        - ``n`` = polynomial degree
        - ``N`` = amount of coefficients
        - ``k`` = amount of points
        - ``p`` = amount of polynomials


    .. todo::
        - idea for improvement: make use of the sparsity of the exponent matrix and avoid iterating over the zero entries!
        - refac the explanation and documentation of this function.
        - use instances of :class:`MultiIndex` and/or :class:`Grid` instead of the array representations of them.
        - ship this to the submodule ``newton_polynomials``.

    :param xx: Arguemnt array with shape ``(m, k)`` the ``k`` points to evaluate on with dimensionality ``m``.
    :type xx: np.ndarray
    :param coefficients: The coefficients of the Newton polynomials.
        NOTE: format fixed such that 'lagrange2newton' conversion matrices can be passed
        as the Newton coefficients of all Lagrange monomials of a polynomial without prior transponation
    :type coefficients: np.ndarray, shape = (N, p)
    :param exponents: a multi index ``alpha`` for every Newton polynomial corresponding to the exponents of this ``monomial``
    :type exponents: np.ndarray, shape = (m, N)
    :param generating_points: Grid values for every dimension (e.g. Leja ordered Chebychev values).
        the values determining the locations of the hyperplanes of the interpolation grid.
        the ordering of the values determine the spacial distribution of interpolation nodes.
        (relevant for the approximation properties and the numerical stability).
    :type generating_points: np.ndarray, shape = (m, n+1)
    :param verify_input: weather the data types of the input should be checked. turned off by default for speed.
    :type verify_input: bool, optional
    :param batch_size: batch size of query points
    :type batch_size: int, optional

    :raise TypeError: If the input ``generating_points`` do not have ``dtype = float``.

    :return: (k, p) the value of each input polynomial at each point. TODO squeezed into the expected shape (1D if possible). Notice, format fixed such that the regression can use the result as transformation matrix without transponation

    Notes
    -----
    - This method is faster than the recursive implementation of ``tree.eval_lp(...)`` for a single point and a single polynomial (1 set of coeffs):
        - time complexity: :math:`O(mn+mN) = O(m(n+N)) = ...`
        - pre-computations: :math:`O(mn)`
        - evaluation: :math:`O(mN)`
        - space complexity: :math:`O(mn)` (precomputing and storing the products)
        - evaluation: :math:`O(0)`
    - advantage:
        - just operating on numpy arrays, can be just-in-time (jit) compiled
        - can evaluate multiple polynomials without recomputing all intermediary results

    See Also
    --------
    evaluate_multiple : ``numba`` accelerated implementation which is called internally by this function.
    convert_eval_output: ``numba`` accelerated implementation of the output converter.
    """

    N, m_1 = exponents.shape
    nr_points, m_2 = xx.shape
    assert m_1 == m_2
    
    # NOTE: the downstream numba-accelerated function does not support kwargs,
    # so the maximum exponent per dimension must be computed here
    max_exponents = np.max(exponents, axis=0)

    # Create placeholders for the final and intermediate results
    monomials_placeholder = np.empty(N, dtype=FLOAT_DTYPE)
    output_placeholder = np.empty(nr_points, dtype=FLOAT_DTYPE)
    prod_placeholder = np.empty(
        (np.max(max_exponents) + 1, m_1), dtype=FLOAT_DTYPE
    )

    eval_newton_polynomial(
        xx,
        exponents,
        generating_points,
        coefficients,
        max_exponents,
        prod_placeholder,
        monomials_placeholder,
        output_placeholder,
    )

    return output_placeholder
