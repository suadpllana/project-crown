import numpy as np


def york_fit(x, sx, y, sy, rho):
    """Best-fit straight line y = a + b x for data with correlated errors in x and y.

    Implements the maximum-likelihood / least-squares solution of York et al. (2004)
    with standard errors evaluated at the least-squares-adjusted points.

    Parameters
    ----------
    x, sx, y, sy, rho : array_like, shape (n,), n >= 3
        Data, 1-sigma absolute uncertainties (> 0) and error correlations (|rho| < 1).

    Returns
    -------
    dict
        'intercept' : float, a
        'slope'     : float, b
        'cov'       : ndarray (2, 2), covariance of (a, b) in the order [intercept, slope]
        'mswd'      : float, sum of weighted squared residuals / (n - 2)
        'n'         : int, number of points

    Raises
    ------
    ValueError
        n < 3, unequal lengths, non-positive uncertainties, |rho| >= 1, all x equal,
        or non-finite input.
    """
    raise NotImplementedError
