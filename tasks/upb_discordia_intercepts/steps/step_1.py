import numpy as np


def tera_wasserburg_to_wetherill(X, sX, Y, sY, rho):
    """Convert Tera-Wasserburg U-Pb ratios to Wetherill ratios with error propagation.

    Parameters
    ----------
    X, sX : array_like, shape (n,)
        238U/206Pb and its 1-sigma absolute uncertainty (both > 0).
    Y, sY : array_like, shape (n,)
        207Pb/206Pb and its 1-sigma absolute uncertainty (both > 0).
    rho : array_like, shape (n,)
        Correlation coefficient between the errors of X and Y, |rho| < 1.

    Returns
    -------
    numpy.ndarray, shape (n, 5)
        Columns [x, sx, y, sy, rho_xy]: x = 207Pb/235U, y = 206Pb/238U, their 1-sigma
        absolute uncertainties and the correlation of their errors, from first-order
        propagation of the full covariance. Use 238U/235U = 137.818.

    Raises
    ------
    ValueError
        Unequal/empty arrays, non-positive ratios or uncertainties, |rho| >= 1, non-finite input.
    """
    raise NotImplementedError
