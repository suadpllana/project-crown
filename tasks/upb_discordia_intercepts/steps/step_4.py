import numpy as np


def intercept_age_sigmas(intercept, slope, cov, ages):
    """First-order 1-sigma uncertainties of concordia-intercept ages.

    Parameters
    ----------
    intercept, slope : float
        Wetherill-space line parameters.
    cov : array_like, shape (2, 2)
        Covariance of (intercept, slope), order [intercept, slope]; symmetric, non-negative
        diagonal.
    ages : array_like, shape (k,)
        Intercept ages in Ma (roots of the concordia condition for this line).

    Returns
    -------
    numpy.ndarray, shape (k,)
        1-sigma age uncertainties in Ma from the line-parameter covariance only (decay
        constant uncertainties are excluded; no MSWD scaling).

    Raises
    ------
    ValueError
        If cov is not a finite symmetric 2x2 matrix with non-negative diagonal.
    """
    raise NotImplementedError
