import numpy as np

# Functions from Steps 1-4 are available in this scope.


def discordia_ages(X, sX, Y, sY, rho):
    """Upper and lower concordia-intercept ages of a discordant zircon U-Pb array.

    Parameters
    ----------
    X, sX, Y, sY, rho : array_like, shape (n,), n >= 3
        Tera-Wasserburg data: 238U/206Pb, 207Pb/206Pb, 1-sigma absolute errors, correlation.

    Returns
    -------
    dict
        'intercept', 'slope' : float   York line in Wetherill coordinates
        'cov'                : (2, 2)  York covariance of (intercept, slope), unscaled
        'mswd'               : float
        'n'                  : int
        'ages'               : (2,)    [lower, upper] intercept ages, Ma, ascending
        'sigmas'             : (2,)    1-sigma age uncertainties, Ma, multiplied by
                                       sqrt(MSWD) when MSWD > 1 (never reduced)

    Raises
    ------
    ValueError
        Invalid input (as Steps 1-2) or if the line does not cross concordia exactly twice
        within -1000 ... 5000 Ma.
    """
    raise NotImplementedError
