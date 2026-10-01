import numpy as np


def concordia_intercepts(intercept, slope):
    """Ages at which the line 206Pb/238U = intercept + slope * 207Pb/235U intersects the
    Wetherill concordia.

    Parameters
    ----------
    intercept, slope : float
        Line parameters in Wetherill coordinates.

    Returns
    -------
    numpy.ndarray, shape (k,), k in {0, 1, 2}
        All intersection ages in Ma within the window -1000 Ma <= t <= 5000 Ma, sorted
        ascending (absolute accuracy 1e-4 Ma). Decay constants: lambda_238 = 1.55125e-10/a,
        lambda_235 = 9.8485e-10/a.
    """
    raise NotImplementedError
