import numpy as np


def log_reduced_partition_function_ratio(nu_light, nu_heavy, T):
    """Natural logarithm of the Bigeleisen-Mayer reduced partition function ratio f.

    Parameters
    ----------
    nu_light, nu_heavy : array_like, shape (M,)
        Real harmonic wavenumbers (cm^-1, > 0) of the light and heavy isotopologue.
        Same length M >= 1; the pairing order of modes is irrelevant.
    T : float or array_like
        Temperature(s) in K, > 0. Must be accurate for 1 K <= T <= 1e6 K.

    Returns
    -------
    float (for scalar T) or numpy.ndarray with the shape of T
        ln f, with f defined heavy-over-light (positive when heavy wavenumbers are lower).

    Raises
    ------
    ValueError
        On empty or mismatched arrays, non-positive/non-finite wavenumbers or temperatures.
    """
    raise NotImplementedError
