import numpy as np

# vibrational_wavenumbers (Step 1) and log_reduced_partition_function_ratio (Step 2)
# are available in this scope.


def log_beta(coords, masses_light, masses_heavy, hessian, T):
    """Per-atom ln(beta) of an isotopologue pair sharing one Born-Oppenheimer Hessian.

    Parameters
    ----------
    coords : array_like, shape (N, 3), Angstrom
    masses_light, masses_heavy : array_like, shape (N,), amu
        Masses of the reference (light) and substituted (heavy) isotopologue. The atoms
        whose masses differ are the substituted sites; they must all carry the same
        light -> heavy isotope pair.
    hessian : array_like, shape (3N, 3N), Hartree / bohr^2
    T : float or array_like, K

    Returns
    -------
    float or numpy.ndarray with the shape of T
        ln(beta) = ln(f) / n, n = number of substituted sites; no symmetry numbers.

    Raises
    ------
    ValueError
        If no site is substituted, sites carry different isotope pairs, or shapes differ.
    """
    raise NotImplementedError
