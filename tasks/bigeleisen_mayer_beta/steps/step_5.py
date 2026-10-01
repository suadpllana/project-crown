import numpy as np

# Functions from Steps 1-4 are available in this scope.


def isotope_fractionation(species_a, species_b, temperatures):
    """Equilibrium isotope fractionation 1000 ln(alpha_{a-b}) and its 1/T fit.

    Parameters
    ----------
    species_a, species_b : dict
        Keys 'coords' (N,3) Angstrom, 'masses_light' (N,) amu, 'masses_heavy' (N,) amu,
        'hessian' (3N,3N) Hartree/bohr^2. Both species must substitute the same
        isotope pair (same light and heavy mass at their substituted sites).
    temperatures : array_like, shape (K,)
        Temperatures in K (> 0) containing at least three distinct values.

    Returns
    -------
    dict with
        'thousand_ln_beta_a' : ndarray (K,)   1000 ln(beta_a)
        'thousand_ln_beta_b' : ndarray (K,)   1000 ln(beta_b)
        'thousand_ln_alpha'  : ndarray (K,)   1000 (ln beta_a - ln beta_b)
        'fit'                : ndarray (3,)   [A, B, C] of the unweighted least-squares fit
                                              1000 ln alpha = A x^2 + B x + C, x = 1000/T
        'a_high_T'           : float          1000 (K_a - K_b) / 1e6  (same units as A)

    Raises
    ------
    ValueError
        On mismatched isotope systems, missing keys, or invalid temperatures.
    """
    raise NotImplementedError
