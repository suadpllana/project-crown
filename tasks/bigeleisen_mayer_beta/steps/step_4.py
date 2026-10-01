import numpy as np

# Functions from Steps 1-3 are available in this scope.


def high_temperature_coefficient(coords, masses_light, masses_heavy, hessian):
    """Leading high-temperature coefficient K of the per-atom ln(beta).

    K is defined by ln(beta) = K / T^2 + O(T^-4) as T -> infinity, for the same
    isotopologue pair and conventions as Step 3. The Hessian is translationally
    and rotationally invariant and has no imaginary modes.

    Returns
    -------
    float
        K in K^2.

    Raises
    ------
    ValueError
        If no site is substituted or sites carry different isotope pairs.
    """
    raise NotImplementedError
