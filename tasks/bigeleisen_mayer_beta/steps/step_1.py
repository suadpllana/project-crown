import numpy as np


def vibrational_wavenumbers(coords, masses, hessian):
    """Harmonic vibrational wavenumbers of a molecule from its Cartesian Hessian.

    Parameters
    ----------
    coords : array_like, shape (N, 3)
        Equilibrium Cartesian coordinates in Angstrom (any origin/orientation), N >= 2.
    masses : array_like, shape (N,)
        Atomic masses in unified atomic mass units (amu); all strictly positive.
    hessian : array_like, shape (3N, 3N)
        Cartesian second-derivative matrix of the Born-Oppenheimer energy in
        Hartree / bohr^2, ordered (x1, y1, z1, x2, y2, z2, ...). It may contain
        numerical noise that couples vibrations to overall translation/rotation.

    Returns
    -------
    numpy.ndarray, shape (3N-6,) for non-linear or (3N-5,) for linear molecules
        Vibrational wavenumbers in cm^-1, sorted ascending. Imaginary modes are
        returned as negative numbers (-|nu|).

    Raises
    ------
    ValueError
        On inconsistent shapes, N < 2, non-positive masses or non-finite input.
    """
    raise NotImplementedError
