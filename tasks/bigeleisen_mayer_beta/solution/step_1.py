"""Reference solution for Step 1."""
import numpy as np

# Physical constants (CODATA 2018, exact or recommended values), SI units.
H_PLANCK = 6.62607015e-34        # J s
C_LIGHT = 299792458.0            # m s^-1
K_BOLTZMANN = 1.380649e-23       # J K^-1
AMU = 1.66053906660e-27          # kg
HARTREE = 4.3597447222071e-18    # J
BOHR = 5.29177210903e-11         # m

# Second radiation constant h c / k_B expressed in cm K (wavenumbers in cm^-1).
C2_CM_K = H_PLANCK * C_LIGHT * 100.0 / K_BOLTZMANN


# ---- Step 1 ----
def vibrational_wavenumbers(coords, masses, hessian):
    """Harmonic vibrational wavenumbers (cm^-1) from a Cartesian Hessian.

    coords  : (N, 3) array, Angstrom
    masses  : (N,) array, unified atomic mass units (amu), all > 0
    hessian : (3N, 3N) array, Hartree / bohr^2, rows/cols ordered x1, y1, z1, x2, ...

    Returns a 1-D array of length 3N-6 (non-linear) or 3N-5 (linear), sorted in
    ascending order. Imaginary modes are reported as negative wavenumbers.
    """
    X = np.asarray(coords, dtype=float)
    m = np.asarray(masses, dtype=float)
    H = np.asarray(hessian, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")
    n_atoms = X.shape[0]
    if n_atoms < 2:
        raise ValueError("at least two atoms are required")
    if m.shape != (n_atoms,):
        raise ValueError("masses must have shape (N,)")
    if H.shape != (3 * n_atoms, 3 * n_atoms):
        raise ValueError("hessian must have shape (3N, 3N)")
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(m)) and np.all(np.isfinite(H))):
        raise ValueError("inputs must be finite")
    if np.any(m <= 0.0):
        raise ValueError("masses must be positive")

    # Mass-weighted Hessian (Hartree / (bohr^2 amu)).
    sqrt_m = np.sqrt(np.repeat(m, 3))
    hmw = H / np.outer(sqrt_m, sqrt_m)
    hmw = 0.5 * (hmw + hmw.T)

    # Linearity test from the principal moments of inertia about the centre of mass.
    r = X - (m @ X) / m.sum()
    inertia = np.einsum("i,ij,ik->jk", m, r, r)
    inertia = np.trace(inertia) * np.eye(3) - inertia
    moments = np.linalg.eigvalsh(inertia)
    linear = moments[0] < 1e-6 * moments[-1]
    n_ext = 5 if linear else 6

    # Mass-weighted translation and infinitesimal-rotation vectors.
    w = np.sqrt(m)[:, None]
    ext = []
    for axis in np.eye(3):
        ext.append((w * np.tile(axis, (n_atoms, 1))).ravel())
    for axis in np.eye(3):
        ext.append((w * np.cross(axis, r)).ravel())
    ext = np.array(ext).T

    # Orthonormal basis of the external space: the n_ext dominant left singular vectors.
    u, _, _ = np.linalg.svd(ext, full_matrices=False)
    basis_ext = u[:, :n_ext]

    # Orthonormal basis of the internal (vibrational) space = complement of the external space.
    projector = np.eye(3 * n_atoms) - basis_ext @ basis_ext.T
    evals, evecs = np.linalg.eigh(projector)
    basis_int = evecs[:, evals > 0.5]
    if basis_int.shape[1] != 3 * n_atoms - n_ext:
        raise ValueError("could not construct the vibrational subspace")

    lam = np.linalg.eigvalsh(basis_int.T @ hmw @ basis_int)
    to_si = HARTREE / (BOHR ** 2 * AMU)          # s^-2
    omega = np.sqrt(np.abs(lam) * to_si)         # rad s^-1
    nu = np.sign(lam) * omega / (2.0 * np.pi * C_LIGHT * 100.0)
    return np.sort(nu)
