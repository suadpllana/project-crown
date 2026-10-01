"""Reference solution for Step 3."""
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


# ---- Step 1 (dependency, reference implementation) ----
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


# ---- Step 2 (dependency, reference implementation) ----
def log_reduced_partition_function_ratio(nu_light, nu_heavy, T):
    """Natural log of the Bigeleisen-Mayer reduced partition function ratio f (heavy/light).

    nu_light, nu_heavy : 1-D arrays (same length) of real harmonic wavenumbers, cm^-1, > 0
    T                  : temperature in K, positive scalar or array

    Returns ln f with the shape of T (a Python/NumPy scalar for scalar T).
    """
    nl = np.asarray(nu_light, dtype=float)
    nh = np.asarray(nu_heavy, dtype=float)
    if nl.ndim != 1 or nh.ndim != 1 or nl.shape != nh.shape or nl.size == 0:
        raise ValueError("nu_light and nu_heavy must be non-empty 1-D arrays of equal length")
    if not (np.all(np.isfinite(nl)) and np.all(np.isfinite(nh))):
        raise ValueError("wavenumbers must be finite")
    if np.any(nl <= 0.0) or np.any(nh <= 0.0):
        raise ValueError("wavenumbers must be real and positive")
    t = np.asarray(T, dtype=float)
    if not np.all(np.isfinite(t)) or np.any(t <= 0.0):
        raise ValueError("temperatures must be positive and finite")

    t_flat = np.atleast_1d(t).ravel()
    u_l = C2_CM_K * nl[:, None] / t_flat[None, :]
    u_h = C2_CM_K * nh[:, None] / t_flat[None, :]

    def phi(u):
        # ln[u exp(-u/2) / (1 - exp(-u))], evaluated without overflow or cancellation.
        return np.log(u) - 0.5 * u - np.log(-np.expm1(-u))

    out = np.sum(phi(u_h) - phi(u_l), axis=0)
    if t.ndim == 0:
        return float(out[0])
    return out.reshape(t.shape)


# ---- Step 3 ----
def _substitution_sites(masses_light, masses_heavy):
    ml = np.asarray(masses_light, dtype=float)
    mh = np.asarray(masses_heavy, dtype=float)
    if ml.ndim != 1 or ml.shape != mh.shape:
        raise ValueError("masses_light and masses_heavy must be 1-D arrays of equal length")
    sites = np.flatnonzero(~np.isclose(ml, mh, rtol=1e-12, atol=0.0))
    if sites.size == 0:
        raise ValueError("no isotopic substitution between the two mass vectors")
    if not (np.allclose(ml[sites], ml[sites[0]], rtol=1e-9, atol=0.0)
            and np.allclose(mh[sites], mh[sites[0]], rtol=1e-9, atol=0.0)):
        raise ValueError("all substituted sites must carry the same light -> heavy isotope pair")
    return sites


def log_beta(coords, masses_light, masses_heavy, hessian, T):
    """ln(beta) per substituted atom for the isotopologue pair (heavy vs light).

    The Born-Oppenheimer Hessian is shared by both isotopologues. ln(beta) = ln(f) / n,
    where n is the number of atoms whose mass differs between the two mass vectors.
    Returns the shape of T (scalar for scalar T).
    """
    sites = _substitution_sites(masses_light, masses_heavy)
    nu_l = vibrational_wavenumbers(coords, masses_light, hessian)
    nu_h = vibrational_wavenumbers(coords, masses_heavy, hessian)
    ln_f = log_reduced_partition_function_ratio(nu_l, nu_h, T)
    return ln_f / sites.size
