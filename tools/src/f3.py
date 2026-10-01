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
