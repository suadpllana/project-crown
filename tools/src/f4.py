def high_temperature_coefficient(coords, masses_light, masses_heavy, hessian):
    """K (in K^2) such that ln(beta) -> K / T^2 as T -> infinity (per substituted atom)."""
    sites = _substitution_sites(masses_light, masses_heavy)
    nu_l = vibrational_wavenumbers(coords, masses_light, hessian)
    nu_h = vibrational_wavenumbers(coords, masses_heavy, hessian)
    if np.any(nu_l <= 0.0) or np.any(nu_h <= 0.0):
        raise ValueError("imaginary vibrational modes are not allowed")
    # ln f = sum[ (u^2 - u*^2) / 24 ] + O(T^-4), u = c2 * nu / T
    return C2_CM_K ** 2 * np.sum(nu_l ** 2 - nu_h ** 2) / (24.0 * sites.size)
