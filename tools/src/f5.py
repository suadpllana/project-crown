def isotope_fractionation(species_a, species_b, temperatures):
    """Equilibrium fractionation 1000 ln(alpha_{a-b}) between two species and its fit.

    Each species is a dict with keys 'coords' (N,3) Angstrom, 'masses_light' (N,) amu,
    'masses_heavy' (N,) amu and 'hessian' (3N,3N) Hartree/bohr^2.
    temperatures: 1-D array (K) with at least 3 distinct positive values.
    """
    t = np.asarray(temperatures, dtype=float)
    if t.ndim != 1 or not np.all(np.isfinite(t)) or np.any(t <= 0.0):
        raise ValueError("temperatures must be a 1-D array of positive values")
    if np.unique(t).size < 3:
        raise ValueError("at least three distinct temperatures are required for the fit")

    pairs = []
    for sp in (species_a, species_b):
        for key in ("coords", "masses_light", "masses_heavy", "hessian"):
            if key not in sp:
                raise ValueError("species is missing key '%s'" % key)
        sites = _substitution_sites(sp["masses_light"], sp["masses_heavy"])
        pairs.append((float(np.asarray(sp["masses_light"], float)[sites[0]]),
                      float(np.asarray(sp["masses_heavy"], float)[sites[0]])))
    if not np.allclose(pairs[0], pairs[1], rtol=1e-9, atol=0.0):
        raise ValueError("both species must carry the same isotope substitution")

    def lnb(sp):
        return log_beta(sp["coords"], sp["masses_light"], sp["masses_heavy"], sp["hessian"], t)

    def coef(sp):
        return high_temperature_coefficient(sp["coords"], sp["masses_light"],
                                            sp["masses_heavy"], sp["hessian"])

    b_a = 1000.0 * lnb(species_a)
    b_b = 1000.0 * lnb(species_b)
    alpha = b_a - b_b

    x = 1000.0 / t
    design = np.column_stack([x ** 2, x, np.ones_like(x)])
    fit, *_ = np.linalg.lstsq(design, alpha, rcond=None)

    return {
        "thousand_ln_beta_a": b_a,
        "thousand_ln_beta_b": b_b,
        "thousand_ln_alpha": alpha,
        "fit": fit,
        "a_high_T": 1000.0 * (coef(species_a) - coef(species_b)) / 1.0e6,
    }
