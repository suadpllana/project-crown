def intercept_age_sigmas(intercept, slope, cov, ages):
    """1-sigma uncertainties (Ma) of concordia-intercept ages from the covariance of the
    line parameters, by first-order (implicit-function) propagation. Decay-constant
    uncertainties are not included."""
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        raise ValueError("cov must be a finite 2x2 matrix")
    if cov[0, 0] < 0 or cov[1, 1] < 0 or abs(cov[0, 1] - cov[1, 0]) > 1e-12 * max(abs(cov[0, 1]), 1e-300):
        raise ValueError("cov must be a symmetric covariance matrix")
    ages = np.asarray(ages, dtype=float)
    t = ages * YEARS_PER_MA
    e8, e5 = np.exp(LAMBDA_238 * t), np.exp(LAMBDA_235 * t)
    df_dt = (LAMBDA_238 * e8 - slope * LAMBDA_235 * e5) * YEARS_PER_MA   # per Ma
    if np.any(df_dt == 0):
        raise ValueError("tangent intercept: age uncertainty undefined")
    dt_da = 1.0 / df_dt
    dt_db = np.expm1(LAMBDA_235 * t) / df_dt
    var = dt_da ** 2 * cov[0, 0] + dt_db ** 2 * cov[1, 1] + 2.0 * dt_da * dt_db * cov[0, 1]
    return np.sqrt(np.maximum(var, 0.0))
