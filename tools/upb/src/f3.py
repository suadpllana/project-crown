def _concordia_misfit(t_ma, intercept, slope):
    t = np.asarray(t_ma, dtype=float) * YEARS_PER_MA
    return np.expm1(LAMBDA_238 * t) - intercept - slope * np.expm1(LAMBDA_235 * t)


def _bisect(f, lo, hi, tol=1e-10):
    flo = f(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm == 0.0:
            return mid
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
        if hi - lo <= tol:
            break
    return 0.5 * (lo + hi)


def concordia_intercepts(intercept, slope):
    """Ages (Ma, ascending) at which the line y = intercept + slope * x crosses the
    Wetherill concordia within [T_MIN_MA, T_MAX_MA]. Returns an array of 0, 1 or 2 ages."""
    a, b = float(intercept), float(slope)
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError("intercept and slope must be finite")
    f = lambda t: float(_concordia_misfit(t, a, b))
    # f is strictly concave in t when b > 0 (maximum at t_star) and increasing when b <= 0.
    pieces = [(T_MIN_MA, T_MAX_MA)]
    if b > 0:
        t_star = np.log(LAMBDA_238 / (b * LAMBDA_235)) / (LAMBDA_235 - LAMBDA_238) / YEARS_PER_MA
        if T_MIN_MA < t_star < T_MAX_MA:
            pieces = [(T_MIN_MA, t_star), (t_star, T_MAX_MA)]
    roots = []
    for lo, hi in pieces:
        flo, fhi = f(lo), f(hi)
        if flo == 0.0:
            roots.append(lo)
        elif fhi == 0.0:
            roots.append(hi)
        elif (flo > 0) != (fhi > 0):
            roots.append(_bisect(f, lo, hi))
    roots = sorted(set(roots))
    return np.array(roots, dtype=float)
