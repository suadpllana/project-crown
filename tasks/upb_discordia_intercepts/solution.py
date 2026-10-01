"""Complete reference solution: U-Pb discordia intercept ages from Tera-Wasserburg data."""
import numpy as np

# Decay constants (Jaffey et al. 1971), per year, and present-day 238U/235U (Hiess et al. 2012).
LAMBDA_238 = 1.55125e-10
LAMBDA_235 = 9.8485e-10
U238_U235 = 137.818
YEARS_PER_MA = 1.0e6
# Age window (Ma) in which concordia intercepts are sought.
T_MIN_MA = -1000.0
T_MAX_MA = 5000.0


# ---- Step 1 ----
def tera_wasserburg_to_wetherill(X, sX, Y, sY, rho):
    """Convert Tera-Wasserburg ratios to Wetherill ratios with linear error propagation.

    X = 238U/206Pb, Y = 207Pb/206Pb (1-sigma absolute errors sX, sY, correlation rho).
    Returns an (n, 5) array with columns [x, sx, y, sy, rho_xy] where
    x = 207Pb/235U and y = 206Pb/238U.
    """
    X, sX, Y, sY, rho = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (X, sX, Y, sY, rho))
    n = X.shape[0]
    if X.ndim != 1 or n == 0 or any(v.shape != (n,) for v in (sX, Y, sY, rho)):
        raise ValueError("inputs must be 1-D arrays of equal, non-zero length")
    if not all(np.all(np.isfinite(v)) for v in (X, sX, Y, sY, rho)):
        raise ValueError("inputs must be finite")
    if np.any(X <= 0) or np.any(Y <= 0) or np.any(sX <= 0) or np.any(sY <= 0):
        raise ValueError("ratios and their errors must be positive")
    if np.any(np.abs(rho) >= 1):
        raise ValueError("correlation coefficients must satisfy |rho| < 1")

    x = U238_U235 * Y / X
    y = 1.0 / X
    # Jacobian of (x, y) with respect to (X, Y).
    dx_dX = -U238_U235 * Y / X ** 2
    dx_dY = U238_U235 / X
    dy_dX = -1.0 / X ** 2
    cXX, cYY, cXY = sX ** 2, sY ** 2, rho * sX * sY
    vx = dx_dX ** 2 * cXX + dx_dY ** 2 * cYY + 2.0 * dx_dX * dx_dY * cXY
    vy = dy_dX ** 2 * cXX
    cxy = dx_dX * dy_dX * cXX + dx_dY * dy_dX * cXY
    sx, sy = np.sqrt(vx), np.sqrt(vy)
    return np.column_stack([x, sx, y, sy, cxy / (sx * sy)])


# ---- Step 2 ----
def york_fit(x, sx, y, sy, rho, rtol=1e-15, max_iter=1000):
    """Best straight line y = a + b x for data with correlated errors in x and y
    (York et al. 2004). Returns a dict with intercept, slope, cov (2x2, order
    [intercept, slope]), mswd and n."""
    x, sx, y, sy, rho = (np.asarray(v, dtype=float) for v in (x, sx, y, sy, rho))
    n = x.shape[0] if x.ndim == 1 else 0
    if n < 3 or any(v.shape != (n,) for v in (sx, y, sy, rho)):
        raise ValueError("need at least 3 points given as 1-D arrays of equal length")
    if not all(np.all(np.isfinite(v)) for v in (x, sx, y, sy, rho)):
        raise ValueError("inputs must be finite")
    if np.any(sx <= 0) or np.any(sy <= 0) or np.any(np.abs(rho) >= 1):
        raise ValueError("errors must be positive and |rho| < 1")
    if np.ptp(x) == 0:
        raise ValueError("x values must not all be equal")

    wx, wy = 1.0 / sx ** 2, 1.0 / sy ** 2
    alpha = np.sqrt(wx * wy)

    def state(b):
        w = wx * wy / (wx + b * b * wy - 2.0 * b * rho * alpha)
        xbar, ybar = w @ x / w.sum(), w @ y / w.sum()
        u, v = x - xbar, y - ybar
        beta = w * (u / wy + b * v / wx - (b * u + v) * rho / alpha)
        return w, xbar, ybar, u, v, beta

    b = np.polyfit(x, y, 1)[0]
    for _ in range(max_iter):
        w, xbar, ybar, u, v, beta = state(b)
        b_new = (w * beta) @ v / ((w * beta) @ u)
        converged = abs(b_new - b) <= rtol * abs(b_new) or b_new == b
        b = b_new
        if converged:
            break
    else:
        raise RuntimeError("York iteration did not converge")

    w, xbar, ybar, u, v, beta = state(b)
    a = ybar - b * xbar
    x_adj = xbar + beta
    x_adj_bar = w @ x_adj / w.sum()
    var_b = 1.0 / (w @ (x_adj - x_adj_bar) ** 2)
    var_a = 1.0 / w.sum() + x_adj_bar ** 2 * var_b
    cov_ab = -x_adj_bar * var_b
    mswd = w @ (y - a - b * x) ** 2 / (n - 2)
    return {
        "intercept": float(a),
        "slope": float(b),
        "cov": np.array([[var_a, cov_ab], [cov_ab, var_b]]),
        "mswd": float(mswd),
        "n": int(n),
    }


# ---- Step 3 ----
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


# ---- Step 4 ----
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


# ---- Step 5 ----
def discordia_ages(X, sX, Y, sY, rho):
    """Upper and lower concordia-intercept ages of a discordant U-Pb array given in
    Tera-Wasserburg form. See problem.yaml for the output dictionary."""
    w = tera_wasserburg_to_wetherill(X, sX, Y, sY, rho)
    fit = york_fit(w[:, 0], w[:, 1], w[:, 2], w[:, 3], w[:, 4])
    ages = concordia_intercepts(fit["intercept"], fit["slope"])
    if ages.size != 2:
        raise ValueError("the discordia does not cross concordia twice in the age window")
    sig = intercept_age_sigmas(fit["intercept"], fit["slope"], fit["cov"], ages)
    if fit["mswd"] > 1.0:
        sig = sig * np.sqrt(fit["mswd"])
    return {
        "intercept": fit["intercept"],
        "slope": fit["slope"],
        "cov": fit["cov"],
        "mswd": fit["mswd"],
        "n": fit["n"],
        "ages": ages,
        "sigmas": sig,
    }
