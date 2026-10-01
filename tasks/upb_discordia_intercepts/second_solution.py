"""Independent second reference solution.

Differences from solution.py:
  * Step 1 propagates relative (logarithmic) errors, ln x = ln U + ln Y - ln X and
    ln y = -ln X, instead of the Cartesian Jacobian.
  * Step 2 finds the maximum-likelihood line by Brent-free golden-section minimisation of
    the York objective S(b) (intercept profiled out analytically) over the slope angle,
    and obtains the least-squares-adjusted points geometrically as the Mahalanobis
    projection of each datum onto the line; no beta_i fixed-point iteration.
  * Step 3 solves in the variable s = exp(lambda_238 t), where the concordia condition is
    g(s) = s - 1 - a - b (s^k - 1), k = lambda_235/lambda_238, scanning a grid for sign
    changes and polishing with safeguarded Newton steps.
  * Step 4 re-parametrises the line about its centroid (x0 = -cov_ab/var_b), where the
    centroid ordinate and the slope are uncorrelated, and propagates in that basis.
"""
import numpy as np

L238 = 1.55125e-10
L235 = 9.8485e-10
U_RATIO = 137.818
T_LO, T_HI = -1000.0, 5000.0


def _arr(*vals):
    out = [np.atleast_1d(np.asarray(v, dtype=float)) for v in vals]
    n = out[0].shape[0]
    if out[0].ndim != 1 or n == 0 or any(v.shape != (n,) for v in out):
        raise ValueError("inputs must be 1-D arrays of equal length")
    if not all(np.all(np.isfinite(v)) for v in out):
        raise ValueError("non-finite input")
    return out


def tera_wasserburg_to_wetherill(X, sX, Y, sY, rho):
    X, sX, Y, sY, rho = _arr(X, sX, Y, sY, rho)
    if np.any(X <= 0) or np.any(Y <= 0) or np.any(sX <= 0) or np.any(sY <= 0) or np.any(np.abs(rho) >= 1):
        raise ValueError("invalid ratios, errors or correlations")
    rX, rY = sX / X, sY / Y
    x, y = U_RATIO * Y / X, 1.0 / X
    rx = np.sqrt(rX ** 2 + rY ** 2 - 2.0 * rho * rX * rY)
    rxy = (rX ** 2 - rho * rX * rY) / (rx * rX)
    return np.column_stack([x, rx * x, y, rX * y, rxy])


def _weights(b, sx, sy, r):
    return 1.0 / (sy ** 2 + b * b * sx ** 2 - 2.0 * b * r * sx * sy)


def york_fit(x, sx, y, sy, rho):
    x, sx, y, sy, r = _arr(x, sx, y, sy, rho)
    n = len(x)
    if n < 3:
        raise ValueError("need at least three points")
    if np.any(sx <= 0) or np.any(sy <= 0) or np.any(np.abs(r) >= 1) or np.ptp(x) == 0:
        raise ValueError("invalid errors or correlations")

    def objective(theta):
        b = np.tan(theta)
        w = _weights(b, sx, sy, r)
        a = w @ (y - b * x) / w.sum()
        return w @ (y - a - b * x) ** 2

    grid = np.linspace(-np.pi / 2 + 1e-7, np.pi / 2 - 1e-7, 4001)
    k = int(np.argmin([objective(t) for t in grid]))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    g = 0.5 * (np.sqrt(5.0) - 1.0)
    for _ in range(300):
        c, d = hi - g * (hi - lo), lo + g * (hi - lo)
        if objective(c) < objective(d):
            hi = d
        else:
            lo = c
    theta = 0.5 * (lo + hi)
    # Polish with Newton steps on dS/dtheta (central differences) for full precision.
    for _ in range(20):
        h = 1e-5
        f0, fp, fm = objective(theta), objective(theta + h), objective(theta - h)
        d1, d2 = (fp - fm) / (2 * h), (fp - 2 * f0 + fm) / h ** 2
        if d2 <= 0:
            break
        step = -d1 / d2
        if objective(theta + step) > f0:
            break
        theta += step
        if abs(step) < 1e-15:
            break
    b = np.tan(theta)
    w = _weights(b, sx, sy, r)
    a = w @ (y - b * x) / w.sum()
    resid = y - a - b * x
    x_adj = x - (-b * sx ** 2 + r * sx * sy) * resid * w
    x0 = w @ x_adj / w.sum()
    vb = 1.0 / (w @ (x_adj - x0) ** 2)
    va = 1.0 / w.sum() + x0 ** 2 * vb
    return {"intercept": float(a), "slope": float(b),
            "cov": np.array([[va, -x0 * vb], [-x0 * vb, vb]]),
            "mswd": float(w @ resid ** 2 / (n - 2)), "n": n}


def concordia_intercepts(intercept, slope):
    a, b = float(intercept), float(slope)
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError("non-finite line")
    k = L235 / L238
    g = lambda s: s - 1.0 - a - b * (s ** k - 1.0)
    dg = lambda s: 1.0 - b * k * s ** (k - 1.0)
    s_lo, s_hi = np.exp(L238 * T_LO * 1e6), np.exp(L238 * T_HI * 1e6)
    grid = np.exp(np.linspace(np.log(s_lo), np.log(s_hi), 24001))
    gv = g(grid)
    roots = []
    for i in range(len(grid) - 1):
        if gv[i] == 0.0:
            roots.append(grid[i])
            continue
        if gv[i] * gv[i + 1] < 0:
            lo, hi = grid[i], grid[i + 1]
            s = 0.5 * (lo + hi)
            for _ in range(100):
                fs = g(s)
                if (fs > 0) == (g(lo) > 0):
                    lo = s
                else:
                    hi = s
                nxt = s - fs / dg(s)
                s = nxt if lo < nxt < hi else 0.5 * (lo + hi)
                if hi - lo < 1e-16 * s:
                    break
            roots.append(s)
    if gv[-1] == 0.0:
        roots.append(grid[-1])
    ages = np.log(np.array(roots, dtype=float)) / L238 / 1e6
    return np.sort(ages)


def intercept_age_sigmas(intercept, slope, cov, ages):
    c = np.asarray(cov, dtype=float)
    if c.shape != (2, 2) or not np.all(np.isfinite(c)) or c[0, 0] < 0 or c[1, 1] < 0:
        raise ValueError("invalid covariance")
    if not np.isclose(c[0, 1], c[1, 0], rtol=1e-12, atol=0.0):
        raise ValueError("covariance must be symmetric")
    t = np.asarray(ages, dtype=float) * 1e6
    if c[1, 1] > 0:
        x0 = -c[0, 1] / c[1, 1]
        var_y0 = c[0, 0] - x0 ** 2 * c[1, 1]
    else:
        x0, var_y0 = 0.0, c[0, 0]
    # Line: y = y0 + b (x - x0), y0 = a + b x0; y0 and b are uncorrelated.
    X = np.expm1(L235 * t)
    dfdt = (L238 * np.exp(L238 * t) - slope * L235 * np.exp(L235 * t)) * 1e6
    dt_dy0 = 1.0 / dfdt
    dt_db = (X - x0) / dfdt
    return np.sqrt(np.maximum(dt_dy0 ** 2 * var_y0 + dt_db ** 2 * c[1, 1], 0.0))


def discordia_ages(X, sX, Y, sY, rho):
    w = tera_wasserburg_to_wetherill(X, sX, Y, sY, rho)
    fit = york_fit(*w.T)
    ages = concordia_intercepts(fit["intercept"], fit["slope"])
    if len(ages) != 2:
        raise ValueError("discordia must cross concordia twice")
    sig = intercept_age_sigmas(fit["intercept"], fit["slope"], fit["cov"], ages)
    sig = sig * np.sqrt(max(fit["mswd"], 1.0))
    out = dict(fit)
    out.update({"ages": ages, "sigmas": sig})
    return out
