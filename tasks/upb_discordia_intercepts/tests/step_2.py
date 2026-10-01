"""Tests for Step 2: York (2004) regression with correlated errors.

Expected values come from published results, analytic constructions and an
independent pipeline (log-space error propagation, maximum-likelihood line by direct
minimisation, geometric adjusted points, dense-scan roots, finite differences).
The implementation under test is never used to produce a target.
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Locating the implementation under test:
# 1. functions already defined in this module's globals (candidate code concatenated
#    above the tests), else 2. the file named by $CROWN_IMPL, else 3. _T_DEFAULT.
# ---------------------------------------------------------------------------
_T_MODULE = None


def _t_load():
    global _T_MODULE
    if _T_MODULE is None:
        path = os.environ.get("CROWN_IMPL")
        if path is None:
            path = str(Path(__file__).resolve().parents[1] / _T_DEFAULT)
        spec = importlib.util.spec_from_file_location("_crown_impl_" + _T_TAG, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _T_MODULE = mod
    return _T_MODULE


def _fn(name):
    g = globals()
    if name in g and callable(g[name]):
        return g[name]
    return getattr(_t_load(), name)


# ---------------------------------------------------------------------------
# Independent reference physics/statistics. Nothing here calls the code under test.
# ---------------------------------------------------------------------------
_T_L8 = 1.55125e-10      # 238U decay constant, 1/a (Jaffey et al. 1971)
_T_L5 = 9.8485e-10       # 235U decay constant, 1/a
_T_U = 137.818           # 238U/235U (Hiess et al. 2012)

# Pearson (1901) data with York (1966) weights: the standard benchmark for York fits.
_T_PY_X = np.array([0.0, 0.9, 1.8, 2.6, 3.3, 4.4, 5.2, 6.1, 6.5, 7.4])
_T_PY_Y = np.array([5.9, 5.4, 4.4, 4.6, 3.5, 3.7, 2.8, 2.8, 2.4, 1.5])
_T_PY_WX = np.array([1000.0, 1000.0, 500.0, 800.0, 200.0, 80.0, 60.0, 20.0, 1.8, 1.0])
_T_PY_WY = np.array([1.0, 1.8, 4.0, 8.0, 20.0, 20.0, 70.0, 70.0, 100.0, 500.0])


def _concordia(t_ma):
    t = np.asarray(t_ma, float) * 1e6
    return np.expm1(_T_L5 * t), np.expm1(_T_L8 * t)


def _line_through(t1, t2):
    x1, y1 = _concordia(t1)
    x2, y2 = _concordia(t2)
    b = (y2 - y1) / (x2 - x1)
    return float(y1 - b * x1), float(b)


def _wetherill_from_tw_logspace(X, sX, Y, sY, rho):
    """Independent derivation in relative (logarithmic) errors:
    ln x = ln U + ln Y - ln X, ln y = -ln X."""
    X, sX, Y, sY, rho = (np.asarray(v, float) for v in (X, sX, Y, sY, rho))
    rX, rY = sX / X, sY / Y
    x, y = _T_U * Y / X, 1.0 / X
    rx = np.sqrt(rX ** 2 + rY ** 2 - 2 * rho * rX * rY)
    ry = rX
    rxy = (rX ** 2 - rho * rX * rY) / (rx * ry)
    return np.column_stack([x, rx * x, y, ry * y, rxy])


def _tw_from_wetherill(x, sx, y, sy, rxy):
    """Inverse transform X = 1/y, Y = x/(U y) with linear propagation."""
    x, sx, y, sy, rxy = (np.asarray(v, float) for v in (x, sx, y, sy, rxy))
    X, Y = 1.0 / y, x / (_T_U * y)
    J = np.zeros((len(x), 2, 2))
    J[:, 0, 1] = -1.0 / y ** 2
    J[:, 1, 0] = 1.0 / (_T_U * y)
    J[:, 1, 1] = -x / (_T_U * y ** 2)
    C = np.zeros((len(x), 2, 2))
    C[:, 0, 0], C[:, 1, 1] = sx ** 2, sy ** 2
    C[:, 0, 1] = C[:, 1, 0] = rxy * sx * sy
    D = J @ C @ np.transpose(J, (0, 2, 1))
    sX, sY = np.sqrt(D[:, 0, 0]), np.sqrt(D[:, 1, 1])
    return X, sX, Y, sY, D[:, 0, 1] / (sX * sY)


def _york_weights(b, sx, sy, r):
    return 1.0 / (sy ** 2 + b * b * sx ** 2 - 2 * b * r * sx * sy)


def _york_ml(x, sx, y, sy, r):
    """Maximum-likelihood line by direct minimisation of
    S(b) = min_a sum W_i(b) (y_i - a - b x_i)^2 over the slope angle (grid + golden
    section). Different algorithm from York's fixed-point iteration."""
    x, sx, y, sy, r = (np.asarray(v, float) for v in (x, sx, y, sy, r))

    def s_of(theta):
        b = np.tan(theta)
        w = _york_weights(b, sx, sy, r)
        a = w @ (y - b * x) / w.sum()
        return w @ (y - a - b * x) ** 2

    grid = np.linspace(-np.pi / 2 + 1e-6, np.pi / 2 - 1e-6, 20001)
    vals = np.array([s_of(t) for t in grid])
    k = int(np.argmin(vals))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    g = (np.sqrt(5) - 1) / 2
    c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    for _ in range(200):
        if s_of(c) < s_of(d):
            hi = d
        else:
            lo = c
        c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    b = np.tan(0.5 * (lo + hi))
    w = _york_weights(b, sx, sy, r)
    a = w @ (y - b * x) / w.sum()
    return a, b, w @ (y - a - b * x) ** 2 / (len(x) - 2)


def _york_cov_geometric(a, b, x, sx, y, sy, r):
    """York et al. (2004) standard errors evaluated at the least-squares-adjusted points,
    with the adjusted points obtained geometrically as the Mahalanobis projection of each
    datum onto the line (independent of the beta_i expression)."""
    x, sx, y, sy, r = (np.asarray(v, float) for v in (x, sx, y, sy, r))
    w = _york_weights(b, sx, sy, r)
    resid = y - a - b * x
    # p_adj = p - C n (n.p - a) / (n^T C n), n = (-b, 1)
    cn_x = -b * sx ** 2 + r * sx * sy
    x_adj = x - cn_x * resid * w
    xb = w @ x_adj / w.sum()
    var_b = 1.0 / (w @ (x_adj - xb) ** 2)
    var_a = 1.0 / w.sum() + xb ** 2 * var_b
    return np.array([[var_a, -xb * var_b], [-xb * var_b, var_b]])


def _roots_dense(a, b, lo=-1000.0, hi=5000.0, step=0.25):
    """All sign changes of the concordia misfit on a dense grid, refined by bisection."""
    f = lambda t: np.expm1(_T_L8 * t * 1e6) - a - b * np.expm1(_T_L5 * t * 1e6)
    grid = np.arange(lo, hi + step / 2, step)
    fv = f(grid)
    out = []
    for i in np.flatnonzero(np.sign(fv[:-1]) * np.sign(fv[1:]) <= 0):
        l, h = grid[i], grid[i + 1]
        if fv[i] == 0:
            out.append(l)
            continue
        for _ in range(100):
            m = 0.5 * (l + h)
            if np.sign(f(m)) == np.sign(f(l)):
                l = m
            else:
                h = m
        out.append(0.5 * (l + h))
    out = np.sort(np.array(out, dtype=float))
    if out.size > 1:
        out = out[np.concatenate([[True], np.diff(out) > 1e-6])]
    return out


def _sigma_fd(a, b, cov, ages):
    """Age uncertainties by central finite differences of the roots wrt (a, b)."""
    sig = []
    for k, t0 in enumerate(ages):
        def root(aa, bb):
            r = _roots_dense(aa, bb, t0 - 50.0, t0 + 50.0, 1.0)
            return r[np.argmin(np.abs(r - t0))]
        # Steps of 1e-3 standard deviations: small enough for the linearisation, large
        # enough that bisection round-off (~1e-12 Ma) is negligible.
        ha = 1e-3 * np.sqrt(cov[0, 0]) if cov[0, 0] > 0 else 1e-9
        hb = 1e-3 * np.sqrt(cov[1, 1]) if cov[1, 1] > 0 else 1e-9 * abs(b)
        da = (root(a + ha, b) - root(a - ha, b)) / (2 * ha)
        db = (root(a, b + hb) - root(a, b - hb)) / (2 * hb)
        g = np.array([da, db])
        sig.append(np.sqrt(g @ cov @ g))
    return np.array(sig)


def _synthetic_tw(seed, n, t_upper, t_lower, rel_sX=0.012, rel_sY=0.008, rho_tw=0.1,
                  dispersion=1.0, frac=(0.15, 0.95)):
    """Synthetic LA-ICP-MS-like zircon analyses on a Pb-loss discordia between the
    concordia points t_lower and t_upper (Ma), in Tera-Wasserburg form. Stated errors are
    rel_sX, rel_sY (1 sigma) with correlation rho_tw; the actual scatter is the stated
    errors times `dispersion` (0 = exactly collinear)."""
    rng = np.random.default_rng(seed)
    f = np.linspace(frac[0], frac[1], n)
    xl, yl = _concordia(t_lower)
    xu, yu = _concordia(t_upper)
    x, y = xl + f * (xu - xl), yl + f * (yu - yl)
    X, Y = 1.0 / y, x / (_T_U * y)
    sX, sY = rel_sX * X, rel_sY * Y
    rho = np.full(n, rho_tw)
    if dispersion:
        z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
        eX = sX * z1
        eY = sY * (rho_tw * z1 + np.sqrt(1 - rho_tw ** 2) * z2)
        X, Y = X + dispersion * eX, Y + dispersion * eY
    return X, sX, Y, sY, rho


def _independent_pipeline(X, sX, Y, sY, rho):
    w = _wetherill_from_tw_logspace(X, sX, Y, sY, rho)
    a, b, mswd = _york_ml(*w.T)
    cov = _york_cov_geometric(a, b, *w.T)
    ages = _roots_dense(a, b)
    sig = _sigma_fd(a, b, cov, ages)
    if mswd > 1:
        sig = sig * np.sqrt(mswd)
    return {"intercept": a, "slope": b, "cov": cov, "mswd": mswd, "ages": ages, "sigmas": sig}


def _close(actual, expected, rtol, atol, what):
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    assert a.shape == e.shape, "%s: shape %s, expected %s" % (what, a.shape, e.shape)
    assert np.all(np.isfinite(a)), "%s: non-finite values %r" % (what, a)
    ok = np.abs(a - e) <= atol + rtol * np.abs(e)
    assert np.all(ok), "%s: max abs error %.3e (got %r, expected %r)" % (
        what, np.max(np.abs(a - e)), a, e)


class _raises:
    def __init__(self, *exc):
        self.exc = exc or (Exception,)

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError("expected %s to be raised" % (self.exc,))
        return issubclass(et, self.exc)


def _run_all(namespace):
    failed = 0
    for name, fn in sorted(namespace.items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL", name, "-", type(exc).__name__, exc)
    print("%d failed" % failed)
    return failed


_T_TAG = "step_2"
_T_DEFAULT = "solution/step_2.py"


def _york(*args):
    return _fn("york_fit")(*args)


def _corr_data(seed=1, n=12):
    rng = np.random.default_rng(seed)
    xt = np.linspace(1.0, 10.0, n)
    sx = rng.uniform(0.05, 0.3, n)
    sy = rng.uniform(0.05, 0.4, n)
    r = rng.uniform(-0.8, 0.8, n)
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    x = xt + sx * z1
    y = 2.0 + 0.7 * xt + sy * (r * z1 + np.sqrt(1 - r ** 2) * z2)
    return x, sx, y, sy, r


def test_pearson_york_benchmark():
    sx, sy = 1.0 / np.sqrt(_T_PY_WX), 1.0 / np.sqrt(_T_PY_WY)
    out = _york(_T_PY_X, sx, _T_PY_Y, sy, np.zeros(10))
    # Published values (York et al. 2004; bfsl R package reference output).
    _close(out["intercept"], 5.47991, 0.0, 6e-6, "intercept")
    _close(out["slope"], -0.48053, 0.0, 6e-6, "slope")
    cov = np.asarray(out["cov"])
    assert cov.shape == (2, 2)
    _close(np.sqrt(cov[0, 0]), 0.29497, 0.0, 6e-6, "sigma intercept")
    _close(np.sqrt(cov[1, 1]), 0.05799, 0.0, 6e-6, "sigma slope")
    _close(out["mswd"], 11.87 / 8.0, 0.0, 1e-3, "MSWD (chi2 11.87 on 8 dof)")
    assert int(out["n"]) == 10


def test_correlated_errors_match_maximum_likelihood_line():
    x, sx, y, sy, r = _corr_data()
    out = _york(x, sx, y, sy, r)
    a, b, mswd = _york_ml(x, sx, y, sy, r)
    _close(out["slope"], b, 1e-7, 0.0, "slope")
    _close(out["intercept"], a, 1e-7, 0.0, "intercept")
    _close(out["mswd"], mswd, 1e-6, 0.0, "MSWD")
    _close(out["cov"], _york_cov_geometric(a, b, x, sx, y, sy, r), 1e-5, 0.0, "covariance")
    assert np.asarray(out["cov"])[0, 1] < 0.0   # positive x-centroid -> negative cov(a, b)


def test_axis_swap_symmetry():
    # The York line is symmetric in x and y: swapping axes inverts the slope exactly,
    # gives sigma_b' = sigma_b / b^2 and leaves MSWD unchanged. OLS/WLS fail this.
    x, sx, y, sy, r = _corr_data(seed=5)
    f = _york(x, sx, y, sy, r)
    g = _york(y, sy, x, sx, r)
    b = f["slope"]
    _close(g["slope"], 1.0 / b, 1e-7, 0.0, "swapped slope")
    _close(g["intercept"], -f["intercept"] / b, 1e-7, 1e-10, "swapped intercept")
    _close(np.sqrt(np.asarray(g["cov"])[1, 1]), np.sqrt(np.asarray(f["cov"])[1, 1]) / b ** 2, 1e-5, 0.0,
           "swapped sigma slope")
    _close(g["mswd"], f["mswd"], 1e-6, 0.0, "swapped MSWD")


def test_negligible_x_errors_reduce_to_weighted_least_squares():
    x = np.array([1.0, 2.0, 3.5, 4.0, 6.0, 7.5])
    y = np.array([2.9, 5.2, 7.9, 9.1, 13.2, 15.8])
    sy = np.array([0.1, 0.2, 0.15, 0.3, 0.2, 0.25])
    out = _york(x, 1e-9 * np.ones(6), y, sy, np.zeros(6))
    w = 1 / sy ** 2
    d = w.sum() * (w @ x ** 2) - (w @ x) ** 2
    b = (w.sum() * (w @ (x * y)) - (w @ x) * (w @ y)) / d
    a = ((w @ x ** 2) * (w @ y) - (w @ x) * (w @ (x * y))) / d
    cov = np.array([[w @ x ** 2, -(w @ x)], [-(w @ x), w.sum()]]) / d
    _close(out["slope"], b, 1e-7, 0.0, "WLS slope")
    _close(out["intercept"], a, 1e-7, 1e-9, "WLS intercept")
    _close(out["cov"], cov, 1e-5, 0.0, "WLS covariance")
    _close(out["mswd"], w @ (y - a - b * x) ** 2 / 4, 1e-6, 0.0, "WLS MSWD")


def test_collinear_data_and_error_scaling():
    x = np.array([0.2, 1.1, 2.0, 3.7, 5.0])
    y = 1.5 - 0.3 * x
    sx, sy, r = np.full(5, 0.05), np.full(5, 0.04), np.full(5, 0.3)
    out = _york(x, sx, y, sy, r)
    _close([out["intercept"], out["slope"]], [1.5, -0.3], 1e-7, 1e-10, "exact line")
    assert abs(out["mswd"]) < 1e-9
    xx, sxx, yy, syy, rr = _corr_data(seed=9)
    f1 = _york(xx, sxx, yy, syy, rr)
    f3 = _york(xx, 3 * sxx, yy, 3 * syy, rr)
    _close(f3["slope"], f1["slope"], 1e-7, 0.0, "slope under error scaling")
    _close(f3["cov"], 9 * np.asarray(f1["cov"]), 1e-5, 0.0, "cov under error scaling")
    _close(f3["mswd"], f1["mswd"] / 9, 1e-6, 0.0, "MSWD under error scaling")


def test_invalid_inputs_raise_value_error():
    x, sx, y, sy, r = _corr_data()
    with _raises(ValueError):
        _york(x[:2], sx[:2], y[:2], sy[:2], r[:2])
    with _raises(ValueError):
        _york(x, sx, y, sy, np.where(r > 0, 1.0, r))
    with _raises(ValueError):
        _york(x, 0 * sx, y, sy, r)
    with _raises(ValueError):
        _york(x, sx, y[:-1], sy, r)


if __name__ == "__main__":
    sys.exit(1 if _run_all(dict(globals())) else 0)
