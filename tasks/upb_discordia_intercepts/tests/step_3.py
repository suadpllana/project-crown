"""Tests for Step 3: intersections of a line with the Wetherill concordia.

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


_T_TAG = "step_3"
_T_DEFAULT = "solution/step_3.py"


def _roots(a, b):
    return np.asarray(_fn("concordia_intercepts")(a, b), dtype=float)


def test_line_through_two_concordia_points():
    for t_low, t_up in ((450.0, 2700.0), (0.0, 1850.0), (-300.0, 3200.0), (1200.0, 4400.0), (25.0, 60.0)):
        a, b = _line_through(t_low, t_up)
        _close(_roots(a, b), [t_low, t_up], 0.0, 1e-4, "intercepts of line through %s" % ((t_low, t_up),))


def test_negative_slope_has_a_single_intercept():
    # A line of negative slope crosses the monotonic concordia exactly once.
    x1, y1 = _concordia(1500.0)
    out = _roots(y1 + 0.05 * x1, -0.05)
    _close(out, [1500.0], 0.0, 1e-4, "single intercept")


def test_intercepts_outside_the_window_are_excluded():
    a, b = _line_through(-1500.0, 2000.0)
    _close(_roots(a, b), [2000.0], 0.0, 1e-4, "one intercept inside [-1000, 5000] Ma")
    a, b = _line_through(300.0, 5600.0)
    _close(_roots(a, b), [300.0], 0.0, 1e-4, "upper intercept beyond 5000 Ma excluded")


def test_line_missing_concordia_returns_empty():
    a, b = _line_through(450.0, 2700.0)
    out = _roots(a + 2.0, b)        # shifted entirely above the concordia arc
    assert out.shape == (0,), out


def test_agrees_with_dense_scan():
    rng = np.random.default_rng(3)
    for _ in range(20):
        t_low, t_up = np.sort(rng.uniform(-900.0, 4900.0, 2))
        if t_up - t_low < 5.0:
            continue
        a, b = _line_through(t_low, t_up)
        _close(_roots(a, b), _roots_dense(a, b), 0.0, 1e-4, "random line")


if __name__ == "__main__":
    sys.exit(1 if _run_all(dict(globals())) else 0)
