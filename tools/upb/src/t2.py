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
