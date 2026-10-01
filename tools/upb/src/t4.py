_T_TAG = "step_4"
_T_DEFAULT = "solution/step_4.py"


def _sig(*args):
    return np.asarray(_fn("intercept_age_sigmas")(*args), dtype=float)


def _york_like_cov(a, b, xbar, sb):
    # Covariance with the structure produced by a York fit: cov(a,b) = -xbar var_b.
    return np.array([[(0.3 * sb) ** 2 + xbar ** 2 * sb ** 2, -xbar * sb ** 2], [-xbar * sb ** 2, sb ** 2]])


def test_matches_finite_difference_propagation():
    for t_low, t_up, xbar, sb in ((450.0, 2700.0, 6.0, 2e-4), (0.0, 1850.0, 3.0, 5e-4), (-300.0, 3200.0, 10.0, 1e-4)):
        a, b = _line_through(t_low, t_up)
        cov = _york_like_cov(a, b, xbar, sb)
        ages = np.array([t_low, t_up])
        _close(_sig(a, b, cov, ages), _sigma_fd(a, b, cov, ages), 1e-4, 1e-6, "sigma for %s" % ((t_low, t_up),))


def test_covariance_term_is_required():
    a, b = _line_through(450.0, 2700.0)
    cov = _york_like_cov(a, b, 6.0, 2e-4)
    ages = np.array([450.0, 2700.0])
    full = _sig(a, b, cov, ages)
    diag = _sigma_fd(a, b, np.diag(np.diag(cov)), ages)
    assert np.all(np.abs(full - diag) > 0.05 * diag), (full, diag)


def test_scaling_and_zero_covariance():
    a, b = _line_through(800.0, 2100.0)
    cov = _york_like_cov(a, b, 4.0, 3e-4)
    ages = np.array([800.0, 2100.0])
    s1 = _sig(a, b, cov, ages)
    _close(_sig(a, b, 4.0 * cov, ages), 2.0 * s1, 1e-10, 0.0, "sigma scales with sqrt(cov)")
    _close(_sig(a, b, np.zeros((2, 2)), ages), np.zeros(2), 0.0, 1e-12, "zero covariance")
    out = _sig(a, b, cov, np.array([2100.0]))
    _close(out, s1[1:], 1e-10, 0.0, "single age")


def test_invalid_covariance_raises_value_error():
    a, b = _line_through(450.0, 2700.0)
    with _raises(ValueError):
        _sig(a, b, np.eye(3), np.array([450.0]))
    with _raises(ValueError):
        _sig(a, b, np.array([[-1e-6, 0.0], [0.0, 1e-8]]), np.array([450.0]))


if __name__ == "__main__":
    sys.exit(1 if _run_all(dict(globals())) else 0)
