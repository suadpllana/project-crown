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
