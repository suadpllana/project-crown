_T_TAG = "step_5"
_T_DEFAULT = "solution/step_5.py"


def _run(*args):
    return _fn("discordia_ages")(*args)


def _check_against_independent(out, data, what):
    ref = _independent_pipeline(*data)
    for key in ("intercept", "slope", "cov", "mswd", "ages", "sigmas", "n"):
        assert key in out, "missing key %s" % key
    assert int(out["n"]) == len(data[0])
    _close(out["intercept"], ref["intercept"], 1e-6, 1e-12, what + " intercept")
    _close(out["slope"], ref["slope"], 1e-6, 0.0, what + " slope")
    _close(out["cov"], ref["cov"], 1e-4, 0.0, what + " covariance")
    _close(out["mswd"], ref["mswd"], 1e-6, 1e-12, what + " MSWD")
    _close(out["ages"], ref["ages"], 0.0, 1e-3, what + " ages")
    _close(out["sigmas"], ref["sigmas"], 1e-4, 1e-6, what + " sigmas")
    return ref


def test_noise_free_pb_loss_array_recovers_true_ages():
    data = _synthetic_tw(0, 10, 2700.0, 450.0, dispersion=0.0)
    out = _run(*data)
    _close(out["ages"], [450.0, 2700.0], 0.0, 1e-4, "noise-free ages")
    assert abs(out["mswd"]) < 1e-9
    _check_against_independent(out, data, "noise-free")


def test_typical_scatter_matches_independent_pipeline():
    data = _synthetic_tw(11, 14, 2700.0, 450.0, dispersion=1.0)
    out = _run(*data)
    _check_against_independent(out, data, "LA-ICP-MS-like")


def test_overdispersed_data_inflate_sigmas_by_sqrt_mswd():
    data = _synthetic_tw(21, 12, 1850.0, 300.0, dispersion=2.5)
    out = _run(*data)
    assert out["mswd"] > 1.5, out["mswd"]
    ref = _check_against_independent(out, data, "overdispersed")
    raw = _sigma_fd(ref["intercept"], ref["slope"], ref["cov"], ref["ages"])
    _close(out["sigmas"], raw * np.sqrt(ref["mswd"]), 1e-4, 1e-6, "inflated sigmas")


def test_underdispersed_data_are_not_deflated():
    data = _synthetic_tw(31, 12, 1850.0, 300.0, dispersion=0.4)
    out = _run(*data)
    assert out["mswd"] < 1.0, out["mswd"]
    ref = _check_against_independent(out, data, "underdispersed")
    raw = _sigma_fd(ref["intercept"], ref["slope"], ref["cov"], ref["ages"])
    _close(out["sigmas"], raw, 1e-4, 1e-6, "un-inflated sigmas")


def test_invalid_arrays_raise_value_error():
    X, sX, Y, sY, rho = _synthetic_tw(0, 10, 2700.0, 450.0, dispersion=0.0)
    with _raises(ValueError):
        _run(X[:2], sX[:2], Y[:2], sY[:2], rho[:2])
    # Concordant-only data scattered about one point: the line has negative slope in the
    # Wetherill plot here and crosses concordia once -> no discordia ages.
    x0, y0 = _concordia(1000.0)
    x = x0 + np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    y = y0 - 0.002 * (x - x0)
    Xc, Yc = 1.0 / y, x / (_T_U * y)
    with _raises(ValueError):
        _run(Xc, 0.01 * Xc, Yc, 0.01 * Yc, np.zeros(5))


if __name__ == "__main__":
    sys.exit(1 if _run_all(dict(globals())) else 0)
