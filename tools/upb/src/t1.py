_T_TAG = "step_1"
_T_DEFAULT = "solution/step_1.py"


def _conv(*args):
    return _fn("tera_wasserburg_to_wetherill")(*args)


def _tw_case():
    X = np.array([3.25, 2.10, 4.80, 6.40, 10.0])
    Y = np.array([0.1450, 0.1820, 0.1210, 0.0980, 0.0702])
    sX = np.array([0.04, 0.025, 0.07, 0.10, 0.11])
    sY = np.array([0.0015, 0.0011, 0.0016, 0.0010, 0.0009])
    rho = np.array([0.10, -0.35, 0.0, 0.62, -0.05])
    return X, sX, Y, sY, rho


def test_matches_relative_error_derivation():
    out = np.asarray(_conv(*_tw_case()))
    _close(out, _wetherill_from_tw_logspace(*_tw_case()), 1e-9, 1e-14, "Wetherill ratios and errors")


def test_uses_stated_uranium_ratio():
    X, sX, Y, sY, rho = _tw_case()
    out = np.asarray(_conv(X, sX, Y, sY, rho))
    # 207Pb/235U = (207Pb/206Pb)(238U/235U)/(238U/206Pb) with 238U/235U = 137.818;
    # the older 137.88 shifts x by 4.5e-4 relative and is rejected here.
    _close(out[:, 0], 137.818 * Y / X, 1e-9, 0.0, "207Pb/235U")
    _close(out[:, 2], 1.0 / X, 1e-9, 0.0, "206Pb/238U")


def test_round_trip_recovers_tera_wasserburg_covariance():
    case = _tw_case()
    back = _tw_from_wetherill(*np.asarray(_conv(*case)).T)
    for got, want, name in zip(back, case, ("X", "sX", "Y", "sY", "rho")):
        _close(got, want, 1e-9, 1e-12, "round trip " + name)


def test_error_correlation_is_strong_and_positive_for_uncorrelated_tw_data():
    # With rho_TW = 0 the Wetherill correlation is rX / sqrt(rX^2 + rY^2) > 0.
    X = np.array([4.0]); Y = np.array([0.12])
    out = np.asarray(_conv(X, 0.02 * X, Y, 0.01 * Y, np.array([0.0])))
    assert out.shape == (1, 5)
    _close(out[0, 4], 0.02 / np.sqrt(0.02 ** 2 + 0.01 ** 2), 1e-9, 0.0, "rho_xy")
    _close(out[0, 1] / out[0, 0], np.sqrt(0.02 ** 2 + 0.01 ** 2), 1e-9, 0.0, "relative sx")


def test_invalid_inputs_raise_value_error():
    X, sX, Y, sY, rho = _tw_case()
    with _raises(ValueError):
        _conv(X, sX, Y, sY, np.where(rho > 0.5, 1.0, rho))
    with _raises(ValueError):
        _conv(X, -sX, Y, sY, rho)
    with _raises(ValueError):
        _conv(X[:3], sX, Y, sY, rho)
    with _raises(ValueError):
        _conv(-X, sX, Y, sY, rho)


if __name__ == "__main__":
    sys.exit(1 if _run_all(dict(globals())) else 0)
