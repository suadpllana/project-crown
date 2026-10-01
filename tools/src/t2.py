_T_TAG = "step_2"
_T_DEFAULT = "solution/step_2.py"

_NU_L = np.array([1648.5, 3832.2, 3942.5])            # H2(16)O harmonic, cm^-1
_NU_H = np.array([1641.74985417, 3823.90459504, 3926.43261667])  # H2(18)O, same force field


def _lnf(*args):
    return _fn("log_reduced_partition_function_ratio")(*args)


def test_matches_closed_form_at_ambient_temperatures():
    T = np.array([250.0, 298.15, 400.0, 750.0, 1500.0])
    out = np.asarray(_lnf(_NU_L, _NU_H, T))
    _close(out, _lnf_sinh(_NU_L, _NU_H, T), 1e-7, 1e-12, "ln f vs closed form")
    # Published-scale sanity check: 1000 ln beta(18O, H2O vapour) ~ 65 at 25 C.
    assert 60.0 < 1000.0 * out[1] < 70.0


def test_scalar_temperature_returns_scalar():
    out = _lnf(_NU_L, _NU_H, 298.15)
    assert np.ndim(out) == 0
    _close(out, _lnf_sinh(_NU_L, _NU_H, 298.15)[0], 1e-7, 1e-12, "scalar T")


def test_low_temperature_limit_is_finite_and_exact():
    # At 1 K every u > 2000: f reduces to prod(nu*/nu) exp(sum(u - u*)/2).
    for T in (1.0, 5.0):
        expected = np.sum(np.log(_NU_H / _NU_L)) + _T_C2 * np.sum(_NU_L - _NU_H) / (2.0 * T)
        _close(_lnf(_NU_L, _NU_H, T), expected, 1e-9, 0.0, "ln f at %g K" % T)


def test_high_temperature_series():
    T = np.array([2.0e4, 1.0e5, 1.0e6])
    ul = _T_C2 * _NU_L[:, None] / T
    uh = _T_C2 * _NU_H[:, None] / T
    series = np.sum((ul ** 2 - uh ** 2) / 24.0 - (ul ** 4 - uh ** 4) / 2880.0
                    + (ul ** 6 - uh ** 6) / 181440.0, axis=0)
    _close(_lnf(_NU_L, _NU_H, T), series, 1e-6, 0.0, "ln f high-T series")


def test_mode_pairing_does_not_matter_and_identity_gives_zero():
    T = np.array([300.0, 600.0])
    a = np.asarray(_lnf(_NU_L, _NU_H, T))
    b = np.asarray(_lnf(_NU_L[::-1], _NU_H[[1, 2, 0]], T))
    _close(b, a, 1e-10, 1e-14, "permuted modes")
    _close(_lnf(_NU_L, _NU_L, T), np.zeros(2), 0.0, 1e-12, "identical isotopologues")
    assert np.all(a > 0.0)


def test_array_shape_is_preserved():
    T = np.array([[300.0, 400.0], [500.0, 600.0], [700.0, 800.0]])
    out = np.asarray(_lnf(_NU_L, _NU_H, T))
    _close(out, _lnf_sinh(_NU_L, _NU_H, T.ravel()).reshape(3, 2), 1e-7, 1e-12, "2-D T")


def test_invalid_inputs_raise_value_error():
    with _raises(ValueError):
        _lnf(_NU_L, _NU_H[:2], 300.0)
    with _raises(ValueError):
        _lnf(np.array([-100.0, 1600.0, 3800.0]), _NU_H, 300.0)
    with _raises(ValueError):
        _lnf(_NU_L, _NU_H, 0.0)
    with _raises(ValueError):
        _lnf(_NU_L, _NU_H, np.array([300.0, -5.0]))


if __name__ == "__main__":
    sys.exit(1 if _run_all(dict(globals())) else 0)
