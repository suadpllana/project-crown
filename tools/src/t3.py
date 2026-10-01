_T_TAG = "step_3"
_T_DEFAULT = "solution/step_3.py"

_TEMPS = np.array([273.15, 298.15, 373.15, 573.15, 1000.0])


def _lnb(*args):
    return _fn("log_beta")(*args)


def _expected(builder, ml, mh, n, T):
    x, hess, nu_l = builder(ml)
    _, _, nu_h = builder(mh)
    return x, hess, _lnf_full_partition_functions(x, ml, mh, nu_l, nu_h, T) / n


def test_diatomic_carbon_monoxide_13c():
    k = 1.2066
    x, hess = _diatomic(k, 1.1283, direction=(1.0, 1.0, 0.0))
    ml, mh = [_M["C12"], _M["O16"]], [_M["C13"], _M["O16"]]
    nl, nh = _diatomic_nu(k, *ml), _diatomic_nu(k, *mh)
    _close(_lnb(x, ml, mh, hess, _TEMPS), _lnf_sinh([nl], [nh], _TEMPS), 1e-6, 1e-10, "13C in CO")


def test_water_oxygen_18():
    ml = [_M["O16"], _M["H"], _M["H"]]
    mh = [_M["O18"], _M["H"], _M["H"]]
    x, hess, exp = _expected(_water, ml, mh, 1, _TEMPS)
    _close(_lnb(x, ml, mh, hess, _TEMPS), exp, 1e-6, 1e-10, "18O in H2O")


def test_water_deuterium_single_and_double_substitution():
    ml = [_M["O16"], _M["H"], _M["H"]]
    for mh, n in (([_M["O16"], _M["H"], _M["D"]], 1), ([_M["O16"], _M["D"], _M["D"]], 2)):
        x, hess, exp = _expected(_water, ml, mh, n, _TEMPS)
        _close(_lnb(x, ml, mh, hess, _TEMPS), exp, 1e-6, 1e-10, "D in water, n=%d" % n)


def test_carbon_dioxide_isotopologues():
    ml = [_M["O16"], _M["C12"], _M["O16"]]
    cases = (([_M["O16"], _M["C13"], _M["O16"]], 1),
             ([_M["O16"], _M["C12"], _M["O18"]], 1),
             ([_M["O18"], _M["C12"], _M["O18"]], 2))
    results = []
    for mh, n in cases:
        x, hess, exp = _expected(_co2, ml, mh, n, _TEMPS)
        out = np.asarray(_lnb(x, ml, mh, hess, _TEMPS))
        _close(out, exp, 1e-6, 1e-10, "CO2 isotopologue n=%d" % n)
        results.append(out)
    # Singly and doubly 18O-substituted CO2 give slightly different per-atom beta.
    assert np.all(results[2] > results[1])


def test_scalar_temperature_and_shape():
    ml = [_M["O16"], _M["H"], _M["H"]]
    mh = [_M["O18"], _M["H"], _M["H"]]
    x, hess, exp = _expected(_water, ml, mh, 1, 298.15)
    out = _lnb(x, ml, mh, hess, 298.15)
    assert np.ndim(out) == 0
    _close(out, exp[0], 1e-6, 1e-10, "scalar T")


def test_invalid_substitutions_raise_value_error():
    ml = [_M["O16"], _M["H"], _M["H"]]
    x, hess, _ = _water(ml)
    with _raises(ValueError):
        _lnb(x, ml, ml, hess, 300.0)
    with _raises(ValueError):
        _lnb(x, ml, [_M["O18"], _M["D"], _M["H"]], hess, 300.0)
    with _raises(ValueError):
        _lnb(x, ml, [_M["O18"], _M["H"]], hess, 300.0)


if __name__ == "__main__":
    sys.exit(1 if _run_all(dict(globals())) else 0)
