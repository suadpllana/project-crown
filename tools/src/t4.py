_T_TAG = "step_4"
_T_DEFAULT = "solution/step_4.py"

_HBAR = _T_H / (2.0 * np.pi)
_TRACE_SI = _T_EH / _T_A0 ** 2      # Eh/bohr^2 -> J/m^2


def _coef(*args):
    return _fn("high_temperature_coefficient")(*args)


def _force_constant_route(hess, ml, mh):
    """Bigeleisen-Mayer high-temperature limit from Cartesian force-constant blocks:
    K = hbar^2 / (24 kB^2 n) * sum_sites (1/m - 1/m*) tr(H_aa)."""
    ml, mh = np.asarray(ml, float), np.asarray(mh, float)
    sites = np.flatnonzero(ml != mh)
    total = 0.0
    for a in sites:
        tr = np.trace(hess[3 * a:3 * a + 3, 3 * a:3 * a + 3]) * _TRACE_SI
        total += (1.0 / ml[a] - 1.0 / mh[a]) / _T_AMU * tr
    return _HBAR ** 2 * total / (24.0 * _T_KB ** 2 * sites.size)


def test_diatomic_analytic():
    k = 1.2066
    x, hess = _diatomic(k, 1.1283)
    ml, mh = [_M["C12"], _M["O16"]], [_M["C12"], _M["O18"]]
    nl, nh = _diatomic_nu(k, *ml), _diatomic_nu(k, *mh)
    expected = _T_C2 ** 2 * (nl ** 2 - nh ** 2) / 24.0
    _close(_coef(x, ml, mh, hess), expected, 1e-6, 0.0, "diatomic K")
    _close(_coef(x, ml, mh, hess), _force_constant_route(hess, ml, mh), 1e-6, 0.0, "diatomic K (blocks)")


def test_water_oxygen_and_hydrogen_sites():
    ml = [_M["O16"], _M["H"], _M["H"]]
    x, hess, _ = _water(ml)
    for mh in ([_M["O18"], _M["H"], _M["H"]], [_M["O16"], _M["D"], _M["H"]],
               [_M["O16"], _M["D"], _M["D"]]):
        _close(_coef(x, ml, mh, hess), _force_constant_route(hess, ml, mh), 1e-6, 0.0,
               "water K for %r" % (mh,))


def test_carbon_dioxide_sites_and_rigid_motion():
    ml = [_M["O16"], _M["C12"], _M["O16"]]
    x, hess, _ = _co2(ml)
    xr, hr = _rotate(x, hess, seed=3, shift=(0.4, 0.2, -1.0))
    for mh in ([_M["O16"], _M["C13"], _M["O16"]], [_M["O18"], _M["C12"], _M["O18"]]):
        exp = _force_constant_route(hess, ml, mh)
        _close(_coef(xr, ml, mh, hr), exp, 1e-6, 0.0, "CO2 K for %r" % (mh,))


def test_is_the_limit_of_t_squared_ln_beta():
    ml = [_M["O16"], _M["H"], _M["H"]]
    mh = [_M["O16"], _M["H"], _M["D"]]
    x, hess, nl = _water(ml)
    _, _, nh = _water(mh)
    T = 2.0e5
    lim = T ** 2 * _lnf_full_partition_functions(x, ml, mh, nl, nh, T)[0]
    _close(_coef(x, ml, mh, hess), lim, 1e-4, 0.0, "K vs T^2 ln beta at 2e5 K")


def test_no_substitution_raises_value_error():
    ml = [_M["O16"], _M["H"], _M["H"]]
    x, hess, _ = _water(ml)
    with _raises(ValueError):
        _coef(x, ml, ml, hess)


if __name__ == "__main__":
    sys.exit(1 if _run_all(dict(globals())) else 0)
