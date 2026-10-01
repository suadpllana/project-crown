_T_TAG = "step_1"
_T_DEFAULT = "solution/step_1.py"


def _nu(*args):
    return _fn("vibrational_wavenumbers")(*args)


def test_diatomic_matches_analytic_harmonic_oscillator():
    # H35Cl-like oscillator, arbitrary orientation and origin.
    k = 0.3160
    x, hess = _diatomic(k, 1.2746, direction=(0.3, -0.5, 0.81), origin=(1.0, -2.0, 0.5))
    m = [_M["H"], _M["Cl35"]]
    out = np.asarray(_nu(x, m, hess))
    _close(out, [_diatomic_nu(k, *m)], 1e-6, 1e-4, "diatomic wavenumber")


def test_water_matches_wilson_gf_and_experiment():
    m = [_M["O16"], _M["H"], _M["H"]]
    x, hess, ref = _water(m)
    out = np.asarray(_nu(x, m, hess))
    _close(out, ref, 1e-6, 1e-4, "H2O wavenumbers vs GF")
    # The force field reproduces the experimental harmonic wavenumbers of H2O.
    _close(out, [1648.5, 3832.2, 3942.5], 0.0, 0.01, "H2O wavenumbers vs experiment")


def test_isotopologues_use_their_own_masses():
    for m in ([_M["O16"], _M["D"], _M["D"]], [_M["O16"], _M["H"], _M["D"]],
              [_M["O18"], _M["H"], _M["H"]]):
        x, hess, ref = _water(m)
        _close(np.asarray(_nu(x, m, hess)), ref, 1e-6, 1e-4, "water isotopologue %r" % (m,))
    # D2O is a prediction of the force field; it agrees with experiment to < 0.5 cm^-1.
    m = [_M["O16"], _M["D"], _M["D"]]
    x, hess, _ = _water(m)
    _close(np.asarray(_nu(x, m, hess)), [1206.4, 2763.8, 2888.8], 0.0, 0.5, "D2O vs experiment")


def test_rigid_motion_invariance():
    m = [_M["O16"], _M["H"], _M["D"]]
    x, hess, ref = _water(m)
    xr, hr = _rotate(x, hess, seed=11, shift=(3.0, -1.5, 7.25))
    _close(np.asarray(_nu(xr, m, hr)), ref, 1e-6, 1e-4, "rotated/translated HDO")


def test_linear_molecule_has_3n_minus_5_modes():
    for m in ([_M["O16"], _M["C12"], _M["O16"]], [_M["O16"], _M["C13"], _M["O18"]]):
        x, hess, ref = _co2(m)
        out = np.asarray(_nu(x, m, hess))
        _close(out, ref, 1e-6, 1e-4, "CO2 isotopologue %r" % (m,))
    m = [_M["O16"], _M["C12"], _M["O16"]]
    x, hess, _ = _co2(m)
    _close(np.asarray(_nu(x, m, hess)), [672.9, 672.9, 1354.0, 2396.3], 0.0, 0.01, "CO2 vs experiment")


def test_projection_removes_external_contamination():
    rng = np.random.default_rng(2024)
    x = rng.uniform(-1.6, 1.6, size=(5, 3))
    m = [_M["C12"], _M["H"], _M["O16"], _M["N14"], _M["S32"]]
    target = np.array([25.0, 140.0, 410.0, 780.0, 1050.0, 1300.0, 1620.0, 2900.0, 3400.0])
    hess = _designed_hessian(7, x, m, target, contamination_nu=300.0)
    out = np.asarray(_nu(x, m, hess))
    _close(out, target, 1e-6, 1e-3, "projected spectrum of contaminated Hessian")


def test_imaginary_mode_reported_negative():
    rng = np.random.default_rng(99)
    x = rng.uniform(-1.4, 1.4, size=(4, 3))
    m = [_M["C12"], _M["H"], _M["H"], _M["O16"]]
    target = np.array([-150.0, 300.0, 520.0, 990.0, 1210.0, 3050.0])
    hess = _designed_hessian(5, x, m, target, contamination_nu=120.0)
    out = np.asarray(_nu(x, m, hess))
    _close(out, target, 1e-6, 1e-3, "spectrum with an imaginary mode")


def test_invalid_inputs_raise_value_error():
    m = [_M["O16"], _M["H"], _M["H"]]
    x, hess, _ = _water(m)
    with _raises(ValueError):
        _nu(x, m[:2], hess)
    with _raises(ValueError):
        _nu(x, m, hess[:6, :6])
    with _raises(ValueError):
        _nu(x, [_M["O16"], 0.0, _M["H"]], hess)
    with _raises(ValueError):
        _nu(x[:1], m[:1], hess[:3, :3])


if __name__ == "__main__":
    sys.exit(1 if _run_all(dict(globals())) else 0)
