_T_TAG = "step_5"
_T_DEFAULT = "solution/step_5.py"
#BODY#

_TEMPS = np.linspace(273.15, 1273.15, 11)


def _frac(*args):
    return _fn("isotope_fractionation")(*args)


def test_co2_water_oxygen_18_against_independent_route():
    out = _frac(_species(_co2, _C16, _C18), _species(_water, _W16, _W18), _TEMPS)
    ba = 1000 * _independent_ln_beta(_co2, _C16, _C18, _TEMPS)
    bb = 1000 * _independent_ln_beta(_water, _W16, _W18, _TEMPS)
    _close(out["thousand_ln_beta_a"], ba, 1e-6, 1e-7, "1000 ln beta CO2")
    _close(out["thousand_ln_beta_b"], bb, 1e-6, 1e-7, "1000 ln beta H2O")
    _close(out["thousand_ln_alpha"], ba - bb, 1e-6, 1e-6, "1000 ln alpha")
    x = 1000.0 / _TEMPS
    fit = np.linalg.lstsq(np.column_stack([x ** 2, x, np.ones_like(x)]), ba - bb, rcond=None)[0]
    _close(out["fit"], fit, 1e-5, 1e-5, "fit coefficients (A, B, C)")
    a_ht = 1000 * (_ffc(_co2, _C16, _C18) - _ffc(_water, _W16, _W18)) / 1e6
    _close(out["a_high_T"], a_ht, 1e-6, 0.0, "high-temperature A")


def test_published_co2_water_fractionation_at_25C():
    # Experiment: alpha(CO2(g)-H2O(l)) = 1.04115 (Brenninkmeijer et al. 1983) and
    # alpha(H2O(l)-H2O(v)) = 1.0094 (Majoube 1971) -> 1000 ln alpha(CO2(g)-H2O(v)) ~ 49.7.
    # The harmonic approximation is expected within ~4 permil.
    out = _frac(_species(_co2, _C16, _C18), _species(_water, _W16, _W18), [288.15, 298.15, 308.15])
    val = float(np.asarray(out["thousand_ln_alpha"])[1])
    assert 45.5 < val < 53.5, val


def test_antisymmetry_and_self_fractionation():
    a = _species(_co2, _C16, _C18)
    b = _species(_water, _W16, _W18)
    ab = _frac(a, b, _TEMPS)
    ba = _frac(b, a, _TEMPS)
    _close(ba["thousand_ln_alpha"], -np.asarray(ab["thousand_ln_alpha"]), 1e-9, 1e-9, "antisymmetry")
    _close(ba["fit"], -np.asarray(ab["fit"]), 1e-6, 1e-6, "antisymmetric fit")
    aa = _frac(a, a, _TEMPS)
    _close(aa["thousand_ln_alpha"], np.zeros_like(_TEMPS), 0.0, 1e-9, "self fractionation")


def test_hydrogen_isotopes_water_vs_hydrogen_gas():
    hd = [_M["H"], _M["D"]]
    hh = [_M["H"], _M["H"]]
    out = _frac(_species(_water, _W16, [_M["O16"], _M["H"], _M["D"]]), _species(_h2, hh, hd), _TEMPS)
    exp = 1000 * (_independent_ln_beta(_water, _W16, [_M["O16"], _M["H"], _M["D"]], _TEMPS)
                  - _independent_ln_beta(_h2, hh, hd, _TEMPS))
    _close(out["thousand_ln_alpha"], exp, 1e-6, 1e-5, "1000 ln alpha(H2O-H2), D/H")


def test_invalid_inputs_raise_value_error():
    a = _species(_co2, _C16, _C18)
    b = _species(_water, _W16, [_M["O16"], _M["H"], _M["D"]])
    with _raises(ValueError):
        _frac(a, b, _TEMPS)                       # 18O/16O vs D/H
    with _raises(ValueError):
        _frac(a, _species(_water, _W16, _W18), [300.0, 400.0])
    with _raises(ValueError):
        _frac(a, _species(_water, _W16, _W18), [300.0, 300.0, 400.0])


if __name__ == "__main__":
    sys.exit(1 if _run_all(dict(globals())) else 0)
