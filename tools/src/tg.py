_T_TAG = "general"
_T_DEFAULT = "solution.py"
#BODY#


def test_end_to_end_co2_water_oxygen_isotopes():
    temps = np.array([273.15, 298.15, 323.15, 373.15, 473.15, 673.15, 1073.15])
    out = _fn("isotope_fractionation")(_species(_co2, _C16, _C18), _species(_water, _W16, _W18), temps)
    for key in ("thousand_ln_beta_a", "thousand_ln_beta_b", "thousand_ln_alpha"):
        assert np.asarray(out[key]).shape == temps.shape, key
    assert np.asarray(out["fit"]).shape == (3,)
    exp = 1000 * (_independent_ln_beta(_co2, _C16, _C18, temps) - _independent_ln_beta(_water, _W16, _W18, temps))
    _close(out["thousand_ln_alpha"], exp, 1e-6, 1e-6, "1000 ln alpha(CO2-H2O)")
    alpha = np.asarray(out["thousand_ln_alpha"])
    assert np.all(np.diff(alpha) < 0.0), "fractionation must decrease with temperature here"
    assert 45.5 < alpha[1] < 53.5, alpha[1]


def test_pipeline_is_frame_invariant():
    temps = np.array([300.0, 500.0, 900.0])
    a = _species(_co2, _C16, _C18)
    b = _species(_water, _W16, _W18)
    ref = _fn("isotope_fractionation")(a, b, temps)
    a2, b2 = dict(a), dict(b)
    a2["coords"], a2["hessian"] = _rotate(a["coords"], a["hessian"], seed=1, shift=(5.0, 0.0, -2.0))
    b2["coords"], b2["hessian"] = _rotate(b["coords"], b["hessian"], seed=2, shift=(-1.0, 4.0, 0.5))
    out = _fn("isotope_fractionation")(a2, b2, temps)
    _close(out["thousand_ln_alpha"], ref["thousand_ln_alpha"], 1e-8, 1e-8, "rotated inputs")


def test_fit_recovers_high_temperature_limit():
    temps = np.linspace(2.0e4, 1.0e5, 9)
    out = _fn("isotope_fractionation")(_species(_water, _W16, [_M["O16"], _M["H"], _M["D"]]),
                                       _species(_h2, [_M["H"], _M["H"]], [_M["H"], _M["D"]]), temps)
    A = float(np.asarray(out["fit"])[0])
    a_ht = float(out["a_high_T"])
    exp = 1000 * (_ffc(_water, _W16, [_M["O16"], _M["H"], _M["D"]])
                  - _ffc(_h2, [_M["H"], _M["H"]], [_M["H"], _M["D"]])) / 1e6
    _close(a_ht, exp, 1e-6, 0.0, "a_high_T")
    _close(A, a_ht, 5e-3, 0.0, "fitted A vs analytic high-T limit")


def test_hydrogen_isotopes_magnitude_at_25C():
    # Harmonic alpha(H2O(g)-H2(g)) at 25 C; experiment-derived ~3.5 (3.81 / 1.079),
    # harmonic models overestimate D/H fractionation by several percent.
    out = _fn("isotope_fractionation")(_species(_water, _W16, [_M["O16"], _M["H"], _M["D"]]),
                                       _species(_h2, [_M["H"], _M["H"]], [_M["H"], _M["D"]]),
                                       [273.15, 298.15, 323.15])
    alpha = np.exp(np.asarray(out["thousand_ln_alpha"])[1] / 1000.0)
    assert 3.2 < alpha < 4.3, alpha


if __name__ == "__main__":
    sys.exit(1 if _run_all(dict(globals())) else 0)
