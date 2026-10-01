_T_TAG = "general"
_T_DEFAULT = "solution.py"


def _run(*args):
    return _fn("discordia_ages")(*args)


def test_archean_zircon_with_palaeozoic_pb_loss():
    data = _synthetic_tw(101, 16, 2700.0, 450.0, rel_sX=0.015, rel_sY=0.010, rho_tw=0.15)
    out = _run(*data)
    ref = _independent_pipeline(*data)
    _close(out["ages"], ref["ages"], 0.0, 1e-3, "ages")
    _close(out["sigmas"], ref["sigmas"], 1e-4, 1e-6, "sigmas")
    # Statistical sanity against the generating ages (seeded, deterministic).
    ages, sig = np.asarray(out["ages"]), np.asarray(out["sigmas"])
    assert np.all(np.abs(ages - [450.0, 2700.0]) < 4.0 * sig), (ages, sig)
    assert np.all(sig > 0.5) and np.all(sig < 200.0), sig


def test_recent_pb_loss_gives_near_zero_lower_intercept():
    data = _synthetic_tw(202, 12, 1850.0, 0.0, rel_sX=0.01, rel_sY=0.006, rho_tw=-0.2)
    out = _run(*data)
    ref = _independent_pipeline(*data)
    _close(out["ages"], ref["ages"], 0.0, 1e-3, "ages")
    _close(out["sigmas"], ref["sigmas"], 1e-4, 1e-6, "sigmas")
    assert abs(float(np.asarray(out["ages"])[0])) < 4.0 * float(np.asarray(out["sigmas"])[0]) + 5.0


def test_order_of_analyses_is_irrelevant():
    data = _synthetic_tw(303, 10, 2100.0, 600.0)
    perm = np.random.default_rng(0).permutation(10)
    a = _run(*data)
    b = _run(*(np.asarray(v)[perm] for v in data))
    _close(b["ages"], a["ages"], 0.0, 1e-4, "permuted ages")
    _close(b["sigmas"], a["sigmas"], 1e-6, 1e-9, "permuted sigmas")
    _close(b["mswd"], a["mswd"], 1e-7, 1e-12, "permuted MSWD")


def test_pipeline_consistency_between_steps():
    data = _synthetic_tw(404, 12, 3200.0, 900.0, dispersion=1.8)
    out = _run(*data)
    a, b = out["intercept"], out["slope"]
    ages = np.asarray(out["ages"])
    # Each age lies on both the concordia and the fitted line.
    x, y = _concordia(ages)
    _close(y, a + b * x, 1e-9, 1e-12, "intercepts lie on the discordia")
    raw = _sigma_fd(a, b, np.asarray(out["cov"]), ages)
    scale = np.sqrt(max(float(out["mswd"]), 1.0))
    _close(out["sigmas"], raw * scale, 1e-4, 1e-6, "sigmas from cov and MSWD")


if __name__ == "__main__":
    sys.exit(1 if _run_all(dict(globals())) else 0)
