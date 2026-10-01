"""Tests for Step 4: high-temperature coefficient of ln(beta).

Expected values are computed independently (Wilson GF method, analytic
results, complete partition functions); the implementation under test is never
used to produce a target.
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Locating the implementation under test.
# 1. If the functions are already defined in this module's globals (harness that
#    concatenates candidate code and tests), they are used directly.
# 2. Otherwise the file named by the environment variable CROWN_IMPL is loaded.
# 3. Otherwise the default file for this test module is loaded (see _T_DEFAULT).
# ---------------------------------------------------------------------------
_T_MODULE = None


def _t_load():
    global _T_MODULE
    if _T_MODULE is None:
        path = os.environ.get("CROWN_IMPL")
        if path is None:
            path = str(Path(__file__).resolve().parents[1] / _T_DEFAULT)
        spec = importlib.util.spec_from_file_location("_crown_impl_" + _T_TAG, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _T_MODULE = mod
    return _T_MODULE


def _fn(name):
    g = globals()
    if name in g and callable(g[name]):
        return g[name]
    return getattr(_t_load(), name)


# ---------------------------------------------------------------------------
# Independent physics used to build inputs and expected values. None of this
# calls the implementation under test.
# ---------------------------------------------------------------------------
_T_H = 6.62607015e-34
_T_C = 299792458.0
_T_KB = 1.380649e-23
_T_AMU = 1.66053906660e-27
_T_EH = 4.3597447222071e-18
_T_A0 = 5.29177210903e-11
_T_ANG_TO_BOHR = 1e-10 / _T_A0
_T_LAM_TO_NU = 1.0 / (2.0 * np.pi * _T_C * 100.0)  # omega (rad/s) -> cm^-1
_T_LAM_SI = _T_EH / (_T_A0 ** 2 * _T_AMU)          # Eh/(bohr^2 amu) -> s^-2
_T_C2 = _T_H * _T_C * 100.0 / _T_KB                 # cm K

# Atomic masses (AME2020), amu.
_M = {
    "H": 1.00782503223, "D": 2.01410177812,
    "C12": 12.0, "C13": 13.00335483507,
    "O16": 15.99491461957, "O18": 17.99915961286,
    "N14": 14.00307400443, "S32": 31.9720711744, "Cl35": 34.968852682,
}

# Harmonic valence force fields (atomic units: Eh/bohr^2, Eh/rad^2, Eh/(bohr rad)).
# Water: fitted to the experimental harmonic wavenumbers of H2O (1648.5, 3832.2,
# 3942.5 cm^-1) and the symmetric-stretch harmonic wavenumber of D2O (2763.8 cm^-1).
_T_WATER_F = (0.54306809, -0.00644291, 0.15994866, 0.02727912)  # f_r, f_rr', f_theta, f_r_theta
_T_WATER_GEOM = (0.9572, 104.52)                                 # r_e / Angstrom, theta_e / deg
# CO2: fitted to harmonic wavenumbers 1354.0, 672.9 (doubly degenerate), 2396.3 cm^-1.
_T_CO2_F = (1.0289399, 0.08077316, 0.17963341)                   # f_r, f_rr', f_delta
_T_CO2_R = 1.1600                                                # r_e / Angstrom


def _nu_from_lambda(lam):
    lam = np.asarray(lam, dtype=float)
    return np.sign(lam) * np.sqrt(np.abs(lam) * _T_LAM_SI) * _T_LAM_TO_NU


def _lambda_from_nu(nu):
    nu = np.asarray(nu, dtype=float)
    return np.sign(nu) * (nu / _T_LAM_TO_NU) ** 2 / _T_LAM_SI


def _water_model():
    """Returns coords (Angstrom), Wilson B matrix (bohr-based) and internal F (a.u.)."""
    r_ang, th_deg = _T_WATER_GEOM
    th = np.deg2rad(th_deg)
    r = r_ang * _T_ANG_TO_BOHR
    o = np.zeros(3)
    h1 = r * np.array([np.sin(th / 2), 0.0, np.cos(th / 2)])
    h2 = r * np.array([-np.sin(th / 2), 0.0, np.cos(th / 2)])
    x = np.array([o, h1, h2])
    e1, e2 = h1 / r, h2 / r
    b = np.zeros((3, 9))
    b[0, 3:6], b[0, 0:3] = e1, -e1
    b[1, 6:9], b[1, 0:3] = e2, -e2
    s1 = (np.cos(th) * e1 - e2) / (r * np.sin(th))
    s2 = (np.cos(th) * e2 - e1) / (r * np.sin(th))
    b[2, 3:6], b[2, 6:9], b[2, 0:3] = s1, s2, -s1 - s2
    fr, frr, fth, frth = _T_WATER_F
    f = np.array([[fr, frr, frth], [frr, fr, frth], [frth, frth, fth]])
    return x / _T_ANG_TO_BOHR, b, f


def _co2_model():
    r = _T_CO2_R * _T_ANG_TO_BOHR
    x = np.array([[0.0, 0.0, -r], [0.0, 0.0, 0.0], [0.0, 0.0, r]])  # O, C, O
    b = np.zeros((4, 9))
    b[0, 2], b[0, 5] = -1.0, 1.0
    b[1, 5], b[1, 8] = -1.0, 1.0
    for k, ax in enumerate((0, 1)):
        b[2 + k, ax], b[2 + k, 3 + ax], b[2 + k, 6 + ax] = 1.0 / r, -2.0 / r, 1.0 / r
    fr, frr, fd = _T_CO2_F
    f = np.array([[fr, frr, 0, 0], [frr, fr, 0, 0], [0, 0, fd, 0], [0, 0, 0, fd]])
    return x / _T_ANG_TO_BOHR, b, f


def _cart_hessian(b, f):
    return b.T @ f @ b


def _gf_nu(b, f, masses):
    """Wilson GF wavenumbers (cm^-1, ascending) - no Cartesian projection involved."""
    g = b @ np.diag(1.0 / np.repeat(np.asarray(masses, float), 3)) @ b.T
    lam = np.linalg.eigvals(g @ f).real
    return np.sort(_nu_from_lambda(lam))


def _water(masses):
    x, b, f = _water_model()
    return x, _cart_hessian(b, f), _gf_nu(b, f, masses)


def _co2(masses):
    x, b, f = _co2_model()
    return x, _cart_hessian(b, f), _gf_nu(b, f, masses)


def _diatomic(k_au, r_ang, direction=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 0.0)):
    u = np.asarray(direction, float)
    u = u / np.linalg.norm(u)
    x = np.array([np.asarray(origin, float), np.asarray(origin, float) + r_ang * u])
    blk = k_au * np.outer(u, u)
    hess = np.block([[blk, -blk], [-blk, blk]])
    return x, hess


def _diatomic_nu(k_au, m1, m2):
    mu = m1 * m2 / (m1 + m2)
    return float(_nu_from_lambda(k_au / mu))


def _principal_moments(x, masses):
    m = np.asarray(masses, float)
    r = x - (m @ x) / m.sum()
    i = np.einsum("i,ij,ik->jk", m, r, r)
    return np.linalg.eigvalsh(np.trace(i) * np.eye(3) - i)


def _lnf_full_partition_functions(x, ml, mh, nu_l, nu_h, T):
    """ln f from complete (trans x rot x vib) partition-function ratios and the
    classical atomic-mass factor, without symmetry numbers:
        f = (Q*/Q) * prod_atoms (m/m*)^(3/2)
    Equal to the Bigeleisen-Mayer product when the Teller-Redlich rule holds."""
    ml, mh = np.asarray(ml, float), np.asarray(mh, float)
    T = np.atleast_1d(np.asarray(T, float))[None, :]
    trans = 1.5 * np.log(mh.sum() / ml.sum())
    il, ih = _principal_moments(x, ml), _principal_moments(x, mh)
    if il[0] < 1e-6 * il[-1]:
        rot = np.log(ih[-1] / il[-1])
    else:
        rot = 0.5 * np.sum(np.log(ih / il))
    ul = _T_C2 * np.asarray(nu_l)[:, None] / T
    uh = _T_C2 * np.asarray(nu_h)[:, None] / T
    vib = np.sum((ul - uh) / 2.0 - np.log(-np.expm1(-uh)) + np.log(-np.expm1(-ul)), axis=0)
    atoms = -1.5 * np.sum(np.log(mh / ml))
    return trans + rot + vib + atoms


def _lnf_sinh(nu_l, nu_h, T):
    """Closed form f = prod (nu*/nu) sinh(u/2)/sinh(u*/2); valid for moderate u."""
    T = np.atleast_1d(np.asarray(T, float))[None, :]
    ul = _T_C2 * np.asarray(nu_l, float)[:, None] / T
    uh = _T_C2 * np.asarray(nu_h, float)[:, None] / T
    return np.sum(np.log(uh / ul) + np.log(np.sinh(ul / 2)) - np.log(np.sinh(uh / 2)), axis=0)


def _external_basis(x, masses):
    m = np.asarray(masses, float)
    n = len(m)
    r = x - (m @ x) / m.sum()
    w = np.sqrt(m)[:, None]
    vecs = [(w * np.tile(a, (n, 1))).ravel() for a in np.eye(3)]
    vecs += [(w * np.cross(a, r)).ravel() for a in np.eye(3)]
    q, _ = np.linalg.qr(np.array(vecs).T)
    return q


def _designed_hessian(seed, x, masses, target_nu, contamination_nu=0.0):
    """Cartesian Hessian (Eh/bohr^2) of a non-linear molecule whose vibrational
    spectrum is exactly target_nu (cm^-1). Optionally adds a symmetric perturbation
    E with P E P = 0 (P = projector on the vibrational space): it couples the
    vibrational space to translations/rotations, mimicking finite-difference noise,
    and must be removed by the projection."""
    rng = np.random.default_rng(seed)
    m = np.asarray(masses, float)
    n3 = 3 * len(m)
    ext = _external_basis(x, m)
    full, _ = np.linalg.qr(np.column_stack([ext, rng.standard_normal((n3, n3 - 6))]))
    q_int = full[:, 6:]
    u, _ = np.linalg.qr(rng.standard_normal((n3 - 6, n3 - 6)))
    lam = _lambda_from_nu(target_nu)
    hmw = q_int @ u @ np.diag(lam) @ u.T @ q_int.T
    if contamination_nu:
        a = rng.standard_normal((n3, n3))
        a = 0.5 * (a + a.T)
        a *= _lambda_from_nu(contamination_nu) / np.linalg.norm(a, 2)
        pe = ext @ ext.T
        hmw = hmw + pe @ a + a @ pe - pe @ a @ pe
    s = np.sqrt(np.repeat(m, 3))
    return hmw * np.outer(s, s)


def _rotate(x, hess, seed, shift=(0.0, 0.0, 0.0)):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    n = x.shape[0]
    big = np.kron(np.eye(n), q)
    return x @ q.T + np.asarray(shift), big @ hess @ big.T


def _close(actual, expected, rtol, atol, what):
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    assert a.shape == e.shape, "%s: shape %s, expected %s" % (what, a.shape, e.shape)
    assert np.all(np.isfinite(a)), "%s: non-finite values %r" % (what, a)
    ok = np.abs(a - e) <= atol + rtol * np.abs(e)
    assert np.all(ok), "%s: max abs error %.3e" % (what, np.max(np.abs(a - e)))


class _raises:
    def __init__(self, *exc):
        self.exc = exc or (Exception,)

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError("expected %s to be raised" % (self.exc,))
        return issubclass(et, self.exc)


def _run_all(namespace):
    failed = 0
    for name, fn in sorted(namespace.items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL", name, "-", type(exc).__name__, exc)
    print("%d failed" % failed)
    return failed


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
