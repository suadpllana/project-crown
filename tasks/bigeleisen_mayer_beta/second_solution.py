"""Independent second reference solution.

Differences from solution.py:
  * Step 1 builds the translation/rotation vectors in the principal-axis frame and
    completes an orthonormal internal basis by modified Gram-Schmidt (the scheme
    used by common quantum-chemistry codes), instead of an SVD/projector
    eigendecomposition; linearity is detected from the rank of the rotation vectors.
  * Step 2 evaluates f = prod (nu*/nu) sinh(u/2)/sinh(u*/2) with a stable log-sinh.
  * Step 3 does not use the Bigeleisen-Mayer product at all: it forms the ratio of
    complete translational x rotational x vibrational partition functions and the
    classical atomic-mass factor (Urey's formulation); the Teller-Redlich product
    rule makes this identical to the Bigeleisen-Mayer result.
  * Step 4 uses the force-constant (Cartesian block trace) form of the
    Bigeleisen-Mayer high-temperature limit instead of sums of squared frequencies.
  * Step 5 fits with numpy.polyfit in x = 1000/T.
"""
import numpy as np

H = 6.62607015e-34
C = 299792458.0
KB = 1.380649e-23
AMU = 1.66053906660e-27
EH = 4.3597447222071e-18
A0 = 5.29177210903e-11
HBAR = H / (2.0 * np.pi)


def _gram_schmidt(vectors, tol):
    basis = []
    for v in vectors:
        w = np.array(v, dtype=float)
        for _ in range(2):
            for b in basis:
                w = w - (b @ w) * b
        nrm = np.linalg.norm(w)
        if nrm > tol:
            basis.append(w / nrm)
    return basis


def _check(coords, masses, hessian):
    x = np.asarray(coords, dtype=float)
    m = np.asarray(masses, dtype=float)
    h = np.asarray(hessian, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or x.shape[0] < 2:
        raise ValueError("coords must be (N, 3) with N >= 2")
    n = x.shape[0]
    if m.shape != (n,) or h.shape != (3 * n, 3 * n):
        raise ValueError("inconsistent shapes")
    if np.any(~np.isfinite(m)) or np.any(m <= 0) or np.any(~np.isfinite(x)) or np.any(~np.isfinite(h)):
        raise ValueError("invalid masses or non-finite input")
    return x, m, h


def vibrational_wavenumbers(coords, masses, hessian):
    x, m, h = _check(coords, masses, hessian)
    n = len(m)
    r = x - (m @ x) / m.sum()
    tensor = np.einsum("i,ij,ik->jk", m, r, r)
    tensor = np.trace(tensor) * np.eye(3) - tensor
    _, axes = np.linalg.eigh(tensor)
    p = r @ axes                                        # principal-axis coordinates
    sm = np.sqrt(m)
    vecs = []
    for a in range(3):                                  # translations
        v = np.zeros((n, 3))
        v[:, a] = sm
        vecs.append(v.ravel())
    for a in range(3):                                  # rotations about principal axes
        e = np.zeros(3)
        e[a] = 1.0
        v = (sm[:, None] * np.cross(e, p)) @ axes.T     # back to the input frame
        vecs.append(v.ravel())
    scale = np.sqrt(m.sum() * max(1.0, np.max(np.sum(r ** 2, axis=1))))
    ext = _gram_schmidt(vecs, 1e-6 * scale)
    full = _gram_schmidt(ext + list(np.eye(3 * n)), 1e-8)
    internal = np.array(full[len(ext):]).T
    if internal.shape[1] != 3 * n - len(ext):
        raise ValueError("basis construction failed")
    hmw = h / np.sqrt(np.outer(np.repeat(m, 3), np.repeat(m, 3)))
    hmw = 0.5 * (hmw + hmw.T)
    lam = np.linalg.eigvalsh(internal.T @ hmw @ internal) * EH / (A0 ** 2 * AMU)
    nu = np.sign(lam) * np.sqrt(np.abs(lam)) / (2.0 * np.pi * C * 100.0)
    return np.sort(nu)


def _log_sinh(y):
    return y + np.log1p(-np.exp(-2.0 * y)) - np.log(2.0)


def _temps(T):
    t = np.asarray(T, dtype=float)
    if np.any(~np.isfinite(t)) or np.any(t <= 0):
        raise ValueError("temperatures must be positive")
    return t


def log_reduced_partition_function_ratio(nu_light, nu_heavy, T):
    a = np.asarray(nu_light, dtype=float)
    b = np.asarray(nu_heavy, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError("frequency arrays must be 1-D with equal length")
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)) or np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("frequencies must be real and positive")
    t = _temps(T)
    c2 = H * C * 100.0 / KB
    total = np.zeros(t.shape)
    for nl, nh in zip(a, b):
        total = total + np.log(nh / nl) + _log_sinh(c2 * nl / (2 * t)) - _log_sinh(c2 * nh / (2 * t))
    return float(total) if t.ndim == 0 else total


def _sites(ml, mh):
    ml = np.asarray(ml, dtype=float)
    mh = np.asarray(mh, dtype=float)
    if ml.ndim != 1 or ml.shape != mh.shape:
        raise ValueError("mass vectors must have equal length")
    idx = [i for i in range(len(ml)) if abs(ml[i] - mh[i]) > 1e-12 * abs(ml[i])]
    if not idx:
        raise ValueError("no substitution")
    if len({round(ml[i], 9) for i in idx}) != 1 or len({round(mh[i], 9) for i in idx}) != 1:
        raise ValueError("mixed substitutions")
    return ml, mh, idx


def _moments(x, m):
    r = x - (m @ x) / m.sum()
    t = np.einsum("i,ij,ik->jk", m, r, r)
    return np.linalg.eigvalsh(np.trace(t) * np.eye(3) - t)


def log_beta(coords, masses_light, masses_heavy, hessian, T):
    ml, mh, idx = _sites(masses_light, masses_heavy)
    x = np.asarray(coords, dtype=float)
    nl = vibrational_wavenumbers(x, ml, hessian)
    nh = vibrational_wavenumbers(x, mh, hessian)
    if np.any(nl <= 0) or np.any(nh <= 0):
        raise ValueError("imaginary frequencies")
    t = _temps(T)
    c2 = H * C * 100.0 / KB
    il, ih = _moments(x, ml), _moments(x, mh)
    if len(nl) == 3 * len(ml) - 5:
        ln_rot = np.log(ih[-1] / il[-1])
    else:
        ln_rot = 0.5 * np.sum(np.log(ih / il))
    ln_trans = 1.5 * np.log(mh.sum() / ml.sum())
    tt = np.atleast_1d(t)[..., None]
    ul, uh = c2 * nl / tt, c2 * nh / tt
    ln_vib = np.sum(-uh / 2 - np.log1p(-np.exp(-uh)) + ul / 2 + np.log1p(-np.exp(-ul)), axis=-1)
    ln_f = ln_trans + ln_rot + ln_vib - 1.5 * np.sum(np.log(mh / ml))
    out = ln_f / len(idx)
    return float(out[0]) if t.ndim == 0 else out.reshape(t.shape)


def high_temperature_coefficient(coords, masses_light, masses_heavy, hessian):
    ml, mh, idx = _sites(masses_light, masses_heavy)
    _check(coords, ml, hessian)
    h = np.asarray(hessian, dtype=float)
    s = 0.0
    for a in idx:
        trace = np.trace(h[3 * a:3 * a + 3, 3 * a:3 * a + 3]) * EH / A0 ** 2
        s += (1.0 / ml[a] - 1.0 / mh[a]) / AMU * trace
    return HBAR ** 2 * s / (24.0 * KB ** 2 * len(idx))


def isotope_fractionation(species_a, species_b, temperatures):
    t = np.asarray(temperatures, dtype=float)
    if t.ndim != 1 or len(set(t.tolist())) < 3 or np.any(t <= 0):
        raise ValueError("need >= 3 distinct positive temperatures")
    pairs = []
    for sp in (species_a, species_b):
        if not all(k in sp for k in ("coords", "masses_light", "masses_heavy", "hessian")):
            raise ValueError("species dict is missing a required key")
        ml, mh, idx = _sites(sp["masses_light"], sp["masses_heavy"])
        pairs.append((ml[idx[0]], mh[idx[0]]))
    if abs(pairs[0][0] - pairs[1][0]) > 1e-9 * pairs[0][0] or abs(pairs[0][1] - pairs[1][1]) > 1e-9 * pairs[0][1]:
        raise ValueError("different isotope systems")
    res = {}
    for key, sp in (("a", species_a), ("b", species_b)):
        res[key] = 1000.0 * log_beta(sp["coords"], sp["masses_light"], sp["masses_heavy"], sp["hessian"], t)
        res["k" + key] = high_temperature_coefficient(sp["coords"], sp["masses_light"],
                                                      sp["masses_heavy"], sp["hessian"])
    alpha = res["a"] - res["b"]
    fit = np.polyfit(1000.0 / t, alpha, 2)
    return {
        "thousand_ln_beta_a": res["a"],
        "thousand_ln_beta_b": res["b"],
        "thousand_ln_alpha": alpha,
        "fit": np.asarray(fit),
        "a_high_T": 1000.0 * (res["ka"] - res["kb"]) / 1.0e6,
    }
