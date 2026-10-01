

def _species(builder, ml, mh):
    x, hess, _ = builder(ml)
    return {"coords": x, "masses_light": np.array(ml), "masses_heavy": np.array(mh), "hessian": hess}


def _independent_ln_beta(builder, ml, mh, T):
    ml, mh = np.asarray(ml, float), np.asarray(mh, float)
    n = int(np.sum(ml != mh))
    x, _, nl = builder(ml)
    _, _, nh = builder(mh)
    return _lnf_full_partition_functions(x, ml, mh, nl, nh, T) / n


def _ffc(builder, ml, mh):
    """High-T coefficient from force-constant blocks (independent of frequencies)."""
    ml, mh = np.asarray(ml, float), np.asarray(mh, float)
    _, hess, _ = builder(ml)
    sites = np.flatnonzero(ml != mh)
    hbar = _T_H / (2 * np.pi)
    tot = sum((1 / ml[a] - 1 / mh[a]) / _T_AMU * np.trace(hess[3 * a:3 * a + 3, 3 * a:3 * a + 3])
              * _T_EH / _T_A0 ** 2 for a in sites)
    return hbar ** 2 * tot / (24 * _T_KB ** 2 * sites.size)


def _h2(m):
    k = _T_H2_K
    x, hess = _diatomic(k, 0.74144)
    return x, hess, np.array([_diatomic_nu(k, *m)])


# H2 harmonic force constant from omega_e = 4401.21 cm^-1 (Huber & Herzberg).
_T_H2_K = float(_lambda_from_nu(4401.21) * _M["H"] / 2.0)

_W16 = [_M["O16"], _M["H"], _M["H"]]
_W18 = [_M["O18"], _M["H"], _M["H"]]
_C16 = [_M["O16"], _M["C12"], _M["O16"]]
_C18 = [_M["O16"], _M["C12"], _M["O18"]]
