def log_reduced_partition_function_ratio(nu_light, nu_heavy, T):
    """Natural log of the Bigeleisen-Mayer reduced partition function ratio f (heavy/light).

    nu_light, nu_heavy : 1-D arrays (same length) of real harmonic wavenumbers, cm^-1, > 0
    T                  : temperature in K, positive scalar or array

    Returns ln f with the shape of T (a Python/NumPy scalar for scalar T).
    """
    nl = np.asarray(nu_light, dtype=float)
    nh = np.asarray(nu_heavy, dtype=float)
    if nl.ndim != 1 or nh.ndim != 1 or nl.shape != nh.shape or nl.size == 0:
        raise ValueError("nu_light and nu_heavy must be non-empty 1-D arrays of equal length")
    if not (np.all(np.isfinite(nl)) and np.all(np.isfinite(nh))):
        raise ValueError("wavenumbers must be finite")
    if np.any(nl <= 0.0) or np.any(nh <= 0.0):
        raise ValueError("wavenumbers must be real and positive")
    t = np.asarray(T, dtype=float)
    if not np.all(np.isfinite(t)) or np.any(t <= 0.0):
        raise ValueError("temperatures must be positive and finite")

    t_flat = np.atleast_1d(t).ravel()
    u_l = C2_CM_K * nl[:, None] / t_flat[None, :]
    u_h = C2_CM_K * nh[:, None] / t_flat[None, :]

    def phi(u):
        # ln[u exp(-u/2) / (1 - exp(-u))], evaluated without overflow or cancellation.
        return np.log(u) - 0.5 * u - np.log(-np.expm1(-u))

    out = np.sum(phi(u_h) - phi(u_l), axis=0)
    if t.ndim == 0:
        return float(out[0])
    return out.reshape(t.shape)
