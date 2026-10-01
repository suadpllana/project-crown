def york_fit(x, sx, y, sy, rho, rtol=1e-15, max_iter=1000):
    """Best straight line y = a + b x for data with correlated errors in x and y
    (York et al. 2004). Returns a dict with intercept, slope, cov (2x2, order
    [intercept, slope]), mswd and n."""
    x, sx, y, sy, rho = (np.asarray(v, dtype=float) for v in (x, sx, y, sy, rho))
    n = x.shape[0] if x.ndim == 1 else 0
    if n < 3 or any(v.shape != (n,) for v in (sx, y, sy, rho)):
        raise ValueError("need at least 3 points given as 1-D arrays of equal length")
    if not all(np.all(np.isfinite(v)) for v in (x, sx, y, sy, rho)):
        raise ValueError("inputs must be finite")
    if np.any(sx <= 0) or np.any(sy <= 0) or np.any(np.abs(rho) >= 1):
        raise ValueError("errors must be positive and |rho| < 1")
    if np.ptp(x) == 0:
        raise ValueError("x values must not all be equal")

    wx, wy = 1.0 / sx ** 2, 1.0 / sy ** 2
    alpha = np.sqrt(wx * wy)

    def state(b):
        w = wx * wy / (wx + b * b * wy - 2.0 * b * rho * alpha)
        xbar, ybar = w @ x / w.sum(), w @ y / w.sum()
        u, v = x - xbar, y - ybar
        beta = w * (u / wy + b * v / wx - (b * u + v) * rho / alpha)
        return w, xbar, ybar, u, v, beta

    b = np.polyfit(x, y, 1)[0]
    for _ in range(max_iter):
        w, xbar, ybar, u, v, beta = state(b)
        b_new = (w * beta) @ v / ((w * beta) @ u)
        converged = abs(b_new - b) <= rtol * abs(b_new) or b_new == b
        b = b_new
        if converged:
            break
    else:
        raise RuntimeError("York iteration did not converge")

    w, xbar, ybar, u, v, beta = state(b)
    a = ybar - b * xbar
    x_adj = xbar + beta
    x_adj_bar = w @ x_adj / w.sum()
    var_b = 1.0 / (w @ (x_adj - x_adj_bar) ** 2)
    var_a = 1.0 / w.sum() + x_adj_bar ** 2 * var_b
    cov_ab = -x_adj_bar * var_b
    mswd = w @ (y - a - b * x) ** 2 / (n - 2)
    return {
        "intercept": float(a),
        "slope": float(b),
        "cov": np.array([[var_a, cov_ab], [cov_ab, var_b]]),
        "mswd": float(mswd),
        "n": int(n),
    }
