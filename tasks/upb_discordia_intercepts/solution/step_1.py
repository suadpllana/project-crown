"""Reference solution for Step 1."""
import numpy as np

# Decay constants (Jaffey et al. 1971), per year, and present-day 238U/235U (Hiess et al. 2012).
LAMBDA_238 = 1.55125e-10
LAMBDA_235 = 9.8485e-10
U238_U235 = 137.818
YEARS_PER_MA = 1.0e6
# Age window (Ma) in which concordia intercepts are sought.
T_MIN_MA = -1000.0
T_MAX_MA = 5000.0


# ---- Step 1 ----
def tera_wasserburg_to_wetherill(X, sX, Y, sY, rho):
    """Convert Tera-Wasserburg ratios to Wetherill ratios with linear error propagation.

    X = 238U/206Pb, Y = 207Pb/206Pb (1-sigma absolute errors sX, sY, correlation rho).
    Returns an (n, 5) array with columns [x, sx, y, sy, rho_xy] where
    x = 207Pb/235U and y = 206Pb/238U.
    """
    X, sX, Y, sY, rho = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (X, sX, Y, sY, rho))
    n = X.shape[0]
    if X.ndim != 1 or n == 0 or any(v.shape != (n,) for v in (sX, Y, sY, rho)):
        raise ValueError("inputs must be 1-D arrays of equal, non-zero length")
    if not all(np.all(np.isfinite(v)) for v in (X, sX, Y, sY, rho)):
        raise ValueError("inputs must be finite")
    if np.any(X <= 0) or np.any(Y <= 0) or np.any(sX <= 0) or np.any(sY <= 0):
        raise ValueError("ratios and their errors must be positive")
    if np.any(np.abs(rho) >= 1):
        raise ValueError("correlation coefficients must satisfy |rho| < 1")

    x = U238_U235 * Y / X
    y = 1.0 / X
    # Jacobian of (x, y) with respect to (X, Y).
    dx_dX = -U238_U235 * Y / X ** 2
    dx_dY = U238_U235 / X
    dy_dX = -1.0 / X ** 2
    cXX, cYY, cXY = sX ** 2, sY ** 2, rho * sX * sY
    vx = dx_dX ** 2 * cXX + dx_dY ** 2 * cYY + 2.0 * dx_dX * dx_dY * cXY
    vy = dy_dX ** 2 * cXX
    cxy = dx_dX * dy_dX * cXX + dx_dY * dy_dX * cXY
    sx, sy = np.sqrt(vx), np.sqrt(vy)
    return np.column_stack([x, sx, y, sy, cxy / (sx * sy)])
