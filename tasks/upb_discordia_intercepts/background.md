# Scientific background

## 1. U-Pb dating and discordance

Uranium has two long-lived isotopes that decay, through separate chains, to stable lead:

    238U -> 206Pb   (lambda_238 = 1.55125e-10 / yr)
    235U -> 207Pb   (lambda_235 = 9.8485e-10 / yr)          (Jaffey et al. 1971)

For a closed system containing no initial Pb, the radiogenic ratios after time t are

    206Pb*/238U = exp(lambda_238 t) - 1,      207Pb*/235U = exp(lambda_235 t) - 1.

The locus of these points as t varies is the **Wetherill concordia**, a curve in the plane
x = 207Pb/235U, y = 206Pb/238U. Because lambda_238 < lambda_235, y = (1 + x)^k - 1 with
k = lambda_238/lambda_235 ~ 0.1575, so the curve is strictly **concave**. A straight line can
therefore meet it at most twice, and a line of non-positive slope meets it at most once.

Zircon is the workhorse mineral for U-Pb geochronology. Old zircon grains often lose Pb
during a later thermal or fluid event (or by low-temperature leaching related to radiation
damage). A grain that crystallised at t1 and lost a fraction of its accumulated Pb at t2
plots on the chord between the concordia points C(t1) and C(t2). A suite of grains with
different Pb loss therefore defines a straight **discordia**. Its **upper intercept** with
concordia estimates the crystallisation age t1, and its **lower intercept** estimates the
Pb-loss age t2. Recent Pb loss gives a lower intercept near zero. With analytical scatter
the fitted lower intercept can be slightly negative, and is then reported as such.

## 2. Tera-Wasserburg data and conversion to Wetherill coordinates

In-situ techniques (LA-ICP-MS, SIMS) usually report X = 238U/206Pb and Y = 207Pb/206Pb
(the Tera-Wasserburg diagram), with 1-sigma uncertainties sX, sY and an error correlation
rho. The Wetherill coordinates follow from the present-day isotopic ratio of uranium,
U = 238U/235U = 137.818 (Hiess et al. 2012):

    y = 206Pb/238U = 1 / X
    x = 207Pb/235U = (207Pb/206Pb)(206Pb/238U)(238U/235U) = U * Y / X

The errors must be transformed with the full covariance. To first order,
Cov(x, y) = J Cov(X, Y) J^T with J the Jacobian of (x, y) with respect to (X, Y), and
Cov(X, Y) = [[sX^2, rho sX sY], [rho sX sY, sY^2]]. An equivalent route works in relative
errors, because ln x and ln y are linear in ln X and ln Y. Because both Wetherill ratios
depend on X, their errors are strongly correlated even when rho = 0. Ignoring that
correlation biases both the regression and its uncertainties.

## 3. York regression with correlated errors

Each point carries its own errors in x and y and a correlation between them. The best
straight line y = a + b x in the least-squares and maximum-likelihood sense minimises

    S = sum_i  W_i (y_i - a - b x_i)^2,
    W_i = 1 / (sy_i^2 + b^2 sx_i^2 - 2 b rho_i sx_i sy_i),

where the weights themselves depend on b. For fixed b, the optimal intercept is
a = Ybar - b Xbar, with Xbar and Ybar the W-weighted means. York (1966, 1969) and
York et al. (2004) give an implicit equation for b that is solved by fixed-point
iteration. In the 2004 notation:

    U_i = x_i - Xbar,  V_i = y_i - Ybar,  alpha_i = 1/(sx_i sy_i)
    beta_i = W_i [ U_i sy_i^2 + b V_i sx_i^2 - (b U_i + V_i) rho_i sx_i sy_i ]
    b = sum W_i beta_i V_i / sum W_i beta_i U_i

The **least-squares-adjusted points** are the points on the fitted line closest to each
datum in the metric of its error ellipse. Their abscissae are x_adj_i = Xbar + beta_i;
equivalently, each one is the Mahalanobis projection of (x_i, y_i) onto the line.
York et al. (2004) showed that the least-squares and maximum-likelihood standard errors
coincide when evaluated at these adjusted points:

    var_b = 1 / sum W_i (x_adj_i - x_adj_bar)^2
    var_a = 1 / sum W_i + x_adj_bar^2 var_b
    cov(a, b) = - x_adj_bar var_b

Here x_adj_bar is the W-weighted mean of the adjusted abscissae. Evaluating these
expressions at the observed points, as older implementations did, gives different and
incorrect values.

The goodness of fit is MSWD = S/(n - 2), with two parameters fitted. MSWD ~ 1 indicates
scatter consistent with the stated errors. MSWD > 1 indicates excess scatter
(over-dispersion, e.g. geological variability). A common convention (Ludwig's Isoplot,
"model 1") multiplies the parameter and age uncertainties by sqrt(MSWD) when MSWD > 1, and
leaves them unchanged when MSWD <= 1. This task uses that convention.

The York line is symmetric in the two coordinates: refitting with x and y exchanged gives
slope 1/b, and the same MSWD.

## 4. Intercepts and their uncertainties

An intercept age t satisfies

    f(t; a, b) = exp(lambda_238 t) - 1 - a - b (exp(lambda_235 t) - 1) = 0.

For b > 0, df/dt = lambda_238 e^{lambda_238 t} - b lambda_235 e^{lambda_235 t} vanishes at a
single t*, where f has its maximum (f is concave). Every root therefore lies in one of the
two monotonic pieces on either side of t*. For b <= 0, f is increasing and has at most one
root. Reliable root finding has to search both pieces. A single Newton iteration from one
starting guess generally finds only one intercept.

By the implicit function theorem,

    dt/da = 1 / f_t,     dt/db = (exp(lambda_235 t) - 1) / f_t,    f_t = df/dt,

and to first order sigma_t^2 = g^T C g, where g = (dt/da, dt/db) and C is the covariance
of (a, b). Because a and b are strongly anti-correlated whenever the data centroid lies at
positive x, the covariance term is essential. The lower and upper intercepts respond to
it differently.

## 5. Constants

| Quantity | Value | Source |
|---|---|---|
| lambda_238 | 1.55125e-10 yr^-1 | Jaffey et al. (1971) |
| lambda_235 | 9.8485e-10 yr^-1 | Jaffey et al. (1971) |
| 238U/235U | 137.818 | Hiess et al. (2012) |
| 1 Ma | 1e6 yr | |

Decay-constant uncertainties are not propagated in this task.
