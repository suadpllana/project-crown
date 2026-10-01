# Scientific background

## 1. Why beta-factors

The distribution of a rare heavy isotope (D, 13C, 18O, ...) between two coexisting
phases or molecules at equilibrium is described by the fractionation factor

    alpha_{A-B} = R_A / R_B,           R = (heavy isotope) / (light isotope)

and is reported in isotope geochemistry as `1000 ln(alpha)` ("per mil"). Measured
values (e.g. for CO2-water, carbonate-water or vapour-liquid water) underpin
palaeothermometry, hydrology and carbon-cycle studies. For gas-phase molecules,
alpha can be predicted from harmonic vibrational frequencies: alpha_{A-B} =
beta_A / beta_B, where beta is the *reduced partition function ratio* of the heavy and
light isotopologues of each species (Urey 1947; Bigeleisen & Mayer 1947).

In practice the frequencies come from a quantum-chemistry calculation of **one**
isotopologue: the code outputs an equilibrium geometry and a Cartesian Hessian. Within
the Born-Oppenheimer approximation the potential energy surface, hence the Cartesian
Hessian, is the same for every isotopologue; only the masses change. The frequencies of
each isotopologue therefore have to be recomputed from that Hessian with its own masses.

## 2. Harmonic frequencies from a Cartesian Hessian

Let `H` be the 3N x 3N Cartesian Hessian (energy second derivatives), ordered
x1, y1, z1, x2, ... The mass-weighted Hessian is

    H_mw[i, j] = H[i, j] / sqrt(m_i m_j),

where `m_i` is the mass of the atom that owns coordinate i. Its eigenvalues
`lambda = omega^2` are the squared angular frequencies of the normal modes.

**Translations and rotations.** Overall translations and infinitesimal rotations of a
molecule do not change its energy. In mass-weighted coordinates they are spanned by

* translations: for axis a in {x, y, z}, the vector with component `sqrt(m_k)` on the
  a-coordinate of every atom k;
* rotations: for axis a, the vector whose block for atom k is `sqrt(m_k) (e_a x r_k)`,
  with `r_k` the position of atom k relative to the centre of mass.

For a non-linear molecule these six vectors are linearly independent; for a linear
molecule (including every diatomic) the rotation about the molecular axis vanishes and
only five remain. The vibrational subspace is the orthogonal complement of this external
space, of dimension 3N-6 (non-linear) or 3N-5 (linear). The vibrational frequencies are
the eigenvalues of `H_mw` restricted to that subspace, i.e. of `Q^T H_mw Q` where the
columns of `Q` form an orthonormal basis of the complement (equivalently, the non-trivial
eigenvalues of `P H_mw P` with `P` the projector onto the complement).

For an exact Hessian at a stationary point the external eigenvalues are exactly zero.
Hessians obtained by finite differences or with loose convergence contain noise that
couples the vibrational and external spaces; projection removes it, whereas merely
discarding the smallest eigenvalues of the unprojected matrix does not (and fails
completely when a genuine soft mode is softer than the noise). Quantum-chemistry
packages project for this reason.

**Linearity.** The principal moments of inertia are the eigenvalues of
`I = sum_k m_k (|r_k|^2 1 - r_k r_k^T)`. A molecule is treated as linear when its
smallest principal moment is below `1e-6` times its largest.

**Units.** With the Hessian in Hartree/bohr^2 and masses in amu, an eigenvalue `lambda`
converts to SI as `lambda * E_h / (a_0^2 * amu)` (s^-2). The angular frequency is
`omega = sqrt(lambda)` and the wavenumber is `nu = omega / (2 pi c)` (report in cm^-1,
i.e. use c in cm/s). A negative eigenvalue (a saddle point direction) is reported as a
negative ("imaginary") wavenumber `-sqrt(|lambda|)/(2 pi c)`.

## 3. The Bigeleisen-Mayer reduced partition function ratio

For an ideal-gas molecule in the harmonic-oscillator / rigid-rotor approximation, the
ratio of total partition functions of the heavy (*) and light isotopologues, divided by
the classical (high-temperature) value of the same ratio, is

    f = (Q*/Q) * prod_atoms (m / m*)^{3/2}         (symmetry numbers omitted).

By the Teller-Redlich product rule, the translational and rotational parts of `Q*/Q`
combine with the atomic mass factor into `prod_i (nu*_i / nu_i)`, leaving only
vibrational quantities:

    f = prod_i  (u*_i / u_i) * [exp(-u*_i/2) / (1 - exp(-u*_i))] / [exp(-u_i/2) / (1 - exp(-u_i))],

    u_i = h c nu_i / (k_B T),   i over the 3N-6 (or 3N-5) vibrations.

`f > 1` when the heavy isotope lowers the frequencies (it always does for a stable
molecule). Equivalent closed form: `f = prod_i (u*_i/u_i) sinh(u_i/2)/sinh(u*_i/2)`.

**Numerical range.** At low temperature (u ~ 10^3) the individual factors
`exp(-u/2)` underflow in double precision, so the product must be evaluated as a sum of
logarithms; `ln(1 - exp(-u))` should be evaluated with a cancellation-free expression.
Limits useful for checking:

* T -> 0:   ln f -> sum_i [ln(nu*_i/nu_i) + (u_i - u*_i)/2]   (zero-point energy dominated);
* T -> inf: ln f = sum_i [(u_i^2 - u*_i^2)/24 - (u_i^4 - u*_i^4)/2880 + ...] -> 0.

## 4. Beta-factor conventions

For a molecule in which n equivalent or non-equivalent atoms of the same element are
substituted by the same heavy isotope, the per-atom beta-factor is

    ln(beta) = ln(f) / n.

Symmetry numbers are not included: they enter both sides of an isotope exchange
equilibrium as classical factors and are conventionally excluded from beta. Only one
element/isotope pair may be substituted at a time.

The fractionation between species A and B is

    1000 ln(alpha_{A-B}) = 1000 [ln(beta_A) - ln(beta_B)],

meaningful only when both betas refer to the same isotope pair (e.g. 18O/16O).
Results are commonly summarised by the empirical form

    1000 ln(alpha) = A (10^6 / T^2) + B (10^3 / T) + C,

obtained by ordinary least squares over the computed temperatures.

## 5. High-temperature limit and the force-constant form

From the expansion in Section 3, `ln(beta) -> K / T^2` with

    K = (h c)^2 / (24 k_B^2 n) * sum_i (nu_i^2 - nu*_i^2)        (nu in cm^-1, c in cm/s).

Because the trace of the mass-weighted Hessian equals the sum of squared angular
frequencies (external eigenvalues are zero for an invariant Hessian),
`sum_i (omega_i^2 - omega*_i^2) = sum_{substituted atoms a} (1/m_a - 1/m*_a) tr(H_aa)`,
where `H_aa` is the 3x3 Cartesian block of atom a. This gives the Bigeleisen-Mayer
"force-constant" form used for heavy-element isotopes:

    K = hbar^2 / (24 k_B^2 n) * sum_a (1/m_a - 1/m*_a) tr(H_aa)    (SI units).

In the units of the fit coefficient A, the high-temperature limit of
`1000 ln(alpha_{A-B})` is `1000 (K_A - K_B) / 10^6`.

## 6. Constants (CODATA 2018)

| quantity | value |
|---|---|
| Planck constant h | 6.62607015e-34 J s |
| speed of light c | 299792458 m/s |
| Boltzmann constant k_B | 1.380649e-23 J/K |
| atomic mass constant (1 amu) | 1.66053906660e-27 kg |
| Hartree energy E_h | 4.3597447222071e-18 J |
| Bohr radius a_0 | 5.29177210903e-11 m |

Isotope masses are supplied by the caller (e.g. 1H 1.00782503223, 2H 2.01410177812,
12C 12 exactly, 13C 13.00335483507, 16O 15.99491461957, 18O 17.99915961286 amu; AME2020).
