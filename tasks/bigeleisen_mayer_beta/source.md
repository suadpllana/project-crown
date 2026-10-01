# Sources, provenance and ground-truth evidence

## Scientific sources

* J. Bigeleisen and M. G. Mayer, "Calculation of equilibrium constants for isotopic
  exchange reactions", *J. Chem. Phys.* 15, 261 (1947). Reduced partition function ratio,
  its high-temperature expansion and the force-constant form.
* H. C. Urey, "The thermodynamic properties of isotopic substances", *J. Chem. Soc.*
  562 (1947). Complete partition-function formulation of beta-factors.
* O. Redlich, *Z. Phys. Chem. B* 28, 371 (1935); E. Teller (unpublished, cited therein):
  the Teller-Redlich product rule.
* E. B. Wilson, J. C. Decius, P. C. Cross, *Molecular Vibrations* (McGraw-Hill, 1955):
  Wilson GF method, B-matrix vectors for bond stretches and bends, Eckart conditions.
* P. Richet, Y. Bottinga, M. Javoy, "A review of hydrogen, carbon, nitrogen, oxygen,
  sulphur, and chlorine stable isotope fractionation among gaseous molecules",
  *Annu. Rev. Earth Planet. Sci.* 5, 65 (1977). Context for beta-factor conventions
  (per-atom normalisation, no symmetry numbers) and harmonic-model accuracy.
* C. A. M. Brenninkmeijer, P. Kraft, W. G. Mook, "Oxygen isotope fractionation between
  CO2 and H2O", *Isotope Geoscience* 1, 181 (1983): alpha(CO2(g)-H2O(l)) = 1.04115 at 25 C.
* M. Majoube, *J. Chim. Phys.* 68, 1423 (1971): alpha(18O, H2O(l)-H2O(v)) = 1.0094 at 25 C.
* W. S. Benedict, N. Gailar, E. K. Plyler, *J. Chem. Phys.* 24, 1139 (1956): harmonic
  wavenumbers and equilibrium structure of H2O and D2O (as tabulated in standard
  compilations: H2O 3832.2, 1648.5, 3942.5 cm^-1; D2O 2763.8, 1206.4, 2888.8 cm^-1;
  r_e = 0.9572 A, theta_e = 104.52 deg).
* CO2 harmonic wavenumbers ~1354.0, 672.9 (degenerate), 2396.3 cm^-1 and r_e = 1.160 A,
  rounded from the experimental analyses of Chedin, *J. Mol. Spectrosc.* 76, 430 (1979)
  and Suzuki, *J. Mol. Spectrosc.* 25, 479 (1968).
* K. P. Huber and G. Herzberg, *Constants of Diatomic Molecules* (1979): H2
  omega_e = 4401.21 cm^-1, r_e = 0.74144 A.
* CODATA 2018 fundamental constants; AME2020 atomic masses.

## What comes from the literature and what was designed here

The equations (Sections 2-5 of background.md) are standard and taken from the sources
above. The task design - decomposing the workflow into Hessian projection, a stable
log-space RPFR, a per-atom beta-factor with explicit substitution rules, the analytic
high-temperature coefficient and the fractionation/fit integration - the function
contracts, the validation rules, the force fields and all tests were written for this
task. No public implementation was copied.

## Data used by the tests

All test inputs are generated deterministically inside the test files; no external or
live data are used and no stored numerical targets (`test_data.h5`) are needed.

* **Water force field (real data, fitted here).** A four-parameter harmonic valence
  force field (f_r, f_rr', f_theta, f_r_theta in atomic units) on the experimental
  equilibrium structure, fitted so that the Wilson-GF wavenumbers of H2(16)O reproduce
  the experimental harmonic wavenumbers 1648.5, 3832.2, 3942.5 cm^-1 and D2(16)O
  reproduces 2763.8 cm^-1. The *predicted* D2O wavenumbers 1205.97 and 2888.76 cm^-1
  agree with experiment (1206.4, 2888.8) to < 0.5 cm^-1, an independent validation
  of the force field. Fitted values: f_r = 0.54306809 E_h/a_0^2 (8.45 mdyn/A, cf. the
  literature value 8.45), f_rr' = -0.00644291 E_h/a_0^2, f_theta = 0.15994866 E_h/rad^2,
  f_r_theta = 0.02727912 E_h/(a_0 rad).
* **CO2 force field (real data, fitted here).** f_r = 1.0289399, f_rr' = 0.08077316
  E_h/a_0^2 and f_delta = 0.17963341 E_h/rad^2 on r_e = 1.160 A, reproducing the
  harmonic wavenumbers listed above exactly; the 13C, 18O isotopologues are predictions.
* **H2.** Harmonic force constant from omega_e = 4401.21 cm^-1.
* **Synthetic Hessians (step 1 only).** Non-linear 4- and 5-atom molecules with random
  geometries (numpy default_rng seeds 2024 and 99) whose mass-weighted Hessians are
  constructed (seeds 7 and 5) to have a prescribed vibrational spectrum - including a
  25 cm^-1 soft mode and a -150 cm^-1 imaginary mode - plus a symmetric perturbation E
  with P E P = 0 that couples the vibrational and external spaces with a strength
  equivalent to 300 cm^-1 (resp. 120 cm^-1). This represents finite-difference noise
  in real Hessians; the expected answer is the prescribed spectrum *by construction*.

Cartesian Hessians of the real molecules are formed as H = B^T F B from the Wilson B
matrix at the equilibrium geometry (exact at a stationary point), and are therefore
exactly translation- and rotation-invariant.

## Ground-truth evidence for every expected value

None of the expected values is produced by the reference solution.

| Test target | Independent evidence |
|---|---|
| Diatomic wavenumbers, ln beta and K | Analytic harmonic oscillator nu = sqrt(k/mu)/(2 pi c); closed-form f |
| Polyatomic wavenumbers (H2O, HDO, D2O, H2(18)O, CO2 isotopologues) | Wilson GF method in internal coordinates (no Cartesian projection involved); additionally checked against experimental harmonic wavenumbers (exact for fitted quantities, < 0.5 cm^-1 for D2O predictions) |
| Contaminated / imaginary-mode Hessians | Spectrum prescribed by construction |
| ln f at ambient T | Closed sinh form, a different algebraic expression from the log-space form |
| ln f at 1 K and 5 K | Analytic low-temperature limit (neglected terms < e^-2000) |
| ln f at 2e4-1e6 K | Asymptotic series to u^6; truncation error < 1e-12 relative |
| ln beta of all isotopologue pairs | Complete translational x rotational x vibrational partition-function ratio times the classical mass factor (Urey route), using GF wavenumbers and principal moments; equal to the Bigeleisen-Mayer value only through the Teller-Redlich rule, so this is a genuine independent recomputation |
| K (high-T coefficient) | Bigeleisen-Mayer force-constant form from the 3x3 Cartesian Hessian blocks (no frequencies), and T^2 ln beta at 2e5 K (agreement 1e-4, the O(T^-2) correction) |
| Fit coefficients | Least squares on independently computed 1000 ln alpha |
| Fitted A over 2e4-1e5 K | Must approach the analytic limit a_high_T (tolerance 0.5 %, the size of the u^4 correction) |
| 1000 ln alpha(CO2(g)-H2O(g), 18O) at 25 C | Experiment: 1000 ln(1.04115) + 1000 ln(1.0094) = 49.7 per mil; the harmonic model gives 47.6. The test accepts 45.5-53.5 per mil because the harmonic approximation omits anharmonic corrections of a few per mil for 18O (Richet et al. 1977). |
| alpha(H2O(g)-H2, D/H) at 25 C | Experiment-derived ~3.5 (alpha(H2O(l)-H2) ~ 3.8 divided by alpha(l-v) = 1.079). Harmonic models overestimate D/H fractionation by several percent; the model gives 3.76; accepted range 3.2-4.3. |

Limitations: the published-data checks are plausibility bounds on the harmonic model,
not precision targets; precision targets come from the analytic and independent
recomputations above, which are exact within the model to ~1e-12 relative.

## Validation performed

* Per-step reference solutions pass their tests; solution.py passes general.py.
* second_solution.py (different projection algorithm, Urey partition-function route,
  force-constant high-T limit, polyfit) passes every test file.
* Scaffolds (steps/*.py) fail every test. 17 negative-control mutants fail
  (stub, None, zeros, wrong shapes, no projection, linear molecules mishandled, angular
  frequency, naive product with underflow, missing u*/u prefactor, missing 1/n,
  symmetry numbers included, heavy modes scaled instead of re-diagonalised, wrong
  high-T prefactor, fit in 1/T, a_high_T units, reversed alpha sign, invalid input
  accepted).
* Tests are deterministic (fixed seeds), offline, and run in < 1 s.
