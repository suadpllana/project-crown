# Project Crown: expert scientific coding tasks

This repository holds AfterQuery Project Crown tasks built to the Crown expert guide.
Each task folder is packaged as its own submission ZIP in `dist/`.

| Task | Domain | ZIP | Capability tested |
|---|---|---|---|
| `tasks/bigeleisen_mayer_beta` | chemistry | `dist/bigeleisen_mayer_beta.zip` | Eckart-projected Hessian frequencies, then Bigeleisen-Mayer β-factors, then isotope fractionation |
| `tasks/upb_discordia_intercepts` | earth | `dist/upb_discordia_intercepts.zip` | Tera-Wasserburg to Wetherill error propagation, York regression, concordia intercepts, intercept-age uncertainties |

## upb_discordia_intercepts (Earth science, geochronology)

| Step | Function | Scientific content |
|---|---|---|
| 1 | `tera_wasserburg_to_wetherill` | 238U/206Pb, 207Pb/206Pb to 207Pb/235U, 206Pb/238U with full covariance propagation (238U/235U = 137.818) |
| 2 | `york_fit` | York et al. (2004) line with correlated errors; standard errors at the adjusted points; MSWD |
| 3 | `concordia_intercepts` | Every intersection with the concave Wetherill concordia in [-1000, 5000] Ma |
| 4 | `intercept_age_sigmas` | Implicit-function propagation of cov(a, b) to the intercept ages |
| 5 | `discordia_ages` | Integrated upper and lower intercept ages with the sqrt(MSWD) over-dispersion rule |

## bigeleisen_mayer_beta (Chemistry, isotope geochemistry)

| Step | Function | Scientific content |
|---|---|---|
| 1 | `vibrational_wavenumbers` | Mass-weighted Hessian, Eckart projection (linear molecules, noisy Hessians), imaginary modes |
| 2 | `log_reduced_partition_function_ratio` | Bigeleisen-Mayer RPFR in log space from 1 K to 1e6 K |
| 3 | `log_beta` | Per-atom β; each isotopologue re-diagonalised; no symmetry numbers |
| 4 | `high_temperature_coefficient` | Analytic T^-2 limit |
| 5 | `isotope_fractionation` | 1000 ln α between species, 1000/T fit, high-T limit |

## Layout

```
tasks/<task_id>/               submission content (problem.yaml, background.md, source.md,
                               solution.py, second_solution.py, steps/, solution/, tests/)
tools/src, tools/assemble.py, tools/negative_controls.py          (bigeleisen_mayer_beta)
tools/upb/src, tools/upb/assemble.py, tools/upb/negative_controls.py (upb_discordia_intercepts)
tools/build_zip.py             builds dist/<task_id>.zip for every task (or the ones named)
TASK_PROMPT.md                 reusable prompt for authoring further Crown tasks
```

The `tools/**/src` blocks are the single source for each task's reference code and test
helpers. Edit them, then rerun the matching `assemble.py`.

## Validate

```bash
pip install numpy pytest pyyaml
python tools/upb/negative_controls.py   # 20 mutants killed, scaffolds fail, references pass
python tools/negative_controls.py       # 17 mutants killed, scaffolds fail, references pass
python tools/build_zip.py
```

To run one task's tests directly:

```bash
cd tasks/upb_discordia_intercepts
python -m pytest -q tests/step_1.py tests/step_2.py tests/step_3.py tests/step_4.py tests/step_5.py tests/general.py
CROWN_IMPL=$PWD/second_solution.py python tests/general.py
```
