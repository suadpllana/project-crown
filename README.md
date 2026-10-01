# Project Crown: expert scientific coding task

This repository holds one task for AfterQuery Project Crown, built to the Crown expert
guide.

**Task:** `tasks/bigeleisen_mayer_beta` (domain: chemistry, in the isotope-geochemistry
workflow). It computes equilibrium stable-isotope fractionation factors from a single
quantum-chemistry Cartesian Hessian.

| Step | Function | Scientific content |
|---|---|---|
| 1 | `vibrational_wavenumbers` | Mass-weighted Hessian, Eckart projection of translations and rotations (handles linear molecules and noisy Hessians), imaginary modes |
| 2 | `log_reduced_partition_function_ratio` | Bigeleisen-Mayer RPFR evaluated stably in log space from 1 K to 1e6 K |
| 3 | `log_beta` | Per-atom beta-factor; each isotopologue re-diagonalised with its own masses; substitution rules; no symmetry numbers |
| 4 | `high_temperature_coefficient` | Analytic T^-2 limit (frequency or force-constant form) |
| 5 | `isotope_fractionation` | 1000 ln α between two species, its least-squares fit in 1000/T, and the high-T limit |

## Layout

```
tasks/bigeleisen_mayer_beta/   submission content (problem.yaml, background.md, source.md,
                               solution.py, second_solution.py, steps/, solution/, tests/)
tools/src/                     single source for the reference code and test helpers
tools/assemble.py              regenerates solution/step_N.py, solution.py and tests/*.py
tools/negative_controls.py     mutants, scaffolds and references against all tests
tools/build_zip.py             builds dist/submission.zip
dist/submission.zip            the file to upload
TASK_PROMPT.md                 reusable prompt for authoring further Crown tasks
```

## Validate

```bash
pip install numpy pytest
cd tasks/bigeleisen_mayer_beta
python -m pytest -q tests/step_1.py tests/step_2.py tests/step_3.py tests/step_4.py tests/step_5.py tests/general.py
CROWN_IMPL=$PWD/second_solution.py python -m pytest -q tests/general.py
cd ../.. && python tools/negative_controls.py && python tools/build_zip.py
```

Current status: 35 of 35 tests pass for the reference and for the second solution. All
5 scaffolds fail, and all 17 mutants fail, including stub, None, zero-return,
wrong-shape, and each designed pitfall.
