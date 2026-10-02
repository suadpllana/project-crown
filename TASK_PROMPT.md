# Project Crown task-authoring prompt

Paste everything below the line into a fresh Claude Code session (an empty git repo works).
Fill in the three `{{...}}` fields first. The prompt follows the Crown expert guide and
the pattern of `tasks/bigeleisen_mayer_beta` in this repository. That task passes every
local gate: both reference solutions pass, the scaffolds fail, and 17 mutants are rejected.

> **Limits.** No prompt can guarantee acceptance. The difficulty evaluation and the human
> reviewer make independent judgements. This prompt covers every automated gate the
> guide lists and every common rejection reason. Pick a topic you can defend as a real
> research workflow in your own field.

---

## ROLE

You are a domain scientist in **{{DOMAIN: physics | biology | materials | math | chemistry | earth | ocean}}**
and a careful scientific software engineer. You are writing one task for AfterQuery
Project Crown, a benchmark of expert-written scientific coding tasks.

## TOPIC

**{{TOPIC: a real research or scientific-engineering workflow, for example "ocean mixed-layer depth and buoyancy frequency from a CTD cast using TEOS-10-style equations"}}**

Data available to you: **{{DATA: published constants, tables or real datasets you can cite, or "none / synthetic with seed"}}**

Do not reuse the computational structure of an existing task. These are already taken
in this repository:
- Bigeleisen-Mayer beta-factors from Hessians, and "frequencies → partition functions →
  thermochemistry";
- U-Pb discordia: Tera-Wasserburg → Wetherill conversion, York regression, concordia
  intercepts and their uncertainties. Treat any "errors-in-both-variables regression →
  curve intersection → propagated age" pipeline (Rb-Sr, Sm-Nd, Ar-Ar isochrons) as a near
  duplicate. Similarity checks compare against public benchmarks
(SciCode and others) and against earlier submissions. Changing only the domain wording,
variable names or dataset is rejected.

## WHAT TO BUILD

Create exactly this layout. Do not add containers, harness configs or platform files.

```
tasks/<task_id>/
  problem.yaml          # main problem + `sub_steps` list + contracts + metadata (domain: <token>)
  background.md         # equations, definitions, constants, conventions; no code, no expected values
  source.md             # literature, data provenance, what you designed, ground-truth evidence table
  solution.py           # complete integrated reference solution (all step functions)
  second_solution.py    # independent alternative: different algorithm/derivation/representation
  steps/step_1..N.py    # scaffolds: imports + exact signature + full docstring + raise NotImplementedError
  solution/step_1..N.py # per-step reference; SELF-CONTAINED (copies earlier steps it depends on)
  tests/step_1..N.py    # per-step tests
  tests/general.py      # whole-task integration tests against solution.py
harness/test_data.h5    # ONLY if you truly need stored numerical targets (prefer computing them in tests)
tools/                  # (outside the zip) assemble, negative-control and zip-build scripts
```

Use 4-6 steps. Each step must be a meaningful scientific computation that the next step
needs. Do not split trivial arithmetic into extra steps to make the task look bigger. The
last step, or `solution.py`, must solve the full main problem.

## SCIENTIFIC DESIGN RULES

1. The workflow must be real, something you would assign to a PhD student or research
   engineer. Name the papers or codes in which it is used.
2. The difficulty must be scientific. Build in at least three of the following, each
   based on a mistake that experts actually make:
   - a required projection, constraint, or conservation law that is easy to skip
     (and a test where skipping it gives a wrong answer);
   - numerical stability over a stated range (underflow/overflow, cancellation; use log-space,
     expm1/log1p, etc.), with a test at the extreme of that range;
   - a convention that changes the answer (per-atom normalisation, sign, reference state,
     symmetry factors, units), stated explicitly and tested;
   - recomputing something rather than reusing it (for example, re-diagonalising with new parameters
     instead of rescaling);
   - integration of several individually simple pieces into a quantity with an independent check
     (for example, an analytic limit).
3. State every convention, constant (with values and source, such as CODATA 2018), unit,
   valid range, tolerance, output shape and the scalar-vs-array behaviour. Anything a test
   enforces must appear in `problem.yaml`. If a test checks it, the prompt says it.
4. Valid alternative methods must pass. If two routes are equally correct, say "either
   route is acceptable" and test only inputs on which they agree.

## problem.yaml MUST CONTAIN

`id` and `problem_id` (both equal to the folder name), `title, domain, language,
python_version, dependencies (pinned minimums, numpy only if possible), datasets,
result_type (exact / approximate / asymptotic)`

`main_problem`: description (objective), inputs, outputs, units_and_conventions,
constants, assumptions_and_valid_ranges, accuracy (numeric tolerances), validation_errors.

`sub_steps` (this exact key: the upload panel rejects `subproblems`, `steps` and
`substeps`): for each step give step_number (sequential from 1), solution (the path
`solution/step_N.py`), tests (the path `tests/step_N.py`), function_header (the SciCode-style
def line plus docstring with no body, generated from `steps/step_N.py` by
`tools/sync_problem_yaml.py`), name, function, signature, description
(the computation, conventions, and the pitfall stated as a requirement), inputs with
units/shapes/ranges, output with shape/units, raises (the exact exception type and
conditions), and depends_on.

## REFERENCE SOLUTIONS

- Use only numpy, plus scipy if truly needed. The code must be deterministic, with no
  network access, no wall-clock dependence, and must run in seconds.
- Each `solution/step_k.py` contains the reference code of the earlier steps it uses, so
  it runs on its own.
- `second_solution.py` must differ in substance: a different algorithm, derivation,
  solver or representation. A reformatted copy does not count. It must pass every test
  file.

## TESTS: THE PART THAT DECIDES ACCEPTANCE

Write tests as plain `def test_*():` functions using bare `assert`. They must run under
`pytest tests/step_k.py` and also under `python tests/step_k.py`. Give each test file a
`__main__` runner that exits nonzero on failure. Depend only on numpy; do not import
pytest inside the tests (write a tiny `_raises` context manager instead).

Every test file starts with this loader, so the same file works however the platform
runs it:

```python
import importlib.util, os, sys
from pathlib import Path
import numpy as np

_T_TAG = "step_1"                      # "general" in general.py
_T_DEFAULT = "solution/step_1.py"      # "solution.py" in general.py
_T_MODULE = None

def _t_load():
    global _T_MODULE
    if _T_MODULE is None:
        path = os.environ.get("CROWN_IMPL") or str(Path(__file__).resolve().parents[1] / _T_DEFAULT)
        spec = importlib.util.spec_from_file_location("_crown_impl_" + _T_TAG, path)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        _T_MODULE = mod
    return _T_MODULE

def _fn(name):                          # 1) candidate code concatenated above, 2) $CROWN_IMPL, 3) default file
    g = globals()
    return g[name] if name in g and callable(g[name]) else getattr(_t_load(), name)
```

Prefix every helper and constant in the tests with `_` or `_T_` so nothing collides with
candidate code that has been concatenated above it. Never fall back to the reference
solution when the candidate fails to load. A fallback lets empty solutions pass.

### Expected values must NOT come from the reference solution

For each target, use one of these sources and record it in the `source.md` evidence
table:

- an analytic result: a special case with a closed form, an exact limit, or a sum rule/identity;
- an independent recomputation inside the test file, using a different formulation such
  as another coordinate system, the full partition function instead of a reduced form,
  direct quadrature instead of a series, or a GF method instead of Cartesian
  diagonalisation;
- a value prescribed by construction (for example, build a matrix with a chosen spectrum
  and then add the "noise" the method must remove);
- a published measurement, used only as a loose plausibility bound with a stated
  justification.

### Required test categories in every step file

1. Numerical correctness on scientifically meaningful inputs, preferably real constants or data.
2. Boundary and limit cases: extreme range ends and analytic limits.
3. At least one case that rejects a common wrong approach. Name the approach in a comment.
4. Invariances: frame rotation, permutation, units consistency, antisymmetry, identity gives zero.
5. Shapes: exact output shape, ordering, and scalar-in → scalar-out behaviour if promised.
   Check `shape` before comparing values.
6. Invalid inputs raise the documented exception.
7. `general.py` runs the integrated pipeline end to end against independent targets plus
   at least one consistency check across steps (for example, a fitted coefficient →
   its analytic limit).

### Tolerances

Use `abs(a - e) <= atol + rtol*|e|` with a justification for each tolerance. Typical
sources are differences between constant sets (~1e-8), conditioning, and
series-truncation size. Avoid exact float equality. Do not make tolerances so tight that
a valid method fails, or so loose that a sign or factor-of-2 error passes.

## NEGATIVE CONTROLS: RUN THEM BEFORE YOU SUBMIT

Write `tools/negative_controls.py`. It writes `solution.py` plus an override snippet to
a temporary file, sets `CROWN_IMPL` to that file, runs the relevant test files as
subprocesses, and requires a nonzero exit code. Include at least:

- a stub (`NotImplementedError`), functions that return `None`, zero-returns of the right
  shape, and wrong shapes (extra element, extra dimension, truncated);
- one mutant per pitfall you designed, for example skipped projection, wrong convention,
  missing normalisation, naive unstable formula, a factor of 2π, reused stale quantity,
  wrong sign, wrong units, or silently accepting invalid input;
- each scaffold `steps/step_k.py` must fail `tests/step_k.py`;
- each `solution/step_k.py`, `solution.py` and `second_solution.py` must pass.

Also test the concatenated mode: `cat solution/step_k.py tests/step_k.py > x.py && python x.py`
must pass, and `cat steps/step_k.py tests/step_k.py` must fail.

Run the whole suite twice to show it is deterministic.

## DOCUMENTS

- **background.md**: the science, equations, definitions, constants table, limits and
  conventions. Do not include code or test values.
- **source.md**: references with full citations, what comes from the literature versus
  what you designed, data provenance (synthetic data with its generator, seeds and why
  it is representative; real data pinned to an immutable version), a ground-truth evidence
  table (target → independent evidence → validity range or limitation), and a summary of
  the validation you ran.

## DEFINITION OF DONE (all must be true)

- [ ] `python3 tools/validate_submission.py` prints OK, and the ZIP was built with
      `python3 tools/build_zip.py`, which re-validates the ZIP. Never zip by hand.

- [ ] The layout matches exactly. `domain:` is set. The zip contains `tasks/<task_id>/...`
      and no caches.
- [ ] Every tested requirement is stated in problem.yaml. No hidden requirements.
- [ ] The steps form a dependency chain, and the final step solves the main problem.
- [ ] Per-step references pass their tests, solution.py passes general.py, and
      second_solution.py passes all tests.
- [ ] Scaffolds, stubs, None, zeros and wrong shapes fail. Every designed-pitfall mutant fails.
- [ ] No expected value comes only from the reference. The evidence table covers each target.
- [ ] Tests are deterministic and offline, run in seconds, use only numpy, and contain no
      expected answers in solver-visible files.
- [ ] The task is original: a new computational structure, not a domain-swap of an
      existing task.

When finished, build the zip, run the validation scripts, and report each result.
