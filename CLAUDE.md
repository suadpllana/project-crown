# Project Crown: rules for working in this repository

This repo produces AfterQuery **Project Crown** tasks: expert-written scientific coding
tasks uploaded as one ZIP per task. Read this file before creating or editing any task.

## Non-negotiable workflow

1. Edit task sources (`tasks/<task_id>/...`, or the `tools/**/src` blocks plus the matching
   `assemble.py`). Then run `python3 tools/sync_problem_yaml.py` to regenerate the derived
   problem.yaml fields: `function_header`, `return_line`, `step_description_prompt`,
   `problem_name`, `problem_description_main` and `required_dependencies`. Never hand-edit
   those fields.
2. Run `python3 tools/validate_submission.py`. It must print `OK` for every task.
3. Run the task's negative controls (`tools/negative_controls.py`,
   `tools/upb/negative_controls.py`, ...). They must end with `ALL GOOD`.
4. Build with `python3 tools/build_zip.py [task_id]`. It validates the folder **and** the
   built ZIP, and refuses to write an invalid ZIP.
5. Never hand-zip a task and never edit a ZIP directly. Only ZIPs from `tools/build_zip.py`
   are uploaded.

A pre-commit hook (`.githooks/pre-commit`) runs the validator. Enable it once per clone
with `git config core.hooksPath .githooks`.

## problem.yaml schema (what the Crown upload panel accepts)

```yaml
id: <task_id>                 # same as the folder name: lower-case, digits, underscores
problem_id: <task_id>         # the platform identifies future versions by this id
title: "..."
domain: earth                 # exactly one of physics|biology|materials|math|chemistry|earth|ocean
main_problem: {description, inputs, outputs, units_and_conventions, assumptions_and_valid_ranges, accuracy, ...}
sub_steps:                    # <-- REQUIRED KEY NAME. Not "subproblems", "steps" or "substeps".
  - step_number: 1            # sequential from 1, matching steps/ solution/ tests/ step_N.py
    index: 1                  # optional; if present must equal step_number
    solution: solution/step_1.py   # REQUIRED: path of this step's reference solution
    tests: tests/step_1.py         # REQUIRED: path of this step's tests file
    function_header: |             # REQUIRED (SciCode key): def line + docstring, no body.
      def <python_name>(...):      #   GENERATED from steps/step_1.py by sync_problem_yaml.py
          """..."""
    return_line: "    return ..."  # generated (SciCode key)
    step_description_prompt: |     # generated copy of description (SciCode key)
    name: "..."
    function: <python_name>   # must be defined in steps/step_N.py, solution/step_N.py, solution.py
    signature: "def <python_name>(...)"
    description: |
      ...
    inputs: {...}
    output: "..."
    raises: "..."
    depends_on: [earlier step numbers only]
```

Folder layout per task: `problem.yaml, background.md, source.md, solution.py,
second_solution.py, steps/step_N.py, solution/step_N.py, tests/step_N.py, tests/general.py`.
`harness/test_data.h5` is only for stored numerical targets. Current tasks compute their
targets inside the tests and do not ship it.

When the platform's "Download template" layout differs from the schema above, the template
wins. Update this section, `REQUIRED_TOP`/`REQUIRED_STEP`/`STEPS_KEY` in
`tools/validate_submission.py`, and `TASK_PROMPT.md` together.

## Platform feedback log

Every error the Crown upload panel or a reviewer reports goes here, with the fix and the
check that now prevents it. Add a validator rule before (or together with) the fix.

| Date | Task | Platform message | Fix | Guard |
|---|---|---|---|---|
| 2026-10 | upb_discordia_intercepts (and, latently, bigeleisen_mayer_beta) | `problem.yaml has no "sub_steps" list, which holds the task's steps. It has "subproblems" instead; rename that key to "sub_steps".` The panel then showed "0 steps · no domain". | Renamed `subproblems` → `sub_steps` in every task; added `step_number` per step and top-level `problem_id`. | `validate_submission.py` rejects `subproblems`/`steps`/`substeps` and requires `sub_steps` with sequential `step_number`; `build_zip.py` refuses to build otherwise. |
| 2026-10 | upb_discordia_intercepts (and, latently, bigeleisen_mayer_beta) | `Step 1 of "sub_steps" has no "solution" (the path of its reference solution, like solution/step_1.py) ... has no "tests" (the path of its tests file, like tests/step_1.py)`. Repeated for every step. | Added `solution: solution/step_N.py` and `tests: tests/step_N.py` to every sub_step. | `validate_submission.py` requires both keys, checks they equal `solution/step_N.py` / `tests/step_N.py`, and that the files exist. |
| 2026-10 | upb_discordia_intercepts (and, latently, bigeleisen_mayer_beta) | `sub_step is missing a function_header.` (once per step), then `No sub-step could be parsed for this task.` | Added a SciCode-style `function_header` (def line + docstring) to every step, generated from `steps/step_N.py`. Pre-emptively added the other standard SciCode fields (`return_line`, `step_description_prompt`, `problem_name`, `problem_description_main`, `required_dependencies`). | `validate_submission.py` requires `function_header`, checks it parses, has a docstring, starts with `def <function>(` and equals the scaffold header. It also fails if any derived field is stale. `build_zip.py` re-syncs automatically. |
| 2026-10 | (info) | `tasks/_id_mapping.json is absent; falling back to delivered ids for task identity.` | Informational; identity comes from `problem_id`. | `problem_id` must equal the folder name. |

**Lesson:** the platform's schema is a SciCode derivative (`sub_steps`, `step_number`,
`function_header`, ...) in which file paths (`solution`, `tests`) replace inline code. When a
new key is demanded, check the SciCode field list first.

**Lesson:** the panel reports schema problems a few at a time. The authoritative layout is
the panel's **Download template**. When it is available, copy its exact keys into the
schema above and into the validator at once, rather than fixing one rejection per upload.

## Task-quality rules (from the Crown expert guide)

- A real research workflow with dependent steps, and difficulty that comes from the science.
- Every requirement a test enforces is stated in problem.yaml.
- Expected values never come only from the reference. Use analytic results, independent
  recomputation inside the tests, published values, or construction.
- Tests must reject stub, None, zero and wrong-shape solutions, plus one mutant per
  designed pitfall.
- `second_solution.py` must use a genuinely different method and must pass every test.
- Do not reuse the computational structure of an existing task. See the "taken" list in
  `TASK_PROMPT.md`.
- Commit messages and repository files must not mention model identifiers.
