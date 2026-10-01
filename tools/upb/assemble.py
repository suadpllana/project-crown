"""Assemble solution/step_N.py, solution.py and tests/*.py of the U-Pb discordia task
from the blocks in tools/upb/src. Each solution/step_N.py is self-contained."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent / "src"
TASK = ROOT / "tasks" / "upb_discordia_intercepts"
N_STEPS = 5

header = (SRC / "header.py").read_text()
blocks = [(SRC / f"f{i}.py").read_text() for i in range(1, N_STEPS + 1)]
for sub in ("solution", "tests", "steps"):
    (TASK / sub).mkdir(parents=True, exist_ok=True)

for n in range(1, N_STEPS + 1):
    parts = [f'"""Reference solution for Step {n}."""\n', header]
    for i in range(1, n):
        parts += [f"\n\n# ---- Step {i} (dependency, reference implementation) ----\n", blocks[i - 1]]
    parts += [f"\n\n# ---- Step {n} ----\n", blocks[n - 1]]
    (TASK / "solution" / f"step_{n}.py").write_text("".join(parts))

parts = ['"""Complete reference solution: U-Pb discordia intercept ages from Tera-Wasserburg data."""\n', header]
for i in range(1, N_STEPS + 1):
    parts += [f"\n\n# ---- Step {i} ----\n", blocks[i - 1]]
(TASK / "solution.py").write_text("".join(parts))

lib = (SRC / "testlib.py").read_text()
docs = {
    "step_1": "Tests for Step 1: Tera-Wasserburg -> Wetherill conversion with error propagation.",
    "step_2": "Tests for Step 2: York (2004) regression with correlated errors.",
    "step_3": "Tests for Step 3: intersections of a line with the Wetherill concordia.",
    "step_4": "Tests for Step 4: intercept-age uncertainties.",
    "step_5": "Tests for Step 5: discordia intercept ages from Tera-Wasserburg analyses.",
    "general": "Whole-task integration tests (complete solution).",
}
for tag, src in (("step_1", "t1"), ("step_2", "t2"), ("step_3", "t3"), ("step_4", "t4"),
                 ("step_5", "t5"), ("general", "tg")):
    text = (f'"""{docs[tag]}\n\nExpected values come from published results, analytic constructions and an\n'
            'independent pipeline (log-space error propagation, maximum-likelihood line by direct\n'
            'minimisation, geometric adjusted points, dense-scan roots, finite differences).\n'
            'The implementation under test is never used to produce a target.\n"""\n'
            + lib + "\n\n" + (SRC / f"{src}.py").read_text())
    (TASK / "tests" / f"{tag}.py").write_text(text)
print("assembled", TASK.name)
