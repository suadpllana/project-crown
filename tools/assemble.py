"""Assemble solution/step_N.py and solution.py from the blocks in tools/src.

Each solution/step_N.py is self-contained: it carries the reference code of the
earlier steps it depends on, followed by the code of step N.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tools" / "src"
TASK = ROOT / "tasks" / "bigeleisen_mayer_beta"
header = (SRC / "header.py").read_text()
blocks = [(SRC / f"f{i}.py").read_text() for i in range(1, 6)]

for n in range(1, 6):
    parts = [f'"""Reference solution for Step {n}."""\n', header]
    for i in range(1, n):
        parts.append(f"\n\n# ---- Step {i} (dependency, reference implementation) ----\n")
        parts.append(blocks[i - 1])
    parts.append(f"\n\n# ---- Step {n} ----\n")
    parts.append(blocks[n - 1])
    (TASK / "solution" / f"step_{n}.py").write_text("".join(parts))

parts = ['"""Complete reference solution: equilibrium isotope fractionation from Cartesian Hessians."""\n', header]
for i in range(1, 6):
    parts.append(f"\n\n# ---- Step {i} ----\n")
    parts.append(blocks[i - 1])
(TASK / "solution.py").write_text("".join(parts))
print("assembled")

# ---- tests: shared independent helper library + per-file test bodies ----
lib = (SRC / "testlib.py").read_text()
species_body = (SRC / "t5body.py").read_text()
docs = {
    "step_1": "Tests for Step 1: projected harmonic vibrational wavenumbers.",
    "step_2": "Tests for Step 2: Bigeleisen-Mayer reduced partition function ratio.",
    "step_3": "Tests for Step 3: per-atom ln(beta) for an isotopologue pair.",
    "step_4": "Tests for Step 4: high-temperature coefficient of ln(beta).",
    "step_5": "Tests for Step 5: fractionation between two species and its 1/T fit.",
    "general": "Whole-task integration tests (complete solution).",
}
for tag, src in (("step_1", "t1"), ("step_2", "t2"), ("step_3", "t3"), ("step_4", "t4"),
                 ("step_5", "t5"), ("general", "tg")):
    body = (SRC / f"{src}.py").read_text().replace("#BODY#", species_body)
    text = f'"""{docs[tag]}\n\nExpected values are computed independently (Wilson GF method, analytic\nresults, complete partition functions); the implementation under test is never\nused to produce a target.\n"""\n' + lib + "\n\n" + body
    (TASK / "tests" / f"{tag}.py").write_text(text)
print("tests assembled")
