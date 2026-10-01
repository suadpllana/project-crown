"""Build one submission ZIP per task with the layout required by the Crown expert guide:

dist/<task_id>.zip
└── tasks/<task_id>/{problem.yaml, background.md, source.md, solution.py,
                     second_solution.py, steps/, solution/, tests/}

Usage:  python3 tools/build_zip.py [task_id ...]     (default: every task in tasks/)
No harness/test_data.h5 is included: the tests store no numerical targets.
"""
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["problem.yaml", "background.md", "source.md", "solution.py", "second_solution.py"]
SKIP = {"__pycache__", ".pytest_cache"}


def build(task_id):
    task = ROOT / "tasks" / task_id
    for name in REQUIRED:
        assert (task / name).is_file(), (task_id, name)
    meta = yaml.safe_load((task / "problem.yaml").read_text())
    assert meta["domain"] in {"physics", "biology", "materials", "math", "chemistry", "earth", "ocean"}
    n_steps = len(meta["subproblems"])
    assert [s["index"] for s in meta["subproblems"]] == list(range(1, n_steps + 1))
    for sub in ("steps", "solution", "tests"):
        for n in range(1, n_steps + 1):
            assert (task / sub / f"step_{n}.py").is_file(), (task_id, sub, n)
    assert (task / "tests" / "general.py").is_file()
    out = ROOT / "dist" / f"{task_id}.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(task.rglob("*")):
            if path.is_file() and not (SKIP & set(path.parts)) and path.suffix != ".pyc":
                zf.write(path, Path("tasks") / task_id / path.relative_to(task))
    with zipfile.ZipFile(out) as zf:
        print("%s: %d files, %d steps, domain=%s -> %s" % (task_id, len(zf.namelist()), n_steps,
                                                          meta["domain"], out.relative_to(ROOT)))


if __name__ == "__main__":
    ids = sys.argv[1:] or sorted(p.name for p in (ROOT / "tasks").iterdir() if p.is_dir())
    for task_id in ids:
        build(task_id)
