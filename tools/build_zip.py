"""Build dist/submission.zip with the layout required by the Crown expert guide:

submission.zip
└── tasks/<task-id>/{problem.yaml, background.md, source.md, solution.py,
                     second_solution.py, steps/, solution/, tests/}

No harness/test_data.h5 is included: the tests store no numerical targets.
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "bigeleisen_mayer_beta"
TASK = ROOT / "tasks" / TASK_ID
REQUIRED = ["problem.yaml", "background.md", "source.md", "solution.py", "second_solution.py"]


def main():
    for name in REQUIRED:
        assert (TASK / name).is_file(), name
    for sub in ("steps", "solution", "tests"):
        for n in range(1, 6):
            assert (TASK / sub / f"step_{n}.py").is_file(), (sub, n)
    assert (TASK / "tests" / "general.py").is_file()
    out = ROOT / "dist" / "submission.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(TASK.rglob("*")):
            if path.is_file() and not ({"__pycache__", ".pytest_cache"} & set(path.parts)) and path.suffix != ".pyc":
                zf.write(path, Path("tasks") / TASK_ID / path.relative_to(TASK))
    with zipfile.ZipFile(out) as zf:
        for name in zf.namelist():
            print(name)
    print("wrote", out)


if __name__ == "__main__":
    main()
