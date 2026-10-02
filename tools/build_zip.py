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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_submission import JUNK_PARTS, STEPS_KEY, validate_folder, validate_zip  # noqa: E402
from sync_problem_yaml import sync  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def build(task_id):
    task = ROOT / "tasks" / task_id
    # Regenerate derived problem.yaml fields (function_header, ...) before validating.
    path, cur, want = sync(task)
    if cur != want:
        path.write_text(want)
        print("%s: synced derived fields in problem.yaml" % task_id)
    errors = validate_folder(task_id)
    if errors:
        raise SystemExit("refusing to build %s:\n  - %s" % (task_id, "\n  - ".join(errors)))
    meta = yaml.safe_load((task / "problem.yaml").read_text())
    n_steps = len(meta[STEPS_KEY])
    out = ROOT / "dist" / f"{task_id}.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(task.rglob("*")):
            if path.is_file() and not (JUNK_PARTS & set(path.parts)) and path.suffix != ".pyc":
                zf.write(path, Path("tasks") / task_id / path.relative_to(task))
    errors = validate_zip(out)
    if errors:
        out.unlink()
        raise SystemExit("built ZIP failed validation and was deleted:\n  - " + "\n  - ".join(errors))
    with zipfile.ZipFile(out) as zf:
        print("%s: %d files, %d steps, domain=%s -> %s" % (task_id, len(zf.namelist()), n_steps,
                                                          meta["domain"], out.relative_to(ROOT)))


if __name__ == "__main__":
    ids = sys.argv[1:] or sorted(p.name for p in (ROOT / "tasks").iterdir() if p.is_dir())
    for task_id in ids:
        build(task_id)
