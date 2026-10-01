"""Pre-upload validator for Crown task folders and submission ZIPs.

Encodes every rule the Crown upload panel enforces that we know of. It records each
platform error we have hit so it cannot happen again (see CLAUDE.md, "Platform feedback
log"). build_zip.py runs it and refuses to write a ZIP that fails.

Usage:
    python3 tools/validate_submission.py                 # every folder in tasks/
    python3 tools/validate_submission.py <task_id> ...   # selected folders
    python3 tools/validate_submission.py dist/x.zip      # an already-built ZIP
"""
import ast
import io
import re
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = {"physics", "biology", "materials", "math", "chemistry", "earth", "ocean"}
STEPS_KEY = "sub_steps"
# Keys the platform does NOT accept for the step list (platform error 2026-10, see CLAUDE.md).
FORBIDDEN_STEP_KEYS = {"subproblems", "subproblem", "steps", "substeps", "sub-steps", "subSteps"}
REQUIRED_TOP = ["problem_id", "title", "domain", "main_problem", STEPS_KEY]
REQUIRED_STEP = ["step_number", "name", "function", "signature", "description",
                 "inputs", "output", "depends_on"]
REQUIRED_FILES = ["problem.yaml", "background.md", "source.md", "solution.py", "tests/general.py"]
JUNK_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".DS_Store"}


def _defined_functions(source, label, errors):
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        errors.append("%s: syntax error: %s" % (label, exc))
        return set()
    return {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def validate_files(task_id, files):
    """files: mapping of path relative to the task folder -> text content."""
    errors = []
    if not re.fullmatch(r"[a-z0-9_]+", task_id):
        errors.append("task folder name %r should be lower-case letters, digits and underscores" % task_id)
    for name in REQUIRED_FILES:
        if name not in files:
            errors.append("missing required file %s" % name)
    if "problem.yaml" not in files:
        return errors
    try:
        meta = yaml.safe_load(files["problem.yaml"])
    except yaml.YAMLError as exc:
        return errors + ["problem.yaml is not valid YAML: %s" % exc]
    if not isinstance(meta, dict):
        return errors + ["problem.yaml must be a mapping at top level"]

    bad = FORBIDDEN_STEP_KEYS & set(meta)
    if bad:
        errors.append("problem.yaml uses %s for the step list; the platform requires %r"
                      % (sorted(bad), STEPS_KEY))
    for key in REQUIRED_TOP:
        if key not in meta:
            errors.append("problem.yaml is missing top-level key %r" % key)
    if meta.get("domain") not in DOMAINS:
        errors.append("domain must be one of %s, got %r" % (sorted(DOMAINS), meta.get("domain")))
    pid = meta.get("problem_id", meta.get("id"))
    if pid != task_id:
        errors.append("problem_id %r must equal the task folder name %r" % (pid, task_id))
    if "id" in meta and meta["id"] != pid:
        errors.append("id %r and problem_id %r disagree" % (meta["id"], pid))

    steps = meta.get(STEPS_KEY)
    if not isinstance(steps, list) or not steps:
        errors.append("%r must be a non-empty list" % STEPS_KEY)
        return errors
    n = len(steps)
    if n < 2:
        errors.append("a task needs several dependent steps; found %d" % n)
    for i, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            errors.append("sub_steps[%d] must be a mapping" % (i - 1))
            continue
        for key in REQUIRED_STEP:
            if key not in step:
                errors.append("sub_steps[%d] is missing %r" % (i - 1, key))
        if step.get("step_number") != i:
            errors.append("sub_steps[%d].step_number must be %d (sequential from 1), got %r"
                          % (i - 1, i, step.get("step_number")))
        if "index" in step and step["index"] != step.get("step_number"):
            errors.append("sub_steps[%d]: index and step_number disagree" % (i - 1))
        for dep in step.get("depends_on", []) or []:
            if not (isinstance(dep, int) and 1 <= dep < i):
                errors.append("sub_steps[%d].depends_on may only list earlier steps, got %r" % (i - 1, dep))
        fn = step.get("function")
        sig = str(step.get("signature", ""))
        if fn and ("def %s(" % fn) not in sig:
            errors.append("sub_steps[%d].signature does not define function %r" % (i - 1, fn))
        for sub in ("steps", "solution", "tests"):
            path = "%s/step_%d.py" % (sub, i)
            if path not in files:
                errors.append("missing %s for step %d" % (path, i))
            elif sub != "tests" and fn:
                if fn not in _defined_functions(files[path], path, errors):
                    errors.append("%s does not define %s()" % (path, fn))
        if "tests/step_%d.py" % i in files and "def test_" not in files["tests/step_%d.py" % i]:
            errors.append("tests/step_%d.py contains no test_ functions" % i)
    for sub in ("steps", "solution", "tests"):
        extra = [p for p in files if re.fullmatch(r"%s/step_(\d+)\.py" % sub, p)
                 and int(re.findall(r"\d+", p.split("/")[1])[0]) > n]
        if extra:
            errors.append("files for steps that are not in sub_steps: %s" % extra)
    for code in ("solution.py", "second_solution.py"):
        if code in files:
            defined = _defined_functions(files[code], code, errors)
            missing = [s.get("function") for s in steps if isinstance(s, dict) and s.get("function") not in defined]
            if missing:
                errors.append("%s does not define %s" % (code, missing))
    if "tests/general.py" in files and "def test_" not in files["tests/general.py"]:
        errors.append("tests/general.py contains no test_ functions")
    junk = [p for p in files if JUNK_PARTS & set(Path(p).parts) or p.endswith(".pyc")]
    if junk:
        errors.append("cache/junk files present: %s" % junk[:5])
    return errors


def validate_folder(task_id):
    task = ROOT / "tasks" / task_id
    files = {}
    for p in task.rglob("*"):
        if p.is_file():
            rel = p.relative_to(task).as_posix()
            if JUNK_PARTS & set(p.parts) or p.suffix == ".pyc":
                continue  # build_zip skips these; not an error in the working tree
            files[rel] = p.read_text(errors="replace")
    return validate_files(task_id, files)


def validate_zip(path):
    errors = []
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        tasks = {}
        for name in names:
            parts = name.split("/")
            if parts[0] not in ("tasks", "harness") or (parts[0] == "tasks" and len(parts) < 3):
                errors.append("unexpected path in ZIP: %s" % name)
                continue
            if parts[0] == "tasks" and not name.endswith("/"):
                tasks.setdefault(parts[1], {})["/".join(parts[2:])] = zf.read(name).decode("utf-8", "replace")
        if not tasks:
            errors.append("ZIP contains no tasks/<task_id>/problem.yaml")
        for task_id, files in tasks.items():
            errors += ["%s: %s" % (task_id, e) for e in validate_files(task_id, files)]
    return errors


def main(argv):
    targets = argv or sorted(p.name for p in (ROOT / "tasks").iterdir() if p.is_dir())
    failed = False
    for t in targets:
        errs = validate_zip(t) if t.endswith(".zip") else validate_folder(t)
        print("%-40s %s" % (t, "OK" if not errs else "FAILED"))
        for e in errs:
            print("   - " + e)
        failed |= bool(errs)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
