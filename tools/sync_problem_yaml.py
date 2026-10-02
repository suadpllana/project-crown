"""Fill the derived fields of every tasks/<id>/problem.yaml from their single sources of
truth, so the Crown upload panel finds them and they can never drift:

  sub_steps[k].function_header         <- def line + docstring of steps/step_{k}.py
  sub_steps[k].return_line             <- final return of the reference function
  sub_steps[k].step_description_prompt <- sub_steps[k].description
  problem_name                         <- title
  problem_description_main             <- main_problem.description
  required_dependencies                <- import lines of solution.py

Run after editing a scaffold or problem.yaml:  python3 tools/sync_problem_yaml.py [--check]
With --check nothing is written; exit 1 if any file is out of date (used by the validator).
"""
import ast
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STEP_ORDER = ["step_number", "index", "solution", "tests", "name", "function", "signature",
              "function_header", "return_line", "description", "step_description_prompt",
              "step_background", "inputs", "output", "raises", "depends_on"]
TOP_ORDER = ["id", "problem_id", "title", "problem_name", "domain", "subdomain",
             "difficulty_sources", "language", "python_version", "dependencies",
             "required_dependencies", "datasets", "result_type", "problem_description_main",
             "main_problem", "sub_steps"]


class _Dumper(yaml.SafeDumper):
    pass


def _str(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_Dumper.add_representer(str, _str)


def _function_node(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise SystemExit("function %s not found" % name)


def function_header(scaffold_src, name):
    """def line(s) + docstring, exactly as written in the scaffold (no body)."""
    node = _function_node(scaffold_src, name)
    lines = scaffold_src.splitlines()
    doc = node.body[0]
    if not (isinstance(doc, ast.Expr) and isinstance(getattr(doc, "value", None), ast.Constant)
            and isinstance(doc.value.value, str)):
        raise SystemExit("scaffold for %s must start with a docstring" % name)
    return "\n".join(l.rstrip() for l in lines[node.lineno - 1:doc.end_lineno]) + "\n"


def return_line(solution_src, name):
    node = _function_node(solution_src, name)
    ret = [n for n in node.body if isinstance(n, ast.Return)]
    if ret:
        seg = ast.get_source_segment(solution_src, ret[-1])
        if seg and "\n" not in seg:
            return "    " + seg
    return "    return result"


def _ordered(d, order):
    out = {k: d[k] for k in order if k in d}
    out.update({k: v for k, v in d.items() if k not in out})
    return out


def sync(task_dir):
    path = task_dir / "problem.yaml"
    text = path.read_text()
    meta = yaml.safe_load(text)
    meta["problem_name"] = meta["title"]
    meta["problem_description_main"] = meta["main_problem"]["description"]
    sol = (task_dir / "solution.py").read_text()
    imports = [l for l in sol.splitlines() if l.startswith(("import ", "from "))]
    meta["required_dependencies"] = "\n".join(imports) + "\n"
    steps = []
    for k, step in enumerate(meta["sub_steps"], start=1):
        fn = step["function"]
        step["function_header"] = function_header((task_dir / "steps" / f"step_{k}.py").read_text(), fn)
        step["return_line"] = return_line((task_dir / "solution" / f"step_{k}.py").read_text(), fn)
        step["step_description_prompt"] = step["description"]
        steps.append(_ordered(step, STEP_ORDER))
    meta["sub_steps"] = steps
    meta = _ordered(meta, TOP_ORDER)
    new = yaml.dump(meta, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100)
    return path, text, new


def main(argv):
    check = "--check" in argv
    ids = [a for a in argv if not a.startswith("--")] or sorted(
        p.name for p in (ROOT / "tasks").iterdir() if p.is_dir())
    stale = []
    for task_id in ids:
        path, old, new = sync(ROOT / "tasks" / task_id)
        if old != new:
            stale.append(task_id)
            if not check:
                path.write_text(new)
    if check and stale:
        print("problem.yaml out of date (run python3 tools/sync_problem_yaml.py):", ", ".join(stale))
        return 1
    if not check:
        print("synced:", ", ".join(ids), "(changed: %s)" % (", ".join(stale) or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
