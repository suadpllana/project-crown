"""Negative controls for the U-Pb discordia task: every mutant must FAIL the listed
test files; scaffolds must fail; all references must pass.

Run:  python3 tools/upb/negative_controls.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "tasks" / "upb_discordia_intercepts"
BASE = (TASK / "solution.py").read_text()
ALL = ["step_1", "step_2", "step_3", "step_4", "step_5", "general"]
NAMES = ["tera_wasserburg_to_wetherill", "york_fit", "concordia_intercepts",
         "intercept_age_sigmas", "discordia_ages"]

STUB = "".join("def %s(*a, **k): raise NotImplementedError\n" % n for n in NAMES)
NONE = STUB.replace("raise NotImplementedError", "return None")
ZEROS = '''
def tera_wasserburg_to_wetherill(X, sX, Y, sY, rho): return np.zeros((len(X), 5))
def york_fit(x, sx, y, sy, rho): return {"intercept": 0.0, "slope": 0.0, "cov": np.zeros((2, 2)), "mswd": 0.0, "n": len(x)}
def concordia_intercepts(a, b): return np.zeros(2)
def intercept_age_sigmas(a, b, cov, ages): return np.zeros(len(ages))
def discordia_ages(X, sX, Y, sY, rho):
    return {"intercept": 0.0, "slope": 0.0, "cov": np.zeros((2, 2)), "mswd": 0.0, "n": len(X),
            "ages": np.zeros(2), "sigmas": np.zeros(2)}
'''
WRONG_SHAPE = '''
_c0 = tera_wasserburg_to_wetherill
def tera_wasserburg_to_wetherill(*a): return _c0(*a).T
_y0 = york_fit
def york_fit(*a):
    o = _y0(*a); o["cov"] = np.diag(o["cov"]); return o
_r0 = concordia_intercepts
def concordia_intercepts(a, b): return np.atleast_2d(_r0(a, b))
_d0 = discordia_ages
def discordia_ages(*a):
    o = _d0(*a); o["ages"] = o["ages"][:1]; o["sigmas"] = o["sigmas"][None, :]; return o
'''
TW_NO_CORRELATION = '''
_c0 = tera_wasserburg_to_wetherill
def tera_wasserburg_to_wetherill(X, sX, Y, sY, rho):
    return _c0(X, sX, Y, sY, np.zeros_like(np.asarray(rho, float)))
'''
OLD_URANIUM_RATIO = "\nU238_U235 = 137.88\n"
WETHERILL_RHO_ZERO = '''
_c0 = tera_wasserburg_to_wetherill
def tera_wasserburg_to_wetherill(*a):
    o = _c0(*a); o[:, 4] = 0.0; return o
'''
OLS = '''
def york_fit(x, sx, y, sy, rho):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.any(np.asarray(sx) <= 0) or np.any(np.abs(rho) >= 1) or len(y) != len(x): raise ValueError
    A = np.column_stack([np.ones_like(x), x]); p, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    s2 = np.sum((y - A @ p) ** 2) / (len(x) - 2); cov = s2 * np.linalg.inv(A.T @ A)
    return {"intercept": p[0], "slope": p[1], "cov": cov, "mswd": 1.0, "n": len(x)}
'''
YORK_IGNORES_RHO = '''
_y0 = york_fit
def york_fit(x, sx, y, sy, rho): return _y0(x, sx, y, sy, np.zeros_like(np.asarray(rho, float)))
'''
YORK_OBSERVED_POINTS = '''
_y0 = york_fit
def york_fit(x, sx, y, sy, rho):
    o = _y0(x, sx, y, sy, rho); x, sx, sy, rho = (np.asarray(v, float) for v in (x, sx, sy, rho))
    b = o["slope"]; w = 1 / (sy ** 2 + b * b * sx ** 2 - 2 * b * rho * sx * sy)
    xb = w @ x / w.sum(); vb = 1 / (w @ (x - xb) ** 2); va = 1 / w.sum() + xb ** 2 * vb
    o["cov"] = np.array([[va, -xb * vb], [-xb * vb, vb]]); return o
'''
MSWD_WRONG_DOF = '''
_y0 = york_fit
def york_fit(*a):
    o = _y0(*a); o["mswd"] = o["mswd"] * (o["n"] - 2) / o["n"]; return o
'''
COV_SIGN = '''
_y0 = york_fit
def york_fit(*a):
    o = _y0(*a); c = o["cov"].copy(); c[0, 1] = c[1, 0] = -c[0, 1]; o["cov"] = c; return o
'''
NEWTON_ONE_ROOT = '''
def concordia_intercepts(intercept, slope):
    t = 2000.0
    for _ in range(100):
        f = float(_concordia_misfit(t, intercept, slope))
        d = (LAMBDA_238 * np.exp(LAMBDA_238 * t * 1e6) - slope * LAMBDA_235 * np.exp(LAMBDA_235 * t * 1e6)) * 1e6
        t -= f / d
    return np.array([t]) if T_MIN_MA <= t <= T_MAX_MA and np.isfinite(t) else np.array([])
'''
ROOTS_DESCENDING = '''
_r0 = concordia_intercepts
def concordia_intercepts(a, b): return _r0(a, b)[::-1]
'''
NO_WINDOW = '''
T_MIN_MA = -5000.0
T_MAX_MA = 10000.0
'''
SIGMA_NO_COV = '''
_s0 = intercept_age_sigmas
def intercept_age_sigmas(a, b, cov, ages):
    c = np.asarray(cov, float); return _s0(a, b, np.diag(np.diag(c)) if c.shape == (2, 2) else c, ages)
'''
SIGMA_UNITS = '''
_s0 = intercept_age_sigmas
def intercept_age_sigmas(a, b, cov, ages): return _s0(a, b, cov, ages) * 1e-6 * 1e6 * 1.0e3
'''
ALWAYS_INFLATE = '''
_d0 = discordia_ages
def discordia_ages(*a):
    o = _d0(*a)
    if o["mswd"] <= 1: o["sigmas"] = o["sigmas"] * np.sqrt(o["mswd"])
    return o
'''
NEVER_INFLATE = '''
_d0 = discordia_ages
def discordia_ages(*a):
    o = _d0(*a)
    if o["mswd"] > 1: o["sigmas"] = o["sigmas"] / np.sqrt(o["mswd"])
    return o
'''
SINGLE_INTERCEPT_ACCEPTED = '''
def discordia_ages(X, sX, Y, sY, rho):
    w = tera_wasserburg_to_wetherill(X, sX, Y, sY, rho)
    fit = york_fit(*w.T); ages = concordia_intercepts(fit["intercept"], fit["slope"])
    ages = np.resize(ages, 2) if ages.size else np.zeros(2)
    sig = intercept_age_sigmas(fit["intercept"], fit["slope"], fit["cov"], ages) * np.sqrt(max(fit["mswd"], 1))
    o = dict(fit); o.update(ages=ages, sigmas=sig); return o
'''

MUTANTS = {
    "stub (NotImplementedError)": (STUB, ALL),
    "returns None": (NONE, ALL),
    "zero returns": (ZEROS, ALL),
    "wrong shapes": (WRONG_SHAPE, ["step_1", "step_2", "step_3", "step_5", "general"]),
    "TW error correlation ignored": (TW_NO_CORRELATION, ["step_1", "step_5", "general"]),
    "238U/235U = 137.88": (OLD_URANIUM_RATIO, ["step_1", "step_5", "general"]),
    "Wetherill correlation set to zero": (WETHERILL_RHO_ZERO, ["step_1", "step_5", "general"]),
    "ordinary least squares": (OLS, ["step_2", "step_5", "general"]),
    "York without error correlations": (YORK_IGNORES_RHO, ["step_2", "step_5", "general"]),
    "York errors at observed points": (YORK_OBSERVED_POINTS, ["step_2", "step_5"]),
    "MSWD with n dof": (MSWD_WRONG_DOF, ["step_2", "step_5", "general"]),
    "cov(a,b) sign flipped": (COV_SIGN, ["step_2", "step_5", "general"]),
    "single Newton root": (NEWTON_ONE_ROOT, ["step_3", "step_5", "general"]),
    "intercepts not ascending": (ROOTS_DESCENDING, ["step_3", "step_5", "general"]),
    "age window not applied": (NO_WINDOW, ["step_3"]),
    "age sigma without covariance": (SIGMA_NO_COV, ["step_4", "step_5", "general"]),
    "age sigma wrong units": (SIGMA_UNITS, ["step_4", "step_5", "general"]),
    "sigmas deflated when MSWD < 1": (ALWAYS_INFLATE, ["step_5", "general"]),
    "sigmas never inflated": (NEVER_INFLATE, ["step_5", "general"]),
    "single intercept accepted": (SINGLE_INTERCEPT_ACCEPTED, ["step_5"]),
}


def run(impl, test):
    env = dict(os.environ, CROWN_IMPL=str(impl))
    p = subprocess.run([sys.executable, str(TASK / "tests" / f"{test}.py")], env=env,
                       capture_output=True, text=True, timeout=600)
    return p.returncode


def main():
    bad = 0
    with tempfile.TemporaryDirectory() as d:
        for name, (snippet, tests) in MUTANTS.items():
            impl = Path(d) / "mutant.py"
            impl.write_text(BASE + "\n\n# ---- mutation ----\n" + snippet)
            survived = [t for t in tests if run(impl, t) == 0]
            bad += bool(survived)
            print("%-42s %s" % (name, "OK (killed)" if not survived else "SURVIVED in %s" % survived))
    for n in range(1, 6):
        rc = run(TASK / "steps" / f"step_{n}.py", f"step_{n}")
        bad += rc == 0
        print("%-42s %s" % ("scaffold steps/step_%d.py" % n, "OK (fails)" if rc else "PASSES (bad)"))
    for n in range(1, 6):
        rc = run(TASK / "solution" / f"step_{n}.py", f"step_{n}")
        bad += rc != 0
        print("%-42s %s" % ("reference solution/step_%d.py" % n, "OK (passes)" if not rc else "FAILS"))
    for ref in ("solution.py", "second_solution.py"):
        failed = [t for t in ALL if run(TASK / ref, t) != 0]
        bad += bool(failed)
        print("%-42s %s" % ("reference " + ref, "OK (passes)" if not failed else "FAILS %s" % failed))
    print("negative controls:", "ALL GOOD" if not bad else "%d problems" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
