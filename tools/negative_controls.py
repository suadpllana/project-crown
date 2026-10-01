"""Negative controls: every mutant below must FAIL the listed test files.

Each mutant is solution.py with function definitions overridden by a snippet.
Run:  python3 tools/negative_controls.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "bigeleisen_mayer_beta"
BASE = (TASK / "solution.py").read_text()
ALL = ["step_1", "step_2", "step_3", "step_4", "step_5", "general"]

STUB = '''
def vibrational_wavenumbers(*a, **k): raise NotImplementedError
def log_reduced_partition_function_ratio(*a, **k): raise NotImplementedError
def log_beta(*a, **k): raise NotImplementedError
def high_temperature_coefficient(*a, **k): raise NotImplementedError
def isotope_fractionation(*a, **k): raise NotImplementedError
'''
NONE = STUB.replace("raise NotImplementedError", "return None")
ZEROS = '''
def vibrational_wavenumbers(coords, masses, hessian):
    n = len(masses); return np.zeros(3 * n - 6 if n > 2 else 3 * n - 5)
def log_reduced_partition_function_ratio(nl, nh, T):
    return np.zeros_like(np.asarray(T, float)) if np.ndim(T) else 0.0
def log_beta(c, ml, mh, h, T):
    return np.zeros_like(np.asarray(T, float)) if np.ndim(T) else 0.0
def high_temperature_coefficient(*a): return 0.0
def isotope_fractionation(a, b, T):
    z = np.zeros(len(T)); return {"thousand_ln_beta_a": z, "thousand_ln_beta_b": z,
        "thousand_ln_alpha": z, "fit": np.zeros(3), "a_high_T": 0.0}
'''
WRONG_SHAPE = '''
_v0 = vibrational_wavenumbers
def vibrational_wavenumbers(coords, masses, hessian):
    return np.concatenate([np.zeros(6), _v0(coords, masses, hessian)])
_l0 = log_reduced_partition_function_ratio
def log_reduced_partition_function_ratio(nl, nh, T):
    return np.atleast_2d(_l0(nl, nh, T))
_f0 = isotope_fractionation
def isotope_fractionation(a, b, T):
    out = _f0(a, b, T); out["fit"] = out["fit"][:2]; out["thousand_ln_alpha"] = out["thousand_ln_alpha"][None, :]; return out
'''
NO_PROJECTION = '''
def vibrational_wavenumbers(coords, masses, hessian):
    m = np.asarray(masses, float); H = np.asarray(hessian, float); n = len(m)
    if H.shape != (3 * n, 3 * n) or np.asarray(coords).shape != (n, 3) or n < 2 or np.any(m <= 0):
        raise ValueError
    s = np.sqrt(np.repeat(m, 3)); lam = np.linalg.eigvalsh(H / np.outer(s, s))
    n_ext = 5 if n == 2 or np.linalg.matrix_rank(np.asarray(coords) - np.asarray(coords)[0], 1e-6) == 1 else 6
    keep = np.sort(np.argsort(np.abs(lam))[n_ext:])
    lam = lam[keep] * HARTREE / (BOHR ** 2 * AMU)
    return np.sort(np.sign(lam) * np.sqrt(np.abs(lam)) / (2 * np.pi * C_LIGHT * 100))
'''
ALWAYS_SIX = '''
_v0 = vibrational_wavenumbers
def vibrational_wavenumbers(coords, masses, hessian):
    nu = _v0(coords, masses, hessian)
    return nu if len(nu) == 3 * len(masses) - 6 else nu[1:]
'''
FORGET_2PI = '''
_v0 = vibrational_wavenumbers
def vibrational_wavenumbers(coords, masses, hessian):
    return 2 * np.pi * _v0(coords, masses, hessian)
'''
NAIVE_PRODUCT = '''
def log_reduced_partition_function_ratio(nu_light, nu_heavy, T):
    nl = np.asarray(nu_light, float); nh = np.asarray(nu_heavy, float)
    if nl.shape != nh.shape or np.any(nl <= 0) or np.any(nh <= 0) or np.any(np.asarray(T) <= 0): raise ValueError
    t = np.asarray(T, float)[..., None]
    ul, uh = C2_CM_K * nl / t, C2_CM_K * nh / t
    q = lambda u: np.exp(-u / 2) / (1 - np.exp(-u))
    with np.errstate(all="ignore"):
        return np.log(np.prod(uh / ul * q(uh) / q(ul), axis=-1))
'''
NO_PREFACTOR = '''
_l0 = log_reduced_partition_function_ratio
def log_reduced_partition_function_ratio(nu_light, nu_heavy, T):
    return _l0(nu_light, nu_heavy, T) - np.sum(np.log(np.asarray(nu_heavy) / np.asarray(nu_light)))
'''
NO_PER_ATOM = '''
_b0 = log_beta
def log_beta(coords, ml, mh, hessian, T):
    n = int(np.sum(~np.isclose(ml, mh, rtol=1e-12, atol=0)))
    return _b0(coords, ml, mh, hessian, T) * n
'''
SYMMETRY_NUMBER = '''
_b0 = log_beta
def log_beta(coords, ml, mh, hessian, T):
    ml, mh = np.asarray(ml, float), np.asarray(mh, float)
    sym = lambda m: 2.0 if len(m) == 3 and abs(m[1] - m[2]) < 1e-9 or len(m) == 2 and abs(m[0] - m[1]) < 1e-9 else 1.0
    sym3 = lambda m: 2.0 if (len(m) == 3 and abs(m[0] - m[2]) < 1e-9 and m[0] != m[1]) else sym(m)
    return _b0(coords, ml, mh, hessian, T) + np.log(sym3(ml) / sym3(mh))
'''
HEAVY_USES_LIGHT_MODES = '''
def log_beta(coords, masses_light, masses_heavy, hessian, T):
    sites = _substitution_sites(masses_light, masses_heavy)
    nu_l = vibrational_wavenumbers(coords, masses_light, hessian)
    # wrong: scale every mode by the reduced-mass ratio of the substituted atom
    ml = np.asarray(masses_light, float)[sites[0]]; mh = np.asarray(masses_heavy, float)[sites[0]]
    nu_h = nu_l * np.sqrt(ml / mh)
    return log_reduced_partition_function_ratio(nu_l, nu_h, T) / sites.size
'''
HIGH_T_FACTOR = '''
_k0 = high_temperature_coefficient
def high_temperature_coefficient(*a):
    return 3.0 * _k0(*a)
'''
FIT_IN_INVERSE_T = '''
_f0 = isotope_fractionation
def isotope_fractionation(a, b, T):
    out = _f0(a, b, T); t = np.asarray(T, float); x = 1.0 / t
    out["fit"] = np.linalg.lstsq(np.column_stack([x ** 2, x, np.ones_like(x)]), out["thousand_ln_alpha"], rcond=None)[0]
    return out
'''
A_HIGH_T_UNITS = '''
_f0 = isotope_fractionation
def isotope_fractionation(a, b, T):
    out = _f0(a, b, T); out["a_high_T"] = out["a_high_T"] * 1e6; return out
'''
ALPHA_SIGN = '''
_f0 = isotope_fractionation
def isotope_fractionation(a, b, T):
    out = _f0(a, b, T); out["thousand_ln_alpha"] = -out["thousand_ln_alpha"]; out["fit"] = -out["fit"]; return out
'''
NO_VALIDATION = '''
_f0 = isotope_fractionation
def isotope_fractionation(a, b, T):
    try:
        return _f0(a, b, T)
    except ValueError:
        z = np.zeros(len(T)); return {"thousand_ln_beta_a": z, "thousand_ln_beta_b": z,
            "thousand_ln_alpha": z, "fit": np.zeros(3), "a_high_T": 0.0}
'''

MUTANTS = {
    "stub (NotImplementedError)": (STUB, ALL),
    "returns None": (NONE, ALL),
    "zero returns": (ZEROS, ALL),
    "wrong shapes": (WRONG_SHAPE, ["step_1", "step_2", "step_5", "general"]),
    "no Eckart projection (drop smallest |eig|)": (NO_PROJECTION, ["step_1"]),
    "linear molecules not handled": (ALWAYS_SIX, ["step_1", "step_3", "step_5", "general"]),
    "angular frequency reported": (FORGET_2PI, ["step_1", "step_3", "step_4", "step_5", "general"]),
    "naive product (underflow)": (NAIVE_PRODUCT, ["step_2"]),
    "missing u*/u prefactor": (NO_PREFACTOR, ["step_2", "step_3", "step_5", "general"]),
    "ln f not divided by n": (NO_PER_ATOM, ["step_3"]),
    "symmetry numbers included": (SYMMETRY_NUMBER, ["step_3"]),
    "heavy modes scaled, not re-diagonalised": (HEAVY_USES_LIGHT_MODES, ["step_3", "step_5", "general"]),
    "wrong high-T prefactor": (HIGH_T_FACTOR, ["step_4", "step_5", "general"]),
    "fit in 1/T instead of 1000/T": (FIT_IN_INVERSE_T, ["step_5", "general"]),
    "a_high_T not scaled by 1e-6": (A_HIGH_T_UNITS, ["step_5", "general"]),
    "alpha sign reversed": (ALPHA_SIGN, ["step_5", "general"]),
    "invalid input silently accepted": (NO_VALIDATION, ["step_5"]),
}


def run(impl, test):
    env = dict(os.environ, CROWN_IMPL=str(impl))
    p = subprocess.run([sys.executable, str(TASK / "tests" / f"{test}.py")], env=env,
                       capture_output=True, text=True, timeout=300)
    return p.returncode


def main():
    bad = 0
    with tempfile.TemporaryDirectory() as d:
        for name, (snippet, tests) in MUTANTS.items():
            impl = Path(d) / "mutant.py"
            impl.write_text(BASE + "\n\n# ---- mutation ----\n" + snippet)
            res = {t: run(impl, t) for t in tests}
            survived = [t for t, rc in res.items() if rc == 0]
            status = "OK (killed)" if not survived else "SURVIVED in %s" % survived
            bad += bool(survived)
            print("%-45s %s" % (name, status))
        for n in range(1, 6):
            rc = run(TASK / "steps" / f"step_{n}.py", f"step_{n}")
            bad += rc == 0
            print("%-45s %s" % ("scaffold steps/step_%d.py" % n, "OK (fails)" if rc else "PASSES (bad)"))
        for n in range(1, 6):
            rc = run(TASK / "solution" / f"step_{n}.py", f"step_{n}")
            bad += rc != 0
            print("%-45s %s" % ("reference solution/step_%d.py" % n, "OK (passes)" if not rc else "FAILS"))
        for ref in ("solution.py", "second_solution.py"):
            res = {t: run(TASK / ref, t) for t in ALL}
            failed = [t for t, rc in res.items() if rc != 0]
            bad += bool(failed)
            print("%-45s %s" % ("reference " + ref, "OK (passes)" if not failed else "FAILS %s" % failed))
    print("negative controls:", "ALL GOOD" if not bad else "%d problems" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
