"""
Classify HOW a submitted law is wrong, not just THAT it's wrong.

For each of the function's parameters, compute the local elasticity (the
effective power-law exponent) of both submitted_law and ground_truth_law at a
fixed reference point: elasticity_p = p * d(expr)/dp / expr. This is the
"local exponent if the whole expression behaved as a pure power law in p near
this point" -- exactly the same quantity you'd estimate empirically from
log-log slopes of experimental data (as we did by hand earlier in this
project), but computed exactly from the formulas via symbolic differentiation
instead of noisy finite differences.

Comparing submitted's elasticity to ground truth's, per parameter, gives a
specific, human-readable diagnosis instead of just "wrong":
  - missing_variable: ground truth depends on this parameter, submitted doesn't
  - extra_variable: submitted depends on a parameter ground truth doesn't
  - sign_flip: both depend on it, but with opposite-signed exponents
    (e.g. ground truth ~ 1/distance^2, submitted ~ distance^2)
  - wrong_exponent: both depend on it with the same sign, but different magnitude
  - other_structural: no single-parameter elasticity mismatch found, but
    structural_verdict still says different -- likely an operator-level
    difference (e.g. additive vs multiplicative combination) that a per-
    parameter power-law lens can't isolate; needs a human look.

This reuses submitted_law_to_sympy / ground_truth_to_sympy from
structural_equivalence.py, so it inherits the same "fails closed to
not_checkable" behavior for control-flow functions.
"""
import sys
from pathlib import Path

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from structural_equivalence import submitted_law_to_sympy, ground_truth_to_sympy, NotCheckable  # noqa: E402

ELASTICITY_ZERO_TOL = 1e-6
ELASTICITY_MATCH_TOL = 0.15  # exponents within this are treated as "the same"


def _reference_point(param_names):
    """Distinct, non-degenerate positive values per parameter -- avoids the
    coincidental cancellations a shared value like 1.0 for everything could cause
    (e.g. (a - b) vanishing when a=b=1)."""
    return {name: 2.0 + 0.7 * i for i, name in enumerate(sorted(param_names))}


def _elasticity_at_point(expr, param_symbol, subs):
    try:
        deriv = sp.diff(expr, param_symbol)
        val = complex(expr.subs(subs))
        deriv_val = complex(deriv.subs(subs))
        if abs(val) < 1e-300:
            return None
        return (subs[param_symbol] * deriv_val / val).real
    except Exception:
        return None


def classify_mismatch(submitted_law_src: str, ground_truth_str: str):
    """Returns a dict:
        {'mistake_type': str, 'details': {param_name: {'sub': float|None, 'gt': float|None, 'issue': str}}}
    mistake_type is one of: missing_variable, extra_variable, sign_flip,
    wrong_exponent, other_structural, not_checkable.
    Priority when multiple parameters have issues: missing_variable > sign_flip
    > extra_variable > wrong_exponent > other_structural.
    """
    try:
        sub_expr, sub_params = submitted_law_to_sympy(submitted_law_src)
        gt_expr = ground_truth_to_sympy(ground_truth_str, [str(p) for p in sub_params])
    except NotCheckable:
        return {"mistake_type": "not_checkable", "details": {}}
    except Exception:
        return {"mistake_type": "not_checkable", "details": {}}

    gt_extra_symbols = gt_expr.free_symbols - sub_params
    point = _reference_point([str(p) for p in sub_params])
    subs = {p: point[str(p)] for p in sub_params}
    subs.update({s: 1.0 for s in gt_extra_symbols})  # arbitrary positive value for the "real constant"

    details = {}
    for p in sorted(sub_params, key=str):
        e_sub = _elasticity_at_point(sub_expr, p, subs)
        e_gt = _elasticity_at_point(gt_expr, p, subs)

        sub_zero = e_sub is None or abs(e_sub) < ELASTICITY_ZERO_TOL
        gt_zero = e_gt is None or abs(e_gt) < ELASTICITY_ZERO_TOL

        if gt_zero and sub_zero:
            issue = "matches"  # neither depends on it here -- fine
        elif gt_zero and not sub_zero:
            issue = "extra_variable"
        elif sub_zero and not gt_zero:
            issue = "missing_variable"
        elif (e_sub < 0) != (e_gt < 0):
            issue = "sign_flip"
        elif abs(e_sub - e_gt) > ELASTICITY_MATCH_TOL:
            issue = "wrong_exponent"
        else:
            issue = "matches"

        details[str(p)] = {"sub": e_sub, "gt": e_gt, "issue": issue}

    issues = [d["issue"] for d in details.values()]
    if "missing_variable" in issues:
        mistake_type = "missing_variable"
    elif "sign_flip" in issues:
        mistake_type = "sign_flip"
    elif "extra_variable" in issues:
        mistake_type = "extra_variable"
    elif "wrong_exponent" in issues:
        mistake_type = "wrong_exponent"
    else:
        mistake_type = "other_structural"

    return {"mistake_type": mistake_type, "details": details}


if __name__ == "__main__":
    def check(name, submitted, ground_truth, expected_type):
        result = classify_mismatch(submitted, ground_truth)
        status = "OK" if result["mistake_type"] == expected_type else f"FAIL (expected {expected_type})"
        print(f"[{status}] {name}: {result['mistake_type']}")
        for param, d in result["details"].items():
            print(f"    {param}: sub={d['sub']}, gt={d['gt']}, issue={d['issue']}")
        assert result["mistake_type"] == expected_type

    check(
        "sign flip on distance (the real qwen38/qwq gravity pattern: -2 vs +2)",
        "def discovered_law(mass1, mass2, distance):\n    C = 1.0\n    return C * mass1 * mass2 / (distance ** 2)",
        "HIDDEN_CONSTANT * mass1 * mass2 * distance ** 2",
        "sign_flip",
    )
    check(
        "missing variable (submitted drops mass2 entirely)",
        "def discovered_law(mass1, mass2, distance):\n    C = 1.0\n    return C * mass1 / (distance ** 2)",
        "HIDDEN_CONSTANT * mass1 * mass2 / distance ** 2",
        "missing_variable",
    )
    check(
        "extra variable (submitted uses gamma, ground truth doesn't)",
        "def discovered_law(gamma, T, M):\n    C = 1.0\n    return C * gamma * T / M",
        "HIDDEN_CONSTANT * T / M",
        "extra_variable",
    )
    check(
        "wrong exponent, same sign (mass1 squared vs linear)",
        "def discovered_law(mass1, distance):\n    C = 1.0\n    return C * mass1 / distance ** 2",
        "HIDDEN_CONSTANT * mass1 ** 2 / distance ** 2",
        "wrong_exponent",
    )
    check(
        "correct exponents throughout (should be constant_equivalent-adjacent, not a mismatch) -- sanity check",
        "def discovered_law(mass1, mass2, distance):\n    C = 6.674e-5\n    return C * mass1 * mass2 / distance ** 2",
        "HIDDEN_CONSTANT * mass1 * mass2 / distance ** 2",
        "other_structural",  # no per-parameter issue found -- correctly falls through (this pair IS constant_equivalent; classify_mismatch is only meant to be called on confirmed mismatches)
    )

    print("\nAll sanity checks passed.")