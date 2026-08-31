"""
Classify HOW a submitted law is wrong, not just THAT it's wrong.

For each of the function's parameters, determine whether the expression
behaves as a genuine local power law in that parameter, and if so, what the
exponent is. The key quantity is elasticity_p = p * d(log(expr))/dp -- for a
true power law p^k (anywhere in the expression, even multiplied/divided by
other factors), this simplifies to the constant k, independent of p itself.
That "is elasticity free of p" check is done SYMBOLICALLY (sympy simplify),
not by sampling points and comparing -- point-sampling is not reliable here:
nested exp/log/trig terms (very common in this benchmark -- radioactive decay,
Bose-Einstein, Malus's law) can have elasticities whose value depends heavily
on the point but whose *relative* change between two arbitrary sample points
still happens to look "stable" purely by coincidence for some parameter
combinations. Checking symbolically whether p remains a free variable in the
simplified elasticity expression avoids that failure mode entirely: if p is
still present, the "exponent" isn't a single number at all, and reporting one
would be misleading (e.g. exp(-x^2.7) has elasticity -2.7*x^2.7 in x -- a real,
confirmed dependence, but not comparable to another expression's power-law
exponent as if it were an apples-to-apples number).

Comparing submitted's elasticity to ground truth's, per parameter, gives a
specific, human-readable diagnosis instead of just "wrong":
  - missing_variable: ground truth depends on this parameter, submitted doesn't
  - extra_variable: submitted depends on a parameter ground truth doesn't
  - sign_flip: both have a genuine (parameter-free) exponent, opposite sign
    (e.g. ground truth ~ 1/distance^2, submitted ~ distance^2)
  - wrong_exponent: both have a genuine exponent, same sign, different magnitude
  - nonlinear_dependence: both depend on it (nonzero derivative), but at least
    one side's elasticity isn't parameter-free (exp/log/trig-wrapped or an
    additive combination) -- confirmed to matter, magnitude not meaningfully
    comparable this way; doesn't drive the mistake_type verdict.
  - other_structural: no single-parameter elasticity mismatch found, but
    structural_verdict still says different -- likely an operator-level
    difference (e.g. additive vs multiplicative combination) that a per-
    parameter power-law lens can't isolate; needs a human look.

This reuses submitted_law_to_sympy / ground_truth_to_sympy from
structural_equivalence.py, so it inherits the same "fails closed to
not_checkable" behavior for control-flow functions.
"""
import cmath
import sys
from pathlib import Path

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from structural_equivalence import submitted_law_to_sympy, ground_truth_to_sympy, NotCheckable  # noqa: E402

ELASTICITY_ZERO_TOL = 1e-6
ELASTICITY_MATCH_TOL = 0.15  # exponents within this are treated as "the same"


def _reference_point(param_names):
    """Distinct, non-degenerate positive values per parameter -- avoids
    coincidental cancellations a shared value like 1.0 for everything could
    cause (e.g. (a - b) vanishing when a=b=1). Only used for (a) the initial
    nonzero-derivative check and (b) evaluating a CONFIRMED-stable elasticity
    to a number -- stability itself is decided symbolically, not by this point."""
    return {name: 1.7 + 0.6 * i for i, name in enumerate(sorted(param_names))}


def _param_diagnosis(expr, param, subs):
    """Returns (elasticity_value_or_None, status) where status in
    {'zero', 'stable', 'nonlinear'}."""
    try:
        deriv_val = complex(sp.diff(expr, param).subs(subs))
        val = complex(expr.subs(subs))
        if cmath.isnan(deriv_val) or cmath.isnan(val) or abs(val) < 1e-300:
            return None, "nonlinear"
        e_at_point = (subs[param] * deriv_val / val).real
    except Exception:
        return None, "nonlinear"

    if abs(e_at_point) < ELASTICITY_ZERO_TOL:
        return 0.0, "zero"

    # Exact symbolic stability check: elasticity = p * d(log(expr))/dp.
    # A genuine power law in p simplifies this to something with no p left in it.
    try:
        elasticity_expr = sp.simplify(param * sp.diff(sp.log(expr), param))
    except Exception:
        return None, "nonlinear"

    if param in elasticity_expr.free_symbols:
        return e_at_point, "nonlinear"  # confirmed nonzero dependence, but not a stable exponent

    try:
        remaining_subs = {k: v for k, v in subs.items() if k != param}
        stable_val = complex(elasticity_expr.subs(remaining_subs))
        if cmath.isnan(stable_val) or cmath.isinf(stable_val):
            return e_at_point, "nonlinear"
        return stable_val.real, "stable"
    except Exception:
        return e_at_point, "nonlinear"


def classify_mismatch(submitted_law_src: str, ground_truth_str: str):
    """Returns a dict:
        {'mistake_type': str, 'details': {param_name: {'sub': float|None, 'gt': float|None, 'issue': str}}}
    mistake_type is one of: missing_variable, extra_variable, sign_flip,
    wrong_exponent, nonlinear_dependence, other_structural, not_checkable.
    Priority when multiple parameters have issues: missing_variable > sign_flip
    > extra_variable > wrong_exponent > nonlinear_dependence > other_structural.
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
        e_sub, status_sub = _param_diagnosis(sub_expr, p, subs)
        e_gt, status_gt = _param_diagnosis(gt_expr, p, subs)

        sub_zero = status_sub == "zero"
        gt_zero = status_gt == "zero"

        if gt_zero and sub_zero:
            issue = "matches"  # neither depends on it here -- fine
        elif gt_zero and not sub_zero:
            issue = "extra_variable"
        elif sub_zero and not gt_zero:
            issue = "missing_variable"
        elif status_sub != "stable" or status_gt != "stable":
            # confirmed both depend on it (nonzero), but at least one side's
            # elasticity isn't a stable power-law exponent -- don't trust a
            # magnitude/sign comparison, just note it's a real (nonlinear) difference
            issue = "nonlinear_dependence"
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
    elif "nonlinear_dependence" in issues:
        mistake_type = "nonlinear_dependence"
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
    check(
        "real m5_radioactive_decay case: nested exp/power dependence -- elasticity is huge and "
        "point-dependent (confirmed mathematically correct earlier), must NOT be reported as a "
        "specific sign_flip/wrong_exponent magnitude for the nonlinear parameters",
        "def discovered_law(N0, lambda_constant, t):\n    import math\n    return N0 * math.exp(- (lambda_constant * t)**2)",
        "N0 * np.exp(-lambda_constant ** 2.71828 * t ** 1.5)",  # N0 exponent now matches (both ^1) --
        # isolates lambda_constant/t as the only real issues, both genuinely nonlinear
        "nonlinear_dependence",
    )

    print("\nAll sanity checks passed.")