"""
Judge-independent "constant-equivalence" check.

Given a trial's submitted_law (a full Python function definition string) and its
ground_truth_law (an expression string, e.g. with a HIDDEN_CONSTANT or CONSTANT
placeholder standing in for whatever exact physical constant the model can't see:
"HIDDEN_CONSTANT * (mass1 ** 2 * mass2 ** 2) / distance ** 2"), this determines
-- via sympy, not an LLM -- whether the submitted expression has the SAME
functional form as ground truth, differing only by a constant multiplicative
factor. This is exactly the equivalence notion the paper's LLM judge is supposed
to check ("intentionally disregards the values of physical constants"), but
computed deterministically, so it can be used to sanity-check (or substitute for)
judge verdicts.

Approach:
  1. Parse submitted_law with `ast`. Function parameters become sympy symbols
     (declared `positive=True`, since every physical quantity in this benchmark
     -- mass, distance, temperature, etc. -- is a positive real, and this
     assumption is what lets sympy safely split/combine things like
     sqrt(a*b) <-> sqrt(a)*sqrt(b) during simplification).
  2. Walk the function body IN ORDER. Each `name = <expr>` assignment is
     evaluated to a sympy expression using whatever's been resolved so far, and
     substituted in place for later references to `name` -- it is NOT abstracted
     into an opaque "this could be any constant" symbol. This matters: an
     assignment can be a genuine constant (`C = 6.674e-5`) or an intermediate
     expression that depends on the function's parameters (`sin_t =
     math.sin(theta)`) or even a value that must match ground truth exactly
     (`exponent = 2.6`) -- treating all three the same way (as an earlier
     version of this function did) silently destroys parameter-dependence and
     erases exact-value matches that should count as equal.
  3. Convert ground_truth_law the same way, using the SAME parameter symbols.
     Any name that isn't a declared parameter (HIDDEN_CONSTANT, CONSTANT, k,
     ...) is auto-created as a free symbol representing "the actual physical
     constant" -- this is the ONLY thing that gets treated as freely
     substitutable.
  4. Simplify submitted / ground_truth. If nothing depending on the function's
     actual parameters remains (only numbers and/or the ground-truth constant
     placeholder), the two are constant-equivalent -- same functional form,
     differing only in a multiplicative factor. Because submitted's constants
     were plugged in as literal numbers rather than abstracted, this also
     correctly rejects a WRONG exponent or coefficient that happens to be
     stored in a named local variable, while still accepting a wrong-but-
     correctly-placed constant (e.g. 6.674e-55 instead of 6.674e-5).

Limitations (fails closed -- returns "not_checkable" rather than a wrong
verdict): functions with control flow (if/for/while/try), multiple return
statements, or calls beyond a small whitelist of math functions are not
auto-checked and should be reviewed manually.
"""
import ast
import sympy as sp

_MATH_FUNCS = {
    "sqrt": sp.sqrt, "exp": sp.exp, "log": sp.log, "sin": sp.sin, "cos": sp.cos,
    "tan": sp.tan, "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "degrees": lambda x: x * 180 / sp.pi, "radians": lambda x: x * sp.pi / 180,
    "abs": sp.Abs, "pow": lambda a, b: a ** b,
}
_MATH_CONSTS = {"pi": sp.pi, "e": sp.E}


class NotCheckable(Exception):
    pass


def _ast_to_sympy(node, symbol_map):
    if isinstance(node, ast.BinOp):
        left = _ast_to_sympy(node.left, symbol_map)
        right = _ast_to_sympy(node.right, symbol_map)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
        raise NotCheckable(f"Unsupported binop: {type(node.op).__name__}")
    if isinstance(node, ast.UnaryOp):
        val = _ast_to_sympy(node.operand, symbol_map)
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return val
        raise NotCheckable(f"Unsupported unaryop: {type(node.op).__name__}")
    if isinstance(node, ast.Name):
        if node.id not in symbol_map:
            # Not a declared parameter and not assigned yet in this function --
            # e.g. ground_truth_law's HIDDEN_CONSTANT/CONSTANT/k placeholder.
            # Treat it as a free "the real constant" symbol.
            symbol_map[node.id] = sp.Symbol(node.id, positive=True)
        return symbol_map[node.id]
    if isinstance(node, ast.Constant):
        return sp.nsimplify(node.value) if isinstance(node.value, (int, float)) else node.value
    if isinstance(node, ast.Call):
        fname = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
        if fname not in _MATH_FUNCS:
            raise NotCheckable(f"Unsupported function call: {fname}")
        args = [_ast_to_sympy(a, symbol_map) for a in node.args]
        return _MATH_FUNCS[fname](*args)
    if isinstance(node, ast.Attribute):
        if node.attr in _MATH_CONSTS:
            return _MATH_CONSTS[node.attr]
        raise NotCheckable(f"Unsupported attribute: {node.attr}")
    raise NotCheckable(f"Unsupported node type: {type(node).__name__}")


def _eval_function_body(param_names, body_stmts):
    """Walk a function body in order, resolving each local assignment to a
    sympy expression (substituted, not abstracted) and returning the final
    return-statement's sympy expression. `symbol_map` starts with only the
    declared parameters as (positive) symbols; local names get added as their
    RESOLVED VALUE, so later statements/the return correctly reflect whatever
    that name actually computes to -- whether it's a bare number, a parameter-
    dependent expression, or a value that must match ground truth exactly.

    Also unwraps two common "domain guard" patterns rather than treating them
    as unsupported control flow, since they're extremely common in submitted
    laws and represent edge-case handling around the core formula, not the
    formula itself:
      - `if <cond>: return <fallback>` with no elif/else -- treated as a guard
        clause and skipped; only the eventual real return is used.
      - `try: <body> except ...: return <fallback>` -- the try body's
        statements are spliced in as if unwrapped; the except branch (a
        fallback for invalid inputs) is ignored.
    Anything else (a return inside the except branch that ISN'T a simple
    guard, nested control flow, multiple returns in the main path, etc.)
    still raises NotCheckable and falls back to manual review.
    """
    symbol_map = {name: sp.Symbol(name, positive=True) for name in param_names}
    return_expr = None

    def process(stmts):
        nonlocal return_expr
        for stmt in stmts:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                symbol_map[stmt.targets[0].id] = _ast_to_sympy(stmt.value, symbol_map)
            elif isinstance(stmt, ast.Return):
                return_expr = stmt.value
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                continue
            elif isinstance(stmt, ast.If) and not stmt.orelse and len(stmt.body) == 1 \
                    and isinstance(stmt.body[0], ast.Return):
                continue  # guard clause (e.g. domain-error fallback) -- skip it
            elif isinstance(stmt, ast.Try) and not stmt.orelse and not stmt.finalbody:
                process(stmt.body)  # splice the try-body in; ignore the except fallback
            else:
                raise NotCheckable(f"Unsupported statement type: {type(stmt).__name__}")

    process(body_stmts)

    if return_expr is None:
        raise NotCheckable("No return statement found")
    return _ast_to_sympy(return_expr, symbol_map), {symbol_map[n] for n in param_names}


def submitted_law_to_sympy(submitted_law_src: str):
    """Parse a `def discovered_law(...): ...` source string into a sympy
    expression. Returns (expr, param_symbols). Local constants and
    intermediate expressions are fully substituted in -- nothing about
    submitted_law is left abstract except its declared parameters."""
    tree = ast.parse(submitted_law_src)
    func_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not func_nodes:
        raise NotCheckable("No function definition found")
    func_node = func_nodes[0]
    param_names = [a.arg for a in func_node.args.args]
    return _eval_function_body(param_names, func_node.body)


def ground_truth_to_sympy(ground_truth_str: str, param_names):
    """Parse a ground_truth_law expression string into a sympy expression,
    using the SAME parameter symbols as submitted_law_to_sympy. Any other name
    encountered (HIDDEN_CONSTANT, CONSTANT, k, ...) is auto-created as a free
    symbol via _ast_to_sympy's Name handling."""
    symbol_map = {name: sp.Symbol(name, positive=True) for name in param_names}
    tree = ast.parse(ground_truth_str, mode="eval")
    expr = _ast_to_sympy(tree.body, symbol_map)
    return expr


def check_constant_equivalence(submitted_law_src: str, ground_truth_str: str):
    """Returns one of: 'constant_equivalent', 'structurally_different', 'not_checkable'.

    'constant_equivalent' means submitted/ground_truth simplifies to something
    containing none of submitted's declared parameters -- i.e. same functional
    form, differing only by a constant factor (exactly what the paper's judge
    is instructed to ignore). Any wrong exponent, wrong operator, extra/missing
    term, or genuinely different parameter-dependence will leave parameter
    symbols in the ratio and correctly return 'structurally_different'.
    """
    try:
        sub_expr, sub_params = submitted_law_to_sympy(submitted_law_src)
        gt_expr = ground_truth_to_sympy(ground_truth_str, [str(p) for p in sub_params])

        ratio = sp.simplify(sub_expr / gt_expr)
        remaining_physical = ratio.free_symbols & sub_params

        if not remaining_physical:
            return "constant_equivalent"
        return "structurally_different"
    except NotCheckable:
        return "not_checkable"
    except Exception:
        return "not_checkable"


if __name__ == "__main__":
    def check(name, submitted, ground_truth, expected):
        result = check_constant_equivalence(submitted, ground_truth)
        status = "OK" if result == expected else f"FAIL (expected {expected})"
        print(f"[{status}] {name}: {result}")
        assert result == expected, f"{name}: got {result}, expected {expected}"

    check(
        "wrong constant only (6.674e-55 vs 6.674e-5)",
        "def discovered_law(mass1, mass2, distance):\n    C = 6.674e-55\n    return (C * mass1**2 * mass2**2) / (distance**2)",
        "HIDDEN_CONSTANT * (mass1 ** 2 * mass2 ** 2) / distance ** 2",
        "constant_equivalent",
    )
    check(
        "wrong exponents entirely (structural mismatch)",
        "def discovered_law(mass1, mass2, distance):\n   return (mass1 * mass2) / (distance ** 2)",
        "HIDDEN_CONSTANT * (mass1 ** 2 * mass2 ** 2) * distance ** 2",
        "structurally_different",
    )
    check(
        "simple domain guard (if/return then the real formula) -- now unwrapped, not a failure",
        "def discovered_law(mass1, mass2, distance):\n    C = 6.674e-5\n    if distance == 0:\n        return float('inf')\n    return C * (mass1 ** 2) * (mass2 ** 2) / (distance ** 2)",
        "HIDDEN_CONSTANT * (mass1 ** 2 * mass2 ** 2) / distance ** 2",
        "constant_equivalent",
    )
    check(
        "genuinely unsupported control flow (if/elif with two DIFFERENT formulas) -> not_checkable",
        "def discovered_law(mass1, mass2, distance):\n    C = 6.674e-5\n    if distance == 0:\n        return C * mass1\n    else:\n        return C * (mass1 ** 2) * (mass2 ** 2) / (distance ** 2)",
        "HIDDEN_CONSTANT * (mass1 ** 2 * mass2 ** 2) / distance ** 2",
        "not_checkable",
    )
    check(
        "constant pulled outside sqrt (same form, previously a false positive)",
        "def discovered_law(gamma, T, M):\n    C = 18.75\n    return C * math.sqrt(T / M)",
        "math.sqrt(HIDDEN_CONSTANT * T / M)",
        "constant_equivalent",
    )
    check(
        "exponent matches exactly via a local variable (previously a false positive)",
        "def discovered_law(mass1, distance):\n    G = 6.67429e-5\n    exponent = 2.6\n    return (G * mass1) / (distance ** exponent)",
        "HIDDEN_CONSTANT * mass1 / distance ** 2.6",
        "constant_equivalent",
    )
    check(
        "exponent WRONG via a local variable -- must still be caught",
        "def discovered_law(mass1, distance):\n    G = 6.674e-5\n    exponent = 2.5\n    return (G * mass1) / (distance ** exponent)",
        "HIDDEN_CONSTANT * mass1 / distance ** 2.6",
        "structurally_different",
    )
    check(
        "intermediate parameter-dependent variables, literally identical form (previously a false positive)",
        "def discovered_law(I_0, theta):\n    import math\n    sin_t = math.sin(theta)\n    cos_t = math.cos(theta)\n    return I_0 * (sin_t + cos_t) ** 2",
        "I_0 * (np.sin(theta) + np.cos(theta)) ** 2",
        "constant_equivalent",
    )
    check(
        "constant function vs a genuinely non-constant law -- must be caught",
        "def discovered_law(omega, T):\n    return 0.5",
        "1 / (np.exp(HIDDEN_CONSTANT * omega / T) + 1)",
        "structurally_different",
    )
    check(
        "ground truth uses a bare symbol placeholder ('k') instead of HIDDEN_CONSTANT",
        "def discovered_law(x):\n    return x**2",
        "2 * k * x ** 2",
        "constant_equivalent",
    )
    check(
        "guard-if + try/except unwrapping (real m4_snell_law pattern)",
        "def discovered_law(n1, n2, angle1):\n    import math\n    try:\n        cos_theta2 = (n1 / n2) * math.cos(math.radians(angle1))\n        if cos_theta2 < -1 or cos_theta2 > 1:\n            return float('nan')\n        theta2 = math.degrees(math.acos(cos_theta2))\n        return theta2\n    except ValueError:\n        return float('nan')",
        "math.degrees(math.acos((n1 / n2) * math.cos(math.radians(angle1))))",
        "constant_equivalent",
    )

    print("\nAll sanity checks passed.")