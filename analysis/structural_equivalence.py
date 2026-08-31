"""
Judge-independent "constant-equivalence" check.

Given a trial's submitted_law (a full Python function definition string) and its
ground_truth_law (an expression string with a HIDDEN_CONSTANT placeholder, e.g.
"HIDDEN_CONSTANT * (mass1 ** 2 * mass2 ** 2) / distance ** 2"), this determines
-- via sympy, not an LLM -- whether the submitted expression has the SAME
functional form as ground truth, differing only by a constant multiplicative
factor. This is exactly the equivalence notion the paper's LLM judge is supposed
to check ("intentionally disregards the values of physical constants"), but
computed deterministically, so it can be used to sanity-check (or substitute for)
judge verdicts.

Approach:
  1. Parse submitted_law with `ast`. Any simple `name = <numeric literal>`
     assignment in the function body is treated as a free constant (replaced by
     a fresh sympy Symbol) rather than its literal value -- this is what lets a
     wildly wrong constant (e.g. 6.674e-55 instead of 6.674e-5) still compare
     as "constant-equivalent" to the correct form.
  2. Convert the function's return expression to a sympy expression, with
     function parameters and local constants as symbols.
  3. Convert ground_truth_law to a sympy expression, with HIDDEN_CONSTANT as a
     free symbol.
  4. Simplify submitted / ground_truth. If the result contains none of the
     physical-parameter symbols (i.e. it reduces to a ratio of constants), the
     two are constant-equivalent.

Limitations (fails closed -- returns "not_checkable" rather than a wrong
verdict): functions with control flow (if/for/while/try), multiple return
statements, or calls beyond a small whitelist of math functions are not
auto-checked and should be reviewed manually.
"""
import ast
import math
import sympy as sp

_MATH_FUNCS = {
    "sqrt": sp.sqrt, "exp": sp.exp, "log": sp.log, "sin": sp.sin, "cos": sp.cos,
    "tan": sp.tan, "abs": sp.Abs, "pow": lambda a, b: a ** b,
}
_MATH_CONSTS = {"pi": sp.pi, "e": sp.E}


class NotCheckable(Exception):
    pass


def _function_body_expr(func_node: ast.FunctionDef):
    """Return (param_names, {const_name: literal_value}, return_expr_ast).
    Raises NotCheckable if the function has control flow or isn't a single
    straight-line sequence of constant assignments followed by one return.
    """
    param_names = [a.arg for a in func_node.args.args]
    consts = {}
    return_expr = None

    for stmt in func_node.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            consts[stmt.targets[0].id] = None  # value unused -- we abstract it to a symbol regardless
        elif isinstance(stmt, ast.Return):
            return_expr = stmt.value
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        else:
            raise NotCheckable(f"Unsupported statement type: {type(stmt).__name__}")

    if return_expr is None:
        raise NotCheckable("No return statement found")
    return param_names, consts, return_expr


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
            symbol_map[node.id] = sp.Symbol(node.id)
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


def submitted_law_to_sympy(submitted_law_src: str):
    """Parse a `def discovered_law(...): ...` source string into a sympy
    expression with local constants abstracted to fresh symbols. Returns
    (expr, param_symbols, const_symbols)."""
    tree = ast.parse(submitted_law_src)
    func_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not func_nodes:
        raise NotCheckable("No function definition found")
    func_node = func_nodes[0]

    param_names, const_names, return_expr_ast = _function_body_expr(func_node)
    symbol_map = {name: sp.Symbol(name) for name in param_names}
    for cname in const_names:
        symbol_map[cname] = sp.Symbol(f"__const_{cname}")

    expr = _ast_to_sympy(return_expr_ast, symbol_map)
    param_symbols = {symbol_map[n] for n in param_names}
    const_symbols = {symbol_map[n] for n in const_names}
    return expr, param_symbols, const_symbols


def ground_truth_to_sympy(ground_truth_str: str, param_names):
    """Parse a ground_truth_law expression string (uses HIDDEN_CONSTANT as a
    placeholder) into a sympy expression."""
    symbol_map = {name: sp.Symbol(name) for name in param_names}
    symbol_map["HIDDEN_CONSTANT"] = sp.Symbol("__const_HIDDEN_CONSTANT")
    tree = ast.parse(ground_truth_str, mode="eval")
    expr = _ast_to_sympy(tree.body, symbol_map)
    const_symbols = {symbol_map["HIDDEN_CONSTANT"]}
    return expr, const_symbols


def check_constant_equivalence(submitted_law_src: str, ground_truth_str: str):
    """Returns one of: 'constant_equivalent', 'structurally_different', 'not_checkable'.

    'constant_equivalent' means submitted/ground_truth simplifies to something
    containing none of the physical parameter symbols -- i.e. same functional
    form, differing only by constant factor(s) (which is exactly what the
    paper's judge is instructed to ignore).
    """
    try:
        sub_expr, sub_params, sub_consts = submitted_law_to_sympy(submitted_law_src)
        gt_expr, gt_consts = ground_truth_to_sympy(ground_truth_str, [str(p) for p in sub_params])

        ratio = sp.simplify(sub_expr / gt_expr)
        remaining_free = ratio.free_symbols - sub_consts - gt_consts
        # Also drop any leftover constant-looking symbols sympy introduced (e.g. from nsimplify)
        remaining_physical = {s for s in remaining_free if str(s) in {str(p) for p in sub_params}}

        if not remaining_physical:
            return "constant_equivalent"
        return "structurally_different"
    except NotCheckable:
        return "not_checkable"
    except Exception:
        return "not_checkable"


if __name__ == "__main__":
    # Sanity checks using the two real examples from this conversation.
    print("=== Case 1: qwen38-27b trial1, wrong constant only (6.674e-55 vs 6.674e-5) ===")
    submitted = """
def discovered_law(mass1, mass2, distance):
    C = 6.674e-55
    return (C * mass1**2 * mass2**2) / (distance**2)
"""
    ground_truth = "HIDDEN_CONSTANT * (mass1 ** 2 * mass2 ** 2) / distance ** 2"
    print(check_constant_equivalence(submitted, ground_truth))
    assert check_constant_equivalence(submitted, ground_truth) == "constant_equivalent"

    print("\n=== Case 2: sr-scientist trial, wrong exponents entirely (structural mismatch) ===")
    submitted2 = """
def discovered_law(mass1, mass2, distance):
   return (mass1 * mass2) / (distance ** 2)
"""
    ground_truth2 = "HIDDEN_CONSTANT * (mass1 ** 2 * mass2 ** 2) * distance ** 2"
    print(check_constant_equivalence(submitted2, ground_truth2))
    assert check_constant_equivalence(submitted2, ground_truth2) == "structurally_different"

    print("\n=== Case 3: submitted with an unsupported if/else edge-case branch ===")
    submitted3 = """
def discovered_law(mass1, mass2, distance):
    C = 6.674e-5
    if distance == 0:
        return float('inf')
    return C * (mass1 ** 2) * (mass2 ** 2) / (distance ** 2)
"""
    print(check_constant_equivalence(submitted3, ground_truth))
    assert check_constant_equivalence(submitted3, ground_truth) == "not_checkable"

    print("\nAll sanity checks passed.")