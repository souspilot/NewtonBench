"""
Bayesian symbolic regression baseline agent for NewtonBench.

Non-LLM baseline that uses PySR (evolutionary symbolic regression) with
Latin Hypercube exploration and ensemble-disagreement active learning.
Operates under the same budget and experiment limits as the LLM agents.

Requires: pysr, scipy, numpy, sympy (pysr auto-installs Julia on first run).
"""

import inspect
import math
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats.qmc import LatinHypercube

from utils.budget import BudgetTracker
from utils.limits import ExperimentLimits

# ---------------------------------------------------------------------------
# Per-module parameter configuration
# ---------------------------------------------------------------------------
# law_params: parameter names in FUNCTION_SIGNATURE order
# ranges: (low, high) for log-uniform sampling (or linear for angles)
# exp_kwargs: mapping from law param name -> experiment kwarg name
#             (omit or set to None if they match)
# log_sample: set to False for params that should be sampled linearly
# ---------------------------------------------------------------------------

_MODULE_CONFIG: Dict[str, Dict[str, Any]] = {
    "m0_gravity": {
        "law_params": ["mass1", "mass2", "distance"],
        "ranges": {"mass1": (1, 1e3), "mass2": (1, 1e3), "distance": (1, 10)},
    },
    "m1_coulomb_force": {
        "law_params": ["q1", "q2", "distance"],
        "ranges": {"q1": (0.1, 10), "q2": (0.1, 10), "distance": (0.1, 10)},
    },
    "m2_magnetic_force": {
        "law_params": ["current1", "current2", "distance"],
        "ranges": {"current1": (1e-3, 1e-1), "current2": (1e-3, 1e-1), "distance": (1e-3, 1e-1)},
    },
    "m3_fourier_law": {
        "law_params": ["k", "A", "delta_T", "d"],
        "ranges": {"k": (0.1, 100), "A": (0.1, 100), "delta_T": (0.1, 100), "d": (0.1, 100)},
    },
    "m4_snell_law": {
        "law_params": ["n1", "n2", "angle1"],
        "ranges": {"n1": (1.0, 1.5), "n2": (1.0, 1.5), "angle1": (5.0, 85.0)},
        "log_sample": {"n1": False, "n2": False, "angle1": False},
        "exp_kwargs": {"n1": "refractive_index_1", "n2": "refractive_index_2", "angle1": "incidence_angle"},
    },
    "m5_radioactive_decay": {
        "law_params": ["N0", "lambda_constant", "t"],
        "ranges": {"N0": (1, 100), "lambda_constant": (1e-3, 0.1), "t": (0.01, 10)},
    },
    "m6_underdamped_harmonic": {
        "law_params": ["k", "m", "b"],
        "ranges": {"k": (1e2, 1e4), "m": (0.1, 10), "b": (0.01, 1)},
    },
    "m7_malus_law": {
        "law_params": ["I_0", "theta"],
        "ranges": {"I_0": (0.1, 100), "theta": (0.05, 1.5)},
        "log_sample": {"theta": False},
    },
    "m8_sound_speed": {
        "law_params": ["gamma", "T", "M"],
        "ranges": {"gamma": (1.3, 1.7), "T": (10, 1000), "M": (1e-3, 0.1)},
        "log_sample": {"gamma": False},
        "exp_kwargs": {"gamma": "adiabatic_index", "T": "temperature", "M": "molar_mass"},
    },
    "m9_hooke_law": {
        "law_params": ["x"],
        "ranges": {"x": (0.01, 10)},
    },
    "m10_be_distribution": {
        "law_params": ["omega", "T"],
        "ranges": {"omega": (1e8, 1e10), "T": (10, 1000)},
        "exp_kwargs": {"T": "temperature"},
    },
    "m11_heat_transfer": {
        "law_params": ["m", "c", "delta_T"],
        "ranges": {"m": (0.1, 100), "c": (0.1, 100), "delta_T": (0.1, 100)},
    },
}

# Fallback ranges when a module is not in the config
_DEFAULT_RANGE = (0.1, 100.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_law_params(function_signature: str) -> List[str]:
    """Extract parameter names from a FUNCTION_SIGNATURE string."""
    match = re.search(r"def discovered_law\(([^)]*)\)", function_signature)
    if not match:
        return []
    return [p.strip() for p in match.group(1).split(",") if p.strip()]


def _get_module_name(module: Any) -> str:
    name = module.__name__
    if "." in name:
        name = name.split(".")[-1]
    return name


def _get_config(module: Any) -> Dict[str, Any]:
    mod_name = _get_module_name(module)
    if mod_name in _MODULE_CONFIG:
        return _MODULE_CONFIG[mod_name]
    law_params = _parse_law_params(getattr(module, "FUNCTION_SIGNATURE", ""))
    return {
        "law_params": law_params,
        "ranges": {p: _DEFAULT_RANGE for p in law_params},
    }


def _sample_points(config: Dict[str, Any], n: int, rng: np.random.Generator) -> List[Dict[str, float]]:
    """Generate n experiment dicts via Latin Hypercube Sampling."""
    law_params = config["law_params"]
    ranges = config.get("ranges", {})
    log_sample = config.get("log_sample", {})
    exp_kwargs = config.get("exp_kwargs", {})
    d = len(law_params)
    if d == 0:
        return []

    sampler = LatinHypercube(d=d, seed=rng)
    unit_samples = sampler.random(n=n)

    experiments = []
    for row in unit_samples:
        exp: Dict[str, float] = {}
        for i, param in enumerate(law_params):
            lo, hi = ranges.get(param, _DEFAULT_RANGE)
            use_log = log_sample.get(param, True) if lo > 0 else False
            if use_log and lo > 0 and hi > lo:
                val = float(np.exp(np.log(lo) + row[i] * (np.log(hi) - np.log(lo))))
            else:
                val = float(lo + row[i] * (hi - lo))
            kwarg_name = exp_kwargs.get(param, param)
            exp[kwarg_name] = val
        experiments.append(exp)
    return experiments


def _extract_scalar(result: Any) -> Optional[float]:
    """Best-effort extraction of a scalar measurement from any system output."""
    if isinstance(result, (int, float)):
        v = float(result)
        return v if math.isfinite(v) else None

    if isinstance(result, str):
        try:
            v = float(result)
            return v if math.isfinite(v) else None
        except (ValueError, TypeError):
            return None

    if not isinstance(result, dict):
        return None

    # Skip 'time' key; look for a single-valued measurement
    non_time_keys = [k for k in result if k != "time"]
    if not non_time_keys:
        return None

    for key in non_time_keys:
        val = result[key]
        # Single scalar string
        if isinstance(val, str):
            if val.lower() == "invalid":
                return None
            try:
                v = float(val)
                return v if math.isfinite(v) else None
            except (ValueError, TypeError):
                continue

        # Single scalar number
        if isinstance(val, (int, float)):
            v = float(val)
            return v if math.isfinite(v) else None

        # List — try to extract from time series
        if isinstance(val, list) and len(val) >= 2:
            try:
                # For velocity lists: compute initial acceleration
                # (heuristic: first derivative of the first non-time series)
                nums = [float(x) if isinstance(x, str) else float(x) for x in val
                        if not isinstance(x, list)]
                if len(nums) >= 2:
                    # Use the time column if available for differentiation
                    time_vals = result.get("time")
                    if time_vals and len(time_vals) >= 2:
                        t = [float(x) if isinstance(x, str) else float(x) for x in time_vals]
                        dt = t[1] - t[0]
                        if dt > 0 and key == "velocity":
                            accel = (nums[1] - nums[0]) / dt
                            return accel if math.isfinite(accel) else None
                    # Fallback: return last value
                    v = nums[-1]
                    return v if math.isfinite(v) else None
            except (ValueError, TypeError, IndexError):
                continue
    return None


def _run_batch(
    module: Any,
    experiments: List[Dict[str, float]],
    config: Dict[str, Any],
    system: str,
    noise_level: float,
    difficulty: str,
    law_version: Optional[str],
    budget: Optional[Any],
    budget_tracker: Optional[BudgetTracker],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run a batch of experiments, return (X, y, cost).

    X has shape (n_valid, n_params), y has shape (n_valid,).
    Only rows with finite scalar outputs are included.
    """
    law_params = config["law_params"]
    exp_kwargs = config.get("exp_kwargs", {})
    cost = 0.0

    if budget is not None and budget_tracker is not None:
        pricing = budget.price_batch(experiments)
        if budget_tracker.would_exceed(pricing.total_cost):
            return np.empty((0, len(law_params))), np.empty(0), 0.0
        budget_tracker.charge(pricing.total_cost, {
            "num_points": len(experiments),
            "precisions": pricing.precisions,
        })
        cost = pricing.total_cost
        measurements = []
        for exp, precision in zip(pricing.cleaned_experiments, pricing.precisions):
            m = budget.measure(module, exp, precision, noise_level=noise_level,
                               difficulty=difficulty, system=system, law_version=law_version)
            measurements.append(m)
    else:
        measurements = []
        for exp in experiments:
            m = module.run_experiment_for_module(
                **exp, noise_level=noise_level, difficulty=difficulty,
                system=system, law_version=law_version,
            )
            measurements.append(m)

    X_rows, y_rows = [], []
    for exp, meas in zip(experiments, measurements):
        scalar = _extract_scalar(meas)
        if scalar is None:
            continue
        row = []
        for param in law_params:
            kwarg_name = exp_kwargs.get(param, param)
            row.append(exp.get(kwarg_name, 0.0))
        X_rows.append(row)
        y_rows.append(scalar)

    if not X_rows:
        return np.empty((0, len(law_params))), np.empty(0), cost
    return np.array(X_rows), np.array(y_rows), cost


def _fit_pysr(X: np.ndarray, y: np.ndarray, param_names: List[str],
              niterations: int = 40, timeout: int = 120) -> Any:
    """Fit PySR on accumulated data. Returns the fitted model or None."""
    try:
        from pysr import PySRRegressor
    except ImportError:
        print("[Bayesian Agent] PySR not installed — run `pip install pysr`")
        return None

    if X.shape[0] < 3:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PySRRegressor(
            niterations=niterations,
            binary_operators=["+", "-", "*", "/", "pow"],
            unary_operators=["exp", "sqrt", "sin", "cos", "log", "abs"],
            populations=20,
            maxsize=25,
            parsimony=0.003,
            timeout_in_seconds=timeout,
            temp_equation_file=True,
            verbosity=0,
            progress=False,
            variable_names=param_names,
        )
        try:
            model.fit(X, y)
        except Exception as e:
            print(f"[Bayesian Agent] PySR fit failed: {e}")
            return None
    return model


def _active_select(model: Any, config: Dict[str, Any], n_candidates: int,
                   batch_size: int, rng: np.random.Generator) -> List[Dict[str, float]]:
    """Select experiments where the Pareto front disagrees most."""
    law_params = config["law_params"]
    exp_kwargs = config.get("exp_kwargs", {})

    candidates = _sample_points(config, n_candidates, rng)
    if not candidates:
        return []

    X_cand = np.array([
        [exp.get(exp_kwargs.get(p, p), 0.0) for p in law_params]
        for exp in candidates
    ])

    try:
        eqs = model.equations_
        if eqs is None or len(eqs) < 2:
            return candidates[:batch_size]

        preds = []
        for _, row in eqs.iterrows():
            try:
                lam = row["lambda_format"]
                pred = np.array([lam(X_cand[i]) for i in range(len(X_cand))])
                preds.append(pred)
            except Exception:
                continue

        if len(preds) < 2:
            return candidates[:batch_size]

        pred_array = np.array(preds)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            variance = np.nanvar(pred_array, axis=0)
        variance = np.nan_to_num(variance, nan=0.0)

        top_idx = np.argsort(variance)[-batch_size:]
        return [candidates[i] for i in top_idx]

    except Exception:
        return candidates[:batch_size]


def _model_to_law_string(model: Any, law_params: List[str],
                         function_signature: str) -> str:
    """Convert PySR's best equation to a discovered_law function string."""
    if model is None:
        return f"{function_signature} return float('nan')"

    try:
        best = model.get_best()
        sympy_expr = best.sympy_format

        import sympy
        expr_str = sympy.pycode(sympy_expr)

        # sympy.pycode uses math. prefix — ensure import is available
        # Build a standalone function
        lines = [
            function_signature,
            "    import math",
            f"    return {expr_str}",
        ]
        return "\n".join(lines)

    except Exception as e:
        print(f"[Bayesian Agent] Failed to convert PySR output: {e}")
        return f"{function_signature} return float('nan')"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def conduct_bayesian_exploration(
    module: Any,
    model_name: str,
    noise_level: float,
    difficulty: str,
    system: str,
    law_version: str = None,
    trial_info: Dict[str, Any] = None,
    budget: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Non-LLM baseline: Latin Hypercube exploration → PySR symbolic regression
    → ensemble-disagreement active learning → final SR fit.
    """
    trial_info = trial_info or {}
    trial_id = trial_info.get("trial_id", "?")
    limits = ExperimentLimits.load()
    config = _get_config(module)
    law_params = config["law_params"]
    function_signature = getattr(module, "FUNCTION_SIGNATURE",
                                 f"def discovered_law({', '.join(law_params)}):")

    budget_tracker = BudgetTracker(budget) if budget is not None else None
    rng = np.random.default_rng()

    total_budget = budget.starting_funds if budget is not None else float("inf")
    BATCH_SIZE = 5  # stay in cheapest pricing tier
    MAX_DATAPOINTS = limits.max_datapoints_per_trial

    X_all = np.empty((0, len(law_params)))
    y_all = np.empty(0)
    datapoints_used = 0
    experiment_log: List[Dict[str, Any]] = []
    rounds = 0

    def remaining_budget() -> float:
        if budget_tracker is None:
            return float("inf")
        return budget_tracker.remaining

    def can_experiment() -> bool:
        return datapoints_used < MAX_DATAPOINTS and remaining_budget() > 0

    print(f"[Bayesian Agent Trial {trial_id}] Starting — {len(law_params)} params, "
          f"system={system}, difficulty={difficulty}")

    # ---- Phase 1: Space-filling exploration (≤40% budget) ----
    phase1_limit = total_budget * 0.4
    phase1_spent = 0.0
    n_phase1 = min(15, MAX_DATAPOINTS)

    experiments = _sample_points(config, n_phase1, rng)
    for batch_start in range(0, len(experiments), BATCH_SIZE):
        if not can_experiment():
            break
        batch = experiments[batch_start:batch_start + BATCH_SIZE]
        remaining = MAX_DATAPOINTS - datapoints_used
        if remaining < len(batch):
            batch = batch[:remaining]

        X_new, y_new, cost = _run_batch(
            module, batch, config, system, noise_level, difficulty,
            law_version, budget, budget_tracker,
        )
        if X_new.shape[0] > 0:
            X_all = np.vstack([X_all, X_new])
            y_all = np.concatenate([y_all, y_new])
            datapoints_used += X_new.shape[0]
            phase1_spent += cost
            rounds += 1
            experiment_log.append({
                "phase": "exploration",
                "round": rounds,
                "points_requested": len(batch),
                "points_collected": X_new.shape[0],
                "cost": cost,
            })
        if phase1_spent >= phase1_limit:
            break

    print(f"[Bayesian Agent Trial {trial_id}] Phase 1 done: {X_all.shape[0]} pts, "
          f"${phase1_spent:.0f} spent")

    # ---- Phase 2: Active refinement (≤40% budget) ----
    phase2_limit = total_budget * 0.4
    phase2_spent = 0.0
    max_active_rounds = 3

    if can_experiment() and X_all.shape[0] >= 5:
        sr_model = _fit_pysr(X_all, y_all, law_params, niterations=30, timeout=90)

        for active_round in range(max_active_rounds):
            if not can_experiment() or phase2_spent >= phase2_limit:
                break
            if sr_model is None:
                break

            remaining = min(BATCH_SIZE, MAX_DATAPOINTS - datapoints_used)
            if remaining <= 0:
                break

            batch = _active_select(sr_model, config, n_candidates=200,
                                   batch_size=remaining, rng=rng)
            if not batch:
                break

            X_new, y_new, cost = _run_batch(
                module, batch, config, system, noise_level, difficulty,
                law_version, budget, budget_tracker,
            )
            if X_new.shape[0] > 0:
                X_all = np.vstack([X_all, X_new])
                y_all = np.concatenate([y_all, y_new])
                datapoints_used += X_new.shape[0]
                phase2_spent += cost
                rounds += 1
                experiment_log.append({
                    "phase": "active",
                    "round": rounds,
                    "points_requested": len(batch),
                    "points_collected": X_new.shape[0],
                    "cost": cost,
                })
                # Refit after new data
                sr_model = _fit_pysr(X_all, y_all, law_params,
                                     niterations=30, timeout=90)

        print(f"[Bayesian Agent Trial {trial_id}] Phase 2 done: "
              f"{X_all.shape[0]} total pts, ${phase2_spent:.0f} spent")

    # ---- Phase 3: Final fit ----
    print(f"[Bayesian Agent Trial {trial_id}] Phase 3: final SR fit on "
          f"{X_all.shape[0]} points")

    final_model = None
    if X_all.shape[0] >= 3:
        final_model = _fit_pysr(X_all, y_all, law_params,
                                niterations=80, timeout=180)

    submitted_law = _model_to_law_string(final_model, law_params, function_signature)
    print(f"[Bayesian Agent Trial {trial_id}] Submitted: "
          f"{submitted_law[:120]}...")

    return {
        "status": "completed",
        "submitted_law": submitted_law,
        "rounds": rounds,
        "total_tokens": 0,
        "num_experiments": datapoints_used,
        "chat_history": experiment_log,
        "exploration_mode": "bayesian_agent",
        "budget": budget_tracker.summary() if budget_tracker is not None else None,
    }
