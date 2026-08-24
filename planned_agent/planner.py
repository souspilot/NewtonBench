"""
Experiment Planner
==================
Parses a NewtonBench task prompt to extract controllable variables and
their scales, detects the system type, then generates controlled-variable
experiment plans.

Fully deterministic — no LLM calls.  Module-agnostic — works from
prompt text alone.
"""

import re
import json
import math
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VariableSpec:
    """A controllable input variable extracted from the prompt."""
    name: str
    min_scale: float = 1e-1
    max_scale: float = 1e3
    log_spaced: bool = True


@dataclass
class ExperimentPlan:
    """Full experiment plan produced by the planner."""
    system_type: str
    variables: List[VariableSpec] = field(default_factory=list)
    sweeps: List[Dict[str, Any]] = field(default_factory=list)
    noise_check: List[Dict[str, Any]] = field(default_factory=list)
    preprocessing_hint: str = ""
    total_experiments: int = 0


# ---------------------------------------------------------------------------
# Integer-typed parameters that must be cast from float
# ---------------------------------------------------------------------------

_INT_PARAMS = {"num_points"}


def sanitize_experiment(exp: Dict[str, Any]) -> Dict[str, Any]:
    """Cast known integer parameters to int."""
    return {k: int(v) if k in _INT_PARAMS else v for k, v in exp.items()}


# ---------------------------------------------------------------------------
# Prompt parser helpers
# ---------------------------------------------------------------------------

def _detect_system_type(prompt: str) -> str:
    lower = prompt.lower()
    if any(kw in lower for kw in [
        "orbital", "2d motion", "centripetal",
        "calorimeter", "band-pass filter", "bandwidth",
    ]):
        return "complex_system"
    if any(kw in lower for kw in [
        "1d motion", "linear motion", "track the position",
        "black-body", "spectrometer", "spectral_radiance",
    ]):
        return "simple_system"
    return "vanilla_equation"


def _extract_variables(prompt: str) -> List[VariableSpec]:
    exp_match = re.search(
        r'<run_experiment>\s*\[\s*\{(.*?)\}', prompt, re.DOTALL,
    )
    if not exp_match:
        return []

    keys_text = exp_match.group(1)
    key_names = re.findall(r'"(\w+)"\s*:', keys_text)
    variables = []

    for name in key_names:
        vs = VariableSpec(name=name)
        scale_pat = (
            rf'[`"\']?{re.escape(name)}[`"\']?'
            r'.*?(?:at least|scale.*?)\s*([\d.]+[eE][+-]?\d+)'
        )
        m = re.search(scale_pat, prompt, re.IGNORECASE | re.DOTALL)
        if m:
            base = float(m.group(1))
            vs.min_scale = base
            vs.max_scale = base * 1e4
        variables.append(vs)

    return variables


# ---------------------------------------------------------------------------
# Sweep generators
# ---------------------------------------------------------------------------

def _log_values(low: float, high: float, n: int = 10) -> List[float]:
    if low <= 0:
        low = 1e-3
    if high <= low:
        high = low * 1e4
    return np.logspace(np.log10(low), np.log10(high), n).tolist()


def _defaults_for(variables: List[VariableSpec]) -> Dict[str, float]:
    return {v.name: math.sqrt(v.min_scale * v.max_scale) for v in variables}


def _plan_vanilla(
    variables: List[VariableSpec], pts: int, noise_reps: int,
) -> ExperimentPlan:
    plan = ExperimentPlan(system_type="vanilla_equation", variables=variables)
    defaults = _defaults_for(variables)

    for _ in range(noise_reps):
        plan.noise_check.append(dict(defaults))

    for v in variables:
        for val in _log_values(v.min_scale, v.max_scale, pts):
            e = dict(defaults); e[v.name] = val
            plan.sweeps.append(e)

    if len(variables) >= 2:
        v0, v1 = variables[0], variables[1]
        for a, b in zip(
            _log_values(v0.min_scale, v0.max_scale, 5),
            _log_values(v1.min_scale, v1.max_scale, 5),
        ):
            e = dict(defaults); e[v0.name] = a; e[v1.name] = b
            plan.sweeps.append(e)

    plan.total_experiments = len(plan.sweeps) + len(plan.noise_check)
    plan.preprocessing_hint = (
        "This is a direct measurement. Each experiment returns the target "
        "quantity. Analyze how the output changes as each input is swept "
        "while others are held fixed."
    )
    return plan


def _plan_dynamics(
    variables: List[VariableSpec],
    system_type: str,
    prompt: str,
    pts: int,
) -> ExperimentPlan:
    plan = ExperimentPlan(system_type=system_type, variables=variables)
    lower = prompt.lower()

    is_trajectory = ("position" in lower and "velocity" in lower) or "time_step" in lower
    is_calorimeter = "calorimeter" in lower or "total_power" in lower
    is_spectrometer = "spectral_radiance" in lower and not is_calorimeter

    sim_param_names = {"duration", "time_step", "initial_velocity", "num_points"}
    physics_vars = [v for v in variables if v.name not in sim_param_names]
    defaults = _defaults_for(variables)

    # --- trajectory systems ---
    if is_trajectory:
        defaults["duration"] = 0.1
        defaults["time_step"] = 0.001
        defaults["initial_velocity"] = 0.0

        for v in physics_vars:
            for val in _log_values(v.min_scale, v.max_scale, pts):
                e = dict(defaults); e[v.name] = val
                plan.sweeps.append(e)

        if len(physics_vars) >= 2:
            v0, v1 = physics_vars[0], physics_vars[1]
            for a, b in zip(
                _log_values(v0.min_scale, v0.max_scale, 4),
                _log_values(v1.min_scale, v1.max_scale, 4),
            ):
                e = dict(defaults); e[v0.name] = a; e[v1.name] = b
                plan.sweeps.append(e)

        plan.preprocessing_hint = (
            "PREPROCESSING STRATEGY FOR TRAJECTORY DATA:\n"
            "Experiments use short duration & small time_step for clean acceleration.\n"
            "1. Compute acceleration: a = (v[1] - v[0]) / time_step\n"
            "   For 2D data: a = sqrt(ax² + ay²) using velocity component differences\n"
            "2. Compute force: F = mass2 * |a| (Newton's second law)\n"
            "3. The initial_velocity is 0, so the first acceleration gives the force\n"
            "   at the starting distance directly.\n"
            "4. You now have (mass1, mass2, distance) → F. Discover F(mass1, mass2, distance)."
        )

    # --- spectrometer systems ---
    elif is_spectrometer:
        for v in physics_vars:
            for val in _log_values(v.min_scale, v.max_scale, pts):
                e = dict(defaults); e[v.name] = val
                plan.sweeps.append(e)

        if len(physics_vars) >= 2:
            v0, v1 = physics_vars[0], physics_vars[1]
            for a, b in zip(
                _log_values(v0.min_scale, v0.max_scale, 5),
                _log_values(v1.min_scale, v1.max_scale, 5),
            ):
                e = dict(defaults); e[v0.name] = a; e[v1.name] = b
                plan.sweeps.append(e)

        plan.preprocessing_hint = (
            "PREPROCESSING STRATEGY FOR SPECTROMETER DATA:\n"
            "Assisting law: spectral_radiance R(ω) ∝ n(ω,T) * ω³\n"
            "1. Compute n = spectral_radiance / probe_frequency³\n"
            "   (the proportionality constant becomes part of the law constant)\n"
            "2. You now have (omega, T) → n. Discover n(omega, T)."
        )

    # --- calorimeter systems ---
    elif is_calorimeter:
        narrow_bw = 1e5
        defaults["bandwidth"] = narrow_bw

        for v in physics_vars:
            if v.name == "bandwidth":
                continue
            for val in _log_values(v.min_scale, v.max_scale, pts):
                e = dict(defaults); e[v.name] = val
                plan.sweeps.append(e)

        non_bw = [v for v in physics_vars if v.name != "bandwidth"]
        if len(non_bw) >= 2:
            v0, v1 = non_bw[0], non_bw[1]
            for a, b in zip(
                _log_values(v0.min_scale, v0.max_scale, 5),
                _log_values(v1.min_scale, v1.max_scale, 5),
            ):
                e = dict(defaults); e[v0.name] = a; e[v1.name] = b
                plan.sweeps.append(e)

        plan.preprocessing_hint = (
            "PREPROCESSING STRATEGY FOR CALORIMETER DATA:\n"
            "Bandwidth is set very narrow so total_power ≈ R(ω_c,T) * bandwidth\n"
            "Assisting law: R(ω,T) = n(ω,T) * ω³\n"
            "1. Compute R = total_power / bandwidth\n"
            "2. Compute n = R / center_frequency³\n"
            "3. You now have (omega, T) → n. Discover n(omega, T)."
        )

    # --- generic fallback ---
    else:
        for v in variables:
            for val in _log_values(v.min_scale, v.max_scale, pts):
                e = dict(defaults); e[v.name] = val
                plan.sweeps.append(e)
        plan.preprocessing_hint = (
            "Use the assisting equations in the prompt to convert system "
            "output into the target quantity, then discover the law."
        )

    plan.total_experiments = len(plan.sweeps) + len(plan.noise_check)
    return plan


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ExperimentPlanner:
    """
    Module-agnostic experiment planner for NewtonBench.

    Usage::

        planner = ExperimentPlanner()
        plan = planner.plan(task_prompt, system_type_override="vanilla_equation")
    """

    def __init__(self, points_per_sweep: int = 10, noise_check_reps: int = 3):
        self.points_per_sweep = points_per_sweep
        self.noise_check_reps = noise_check_reps

    def plan(
        self,
        task_prompt: str,
        system_type_override: Optional[str] = None,
    ) -> ExperimentPlan:
        system_type = system_type_override or _detect_system_type(task_prompt)
        variables = _extract_variables(task_prompt)
        if not variables:
            return ExperimentPlan(system_type=system_type)

        if system_type == "vanilla_equation":
            return _plan_vanilla(
                variables, pts=self.points_per_sweep, noise_reps=self.noise_check_reps,
            )
        return _plan_dynamics(
            variables, system_type=system_type,
            prompt=task_prompt, pts=self.points_per_sweep,
        )

    @staticmethod
    def preprocess_results(
        plan: ExperimentPlan,
        experiments: List[Dict[str, Any]],
        results: List[Any],
        prompt: str,
    ) -> List[Dict[str, Any]]:
        """
        Convert raw experimental results into reduced (inputs → target) rows.

        For trajectory data: extract F = m*a from velocity differences.
        For spectrometer: compute n = R / omega^3.
        For calorimeter: compute n = (P / bandwidth) / omega^3.
        For vanilla/generic: pass through as-is.

        Returns a list of dicts, each with input variable names and "output".
        """
        lower = prompt.lower()
        is_trajectory = ("position" in lower and "velocity" in lower) or "time_step" in lower
        is_calorimeter = "calorimeter" in lower or "total_power" in lower
        is_spectrometer = "spectral_radiance" in lower and not is_calorimeter

        rows = []
        sim_params = {"duration", "time_step", "initial_velocity", "num_points"}

        for exp, res in zip(experiments, results):
            if is_trajectory and isinstance(res, dict) and "velocity" in res:
                # Extract acceleration from first two velocity readings
                vels = res["velocity"]
                dt = exp.get("time_step", 0.001)
                if len(vels) >= 2:
                    v0, v1 = vels[0], vels[1]
                    if isinstance(v0, list):
                        # 2D: compute magnitude of acceleration
                        ax = (float(v1[0]) - float(v0[0])) / dt
                        ay = (float(v1[1]) - float(v0[1])) / dt
                        a_mag = math.sqrt(ax**2 + ay**2)
                    else:
                        a_mag = abs((float(v1) - float(v0)) / dt)
                    # F = m * a (use mass2 as the test particle mass)
                    m_test = exp.get("mass2", exp.get("m2", exp.get("mass_wire", 1.0)))
                    force = m_test * a_mag
                    row = {k: v for k, v in exp.items() if k not in sim_params}
                    row["computed_force"] = force
                    rows.append(row)
                else:
                    continue

            elif is_spectrometer and isinstance(res, (int, float)):
                omega = exp.get("probe_frequency", exp.get("omega", 1.0))
                n = float(res) / (omega ** 3) if omega != 0 else 0.0
                row = {k: v for k, v in exp.items() if k not in sim_params}
                row["computed_n"] = n
                rows.append(row)

            elif is_calorimeter and isinstance(res, (int, float)):
                bw = exp.get("bandwidth", 1.0)
                omega = exp.get("center_frequency", exp.get("omega", 1.0))
                R = float(res) / bw if bw != 0 else 0.0
                n = R / (omega ** 3) if omega != 0 else 0.0
                row = {k: v for k, v in exp.items() if k not in sim_params}
                row.pop("bandwidth", None)
                row["computed_n"] = n
                rows.append(row)

            else:
                # Vanilla / generic: pass through
                row = dict(exp)
                row["output"] = res
                rows.append(row)

        return rows

    @staticmethod
    def format_results_for_llm(
        plan: ExperimentPlan,
        experiments: List[Dict[str, Any]],
        results: List[Any],
        prompt: str = "",
    ) -> str:
        """
        Format data for the LLM.  When a prompt is provided and the data
        is from a trajectory / spectrometer / calorimeter system, the raw
        results are preprocessed first so the LLM sees only the compact
        reduced table.
        """
        lower = prompt.lower()
        needs_preprocess = (
            ("velocity" in lower)
            or ("spectral_radiance" in lower)
            or ("calorimeter" in lower)
            or ("total_power" in lower)
        )

        if needs_preprocess and prompt:
            rows = ExperimentPlanner.preprocess_results(
                plan, experiments, results, prompt,
            )
            lines = ["=" * 60, "PREPROCESSED EXPERIMENTAL DATA", "=" * 60]
            if plan.preprocessing_hint:
                lines += ["", plan.preprocessing_hint, ""]
            lines.append(
                "The raw trajectory / measurement data has already been "
                "preprocessed for you using the assisting equations. "
                "Each row below shows the physics inputs and the computed "
                "target quantity."
            )
            lines.append("")
            lines.append("--- Reduced Data ---")
            for i, row in enumerate(rows):
                lines.append(f"Exp {i + 1}: {json.dumps(row, default=str)}")
            return "\n".join(lines)

        # Vanilla / generic: raw data is already compact
        lines = ["=" * 60, "COLLECTED EXPERIMENTAL DATA", "=" * 60]
        if plan.preprocessing_hint:
            lines += ["", plan.preprocessing_hint, ""]
        lines.append("--- Raw Data ---")
        for i, (exp, res) in enumerate(zip(experiments, results)):
            exp_s = json.dumps(exp, default=str)
            if isinstance(res, (int, float)):
                res_s = f"{res:.15e}" if isinstance(res, float) else str(res)
            else:
                res_s = json.dumps(res, default=str)
            lines.append(f"Exp {i + 1}: {exp_s} → {res_s}")
        return "\n".join(lines)