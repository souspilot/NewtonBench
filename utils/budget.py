# utils/budget.py
"""
Budgeted ("principal investigator") mode for NewtonBench.

When enabled, every <run_experiment> call costs money drawn from a finite grant.
The agent is told the cost model up front (generated here from configs/budget/budget.json)
and every <experiment_output> reports the funds remaining.

Nothing in this module hard-codes cost values or the results directory -- all of
that comes from configs/budget/budget.json so it can be tuned without touching
prompt text or experiment code. See configs/budget/README.md.

Public surface:
    is_budget_enabled(cli_flag)      -> bool          (flag OR NEWTONBENCH_BUDGET env)
    load_budget_config(path=None)    -> BudgetConfig
    BudgetConfig.results_dir         -> str
    BudgetConfig.module_budget(name) -> ModuleBudget  (deep-merged defaults + per-module)
    ModuleBudget.system_note()       -> str           (generic paragraph for the system prompt)
    ModuleBudget.cost_explanation()  -> str           (module-specific cost model for the task prompt)
    ModuleBudget.price_batch(exps)   -> BatchPricing   (strip 'precision', compute cost)
    ModuleBudget.measure(module, exp, precision, ...)  -> measurement (replicate-averaged under noise)
    BudgetTracker(module_budget)     -> running balance, .charge(), .status_line(), .summary()
"""

import os
import json
import math
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_DEFAULT_CONFIG_PATH = os.path.join("configs", "budget", "budget.json")

_BUILTIN_DEFAULTS: Dict[str, Any] = {
    "results_dir": "budget_evaluation_results",
    "currency": "$",
    "defaults": {
        "starting_funds": 1000.0,
        "cost_model": {
            "request_fee_per_param": 30.0,
            "per_datapoint_cost": 15.0,
            "batch_tiers": [[5, 1.0], [10, 2.0], [15, 3.5], [20, 6.0]],
            "default_precision": 1,
            "precision_multipliers": {"1": 1.0, "2": 1.6, "3": 2.5, "4": 4.0, "5": 6.5},
            "precision_sig_figs": {"1": 1, "2": 2, "3": 3, "4": 4, "5": 6},
            "magnitude_surcharge": {
                "reference_low": 1e-3,
                "reference_high": 1e3,
                "fraction_per_decade_outside": 0.15,
            },
        },
        "notes": "",
    },
    "modules": {},
}


def load_budget_config(path: Optional[str] = None) -> "BudgetConfig":
    """Convenience wrapper around BudgetConfig.load()."""
    return BudgetConfig.load(path)


def is_budget_enabled(cli_flag: bool = False) -> bool:
    """Budget mode is on if --budget was passed or NEWTONBENCH_BUDGET is truthy."""
    if cli_flag:
        return True
    return os.environ.get("NEWTONBENCH_BUDGET", "").strip().lower() in ("1", "true", "yes", "on")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class BudgetConfig:
    """Parsed configs/budget/budget.json (or the built-in fallback)."""

    def __init__(self, raw: Dict[str, Any], source: str = "<builtin>"):
        self._raw = raw
        self.source = source
        self.results_dir: str = raw.get("results_dir", "budget_evaluation_results")
        self.currency: str = raw.get("currency", "$")
        self._defaults: Dict[str, Any] = raw.get("defaults", {})
        self._modules: Dict[str, Any] = raw.get("modules", {})

    @classmethod
    def load(cls, path: Optional[str] = None) -> "BudgetConfig":
        path = path or _DEFAULT_CONFIG_PATH
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            raw = _deep_merge(_BUILTIN_DEFAULTS, raw)
            return cls(raw, source=path)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[budget] WARNING: could not read {path} ({e}); using built-in default cost model.")
            return cls(copy.deepcopy(_BUILTIN_DEFAULTS), source="<builtin>")

    def module_budget(self, module_name: str) -> "ModuleBudget":
        merged = _deep_merge(self._defaults, self._modules.get(module_name, {}))
        cm = merged.get("cost_model", {})
        ms = cm.get("magnitude_surcharge", {})
        batch_tiers_raw = cm.get("batch_tiers", [[20, 1.0]])
        batch_tiers = [(int(t[0]), float(t[1])) for t in batch_tiers_raw]
        batch_tiers.sort(key=lambda t: t[0])
        return ModuleBudget(
            module_name=module_name,
            currency=self.currency,
            starting_funds=float(merged.get("starting_funds", 1000.0)),
            request_fee_per_param=float(cm.get("request_fee_per_param", 30.0)),
            per_datapoint_cost=float(cm.get("per_datapoint_cost", 15.0)),
            batch_tiers=batch_tiers,
            default_precision=int(cm.get("default_precision", 1)),
            precision_multipliers={int(k): float(v) for k, v in cm.get("precision_multipliers", {}).items()},
            precision_sig_figs={int(k): int(v) for k, v in cm.get("precision_sig_figs", {}).items()},
            ref_low=float(ms.get("reference_low", 1e-3)),
            ref_high=float(ms.get("reference_high", 1e3)),
            frac_per_decade=float(ms.get("fraction_per_decade_outside", 0.15)),
            notes=str(merged.get("notes", "") or ""),
        )


@dataclass
class BatchPricing:
    total_cost: float
    cleaned_experiments: List[Dict[str, Any]]   # 'precision' removed, ready for the physics call
    precisions: List[int]                       # per-experiment precision actually used
    per_experiment_cost: List[float]
    request_fee: float = 0.0
    num_varied_params: int = 0


@dataclass
class ModuleBudget:
    module_name: str
    currency: str
    starting_funds: float
    request_fee_per_param: float
    per_datapoint_cost: float
    batch_tiers: List[Tuple[int, float]]
    default_precision: int
    precision_multipliers: Dict[int, float]
    precision_sig_figs: Dict[int, int]
    ref_low: float
    ref_high: float
    frac_per_decade: float
    notes: str = ""

    # ---- pricing -----------------------------------------------------------
    def _precision_multiplier(self, precision: int) -> float:
        if precision in self.precision_multipliers:
            return self.precision_multipliers[precision]
        if not self.precision_multipliers:
            return 1.0
        nearest = min(self.precision_multipliers, key=lambda k: abs(k - precision))
        return self.precision_multipliers[nearest]

    def _magnitude_surcharge_fraction(self, exp: Dict[str, Any]) -> float:
        """Extra fraction of a datapoint's cost for parameters outside the ordinary window."""
        frac = 0.0
        lo, hi = math.log10(self.ref_low), math.log10(self.ref_high)
        for key, val in exp.items():
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                continue
            a = abs(float(val))
            if a <= 0.0:
                continue
            d = math.log10(a)
            decades_out = max(0.0, d - hi) + max(0.0, lo - d)
            frac += self.frac_per_decade * decades_out
        return frac

    def _coerce_precision(self, raw: Any) -> int:
        try:
            p = int(round(float(raw)))
        except (TypeError, ValueError):
            return self.default_precision
        keys = self.precision_multipliers.keys()
        if keys:
            return max(min(p, max(keys)), min(keys))
        return max(p, 1)

    def sig_figs(self, precision: int) -> int:
        if precision in self.precision_sig_figs:
            return self.precision_sig_figs[precision]
        if not self.precision_sig_figs:
            return 15
        nearest = min(self.precision_sig_figs, key=lambda k: abs(k - precision))
        return self.precision_sig_figs[nearest]

    def _tiered_datapoint_cost(self, batch_index: int) -> float:
        """Per-datapoint base cost for the (batch_index+1)-th point in a batch.

        Uses marginal-bracket pricing: the first few points are cheap, later
        points in the same request cost more (step-wise convex).
        """
        position = batch_index + 1  # 1-based
        for threshold, multiplier in self.batch_tiers:
            if position <= threshold:
                return self.per_datapoint_cost * multiplier
        # Beyond all tiers: use the last tier's multiplier
        return self.per_datapoint_cost * self.batch_tiers[-1][1]

    def _count_varied_params(self, experiments: List[Dict[str, Any]]) -> int:
        """Count parameters that take more than one distinct value across the batch."""
        if len(experiments) <= 1:
            return max(1, len(experiments[0]) if experiments else 0)
        all_keys: set = set()
        for exp in experiments:
            all_keys.update(exp.keys())
        varied = 0
        for key in all_keys:
            vals = set()
            for exp in experiments:
                if key in exp:
                    v = exp[key]
                    vals.add(v if not isinstance(v, float) or v == v else "__nan__")
            if len(vals) > 1:
                varied += 1
        return max(varied, 1)

    def price_batch(self, experiments: List[Dict[str, Any]]) -> BatchPricing:
        cleaned: List[Dict[str, Any]] = []
        precisions: List[int] = []
        per_exp_cost: List[float] = []

        # Strip precision from experiments before counting varied params
        stripped_exps: List[Dict[str, Any]] = []
        raw_precisions: List[int] = []
        for exp in experiments:
            exp = dict(exp) if isinstance(exp, dict) else {}
            precision = self._coerce_precision(exp.pop("precision", self.default_precision))
            stripped_exps.append(exp)
            raw_precisions.append(precision)

        num_varied = self._count_varied_params(stripped_exps) if stripped_exps else 0
        request_fee = self.request_fee_per_param * num_varied if stripped_exps else 0.0
        total = request_fee

        for i, (exp, precision) in enumerate(zip(stripped_exps, raw_precisions)):
            surcharge = self._magnitude_surcharge_fraction(exp)
            base_cost = self._tiered_datapoint_cost(i)
            cost = base_cost * self._precision_multiplier(precision) * (1.0 + surcharge)
            cleaned.append(exp)
            precisions.append(precision)
            per_exp_cost.append(cost)
            total += cost

        return BatchPricing(
            total_cost=round(total, 4),
            cleaned_experiments=cleaned,
            precisions=precisions,
            per_experiment_cost=[round(c, 4) for c in per_exp_cost],
            request_fee=round(request_fee, 4),
            num_varied_params=num_varied,
        )

    # ---- measurement -----------------------------------------------------
    def measure(self, module, exp: Dict[str, Any], precision: int, *,
                noise_level: float, difficulty: str, system: str, law_version: Optional[str]) -> Any:
        replicates = precision if noise_level and noise_level > 0.0 else 1
        replicates = max(1, int(replicates))
        results = [
            module.run_experiment_for_module(
                **exp, noise_level=noise_level, difficulty=difficulty,
                system=system, law_version=law_version,
            )
            for _ in range(replicates)
        ]
        value = results[0] if replicates == 1 else _average_results(results)
        return _round_to_sig_figs(value, self.sig_figs(precision))

    # ---- prompt text (generated from the numbers above) ------------------
    def system_note(self) -> str:
        c = self.currency
        return (
            "**Research budget:** This mission is *budgeted*. You are the principal "
            "investigator on a grant with finite funds. Every `<run_experiment>` call "
            "is billed against the grant, and every `<experiment_output>` tells you the "
            f"funds remaining (in {c}). Running out of money does not end the mission, "
            f"but every {c} spent past zero is recorded and counts against you. Plan "
            "your experimental campaign the way a real lab lead must: choose batch sizes "
            "carefully, buy precision only where it changes your conclusion, and keep "
            "enough in reserve to verify before you submit."
        )

    def cost_explanation(self) -> str:
        c = self.currency
        pm = self.precision_multipliers or {1: 1.0}
        prec_lines = "\n".join(
            f"      precision {k}:  cost x{pm[k]:g},  reading given to {self.sig_figs(k)} significant figures"
            + ("   (default)" if k == self.default_precision else "")
            for k in sorted(pm)
        )
        tier_lines = []
        prev = 0
        for threshold, mult in self.batch_tiers:
            tier_lines.append(
                f"      points {prev + 1}-{threshold}: {c}{self.per_datapoint_cost * mult:g} each (x{mult:g})"
            )
            prev = threshold
        tier_text = "\n".join(tier_lines)
        lines = [
            "**How experiments are billed (this run only):**",
            f"- Setup fee: {c}{self.request_fee_per_param:g} for each input parameter "
            "you vary in the request. A controlled sweep that changes only one variable "
            f"pays {c}{self.request_fee_per_param:g}; a request where every point differs "
            f"on 3 inputs pays {c}{self.request_fee_per_param * 3:g}.",
            "- Per data point (step-wise pricing — larger batches cost progressively more "
            "per point):",
            tier_text,
            "- Precision / sensor grade: add an optional integer `\"precision\"` field to any parameter set. "
            "A better sensor costs more but reports the reading to more significant figures:",
            prec_lines,
            "    The reading is rounded to that many significant figures even when measurements are "
            "noise-free, so buying precision always buys real information. When measurements are noisy, "
            "precision k additionally averages k repeated samples of that point (error ~ 1/sqrt(k)).",
            f"- Exotic-range surcharge: any parameter whose magnitude falls outside "
            f"[{self.ref_low:g}, {self.ref_high:g}] adds {self.frac_per_decade * 100:g}% to that data point's "
            f"cost per order of magnitude beyond the window (specialised apparatus).",
            f"- Starting funds for this mission: {c}{self.starting_funds:g}.",
        ]
        if self.notes:
            lines.append(f"- Note: {self.notes}")
        lines.append(
            "Example billed request (setup fee for 1 varied param + 2 points, the second one high-precision):\n"
            "<run_experiment>\n"
            "[\n"
            "  {\"...\": 1.0},\n"
            "  {\"...\": 1000.0, \"precision\": 4}\n"
            "]\n"
            "</run_experiment>"
        )
        return "\n".join(lines)


def _average_results(results: List[Any]) -> Any:
    """Componentwise mean of replicate measurements, preserving the result's shape/type."""
    first = results[0]
    if isinstance(first, dict):
        keys = first.keys()
        return {k: _mean_numeric([r.get(k) for r in results]) for k in keys}
    if isinstance(first, bool):
        return sum(1 for r in results if r) * 2 >= len(results)
    if isinstance(first, int) and not isinstance(first, bool):
        m = _mean_numeric(results)
        return int(round(m)) if m == m else first  # m != m  <=> NaN
    return _mean_numeric(results)


def _mean_numeric(values: List[Any]) -> float:
    nums = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v]
    if not nums:
        return float("nan")
    return sum(nums) / len(nums)


def round_sig(x: float, sig: int) -> float:
    """Round a float to `sig` significant figures. NaN/inf/0 pass through."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return x
    if xf == 0.0 or not math.isfinite(xf) or sig >= 15:
        return xf
    return float(f"{xf:.{max(sig, 1)}g}")


def _round_to_sig_figs(value: Any, sig: int) -> Any:
    """Quantise a measurement (scalar, or dict of scalars) to `sig` significant
    figures. Integers (e.g. light-bulb counts) and booleans are left exact."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        return {k: (_round_to_sig_figs(v, sig)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_round_to_sig_figs(v, sig) for v in value)
    if isinstance(value, float):
        return round_sig(value, sig)
    return value


class BudgetTracker:
    """Running grant balance for one trial. Overdraft is allowed and recorded."""

    def __init__(self, module_budget: ModuleBudget):
        self.mb = module_budget
        self.currency = module_budget.currency
        self.starting_funds = module_budget.starting_funds
        self.remaining = module_budget.starting_funds
        self.spent = 0.0
        self.overdraft = 0.0          # max amount the balance has ever been below zero
        self.num_charges = 0
        self.history: List[Dict[str, Any]] = []

    def charge(self, amount: float, meta: Optional[Dict[str, Any]] = None) -> None:
        amount = round(float(amount), 4)
        self.spent = round(self.spent + amount, 4)
        self.remaining = round(self.remaining - amount, 4)
        self.num_charges += 1
        if self.remaining < 0:
            self.overdraft = round(max(self.overdraft, -self.remaining), 4)
        entry = {"amount": amount, "remaining": self.remaining}
        if meta:
            entry.update(meta)
        self.history.append(entry)

    def status_line(self, last_cost: Optional[float] = None) -> str:
        c = self.currency
        bits = [f"Funds remaining: {c}{self.remaining:,.2f}"]
        if last_cost is not None:
            bits.append(f"this experiment cost {c}{last_cost:,.2f}")
        bits.append(f"total spent {c}{self.spent:,.2f} of {c}{self.starting_funds:,.2f}")
        if self.remaining < 0:
            bits.append(f"OVERDRAWN by {c}{-self.remaining:,.2f} -- overspending is recorded")
        return "**Budget:** " + "; ".join(bits) + "."

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "currency": self.currency,
            "module": self.mb.module_name,
            "starting_funds": self.starting_funds,
            "total_spent": self.spent,
            "funds_remaining": self.remaining,
            "overdraft": self.overdraft,
            "overspent": self.remaining < 0,
            "num_billed_requests": self.num_charges,
            "config_source": None,  # filled in by the caller if useful
        }
