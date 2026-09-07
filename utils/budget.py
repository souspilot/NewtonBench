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
from typing import Any, Dict, List, Optional

_DEFAULT_CONFIG_PATH = os.path.join("configs", "budget", "budget.json")

# Fallback used only if configs/budget/budget.json is missing or unreadable while
# budget mode is requested. Keeps a --budget run working (with a warning) instead
# of crashing.
_BUILTIN_DEFAULTS: Dict[str, Any] = {
    "results_dir": "budget_evaluation_results",
    "currency": "$",
    "defaults": {
        "starting_funds": 2000.0,
        "min_spend_fraction": 0.5,
        "cost_model": {
            "request_fee": 50.0,
            "per_datapoint_cost": 10.0,
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
            # Fill any missing top-level pieces from the builtin defaults.
            raw = _deep_merge(_BUILTIN_DEFAULTS, raw)
            return cls(raw, source=path)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[budget] WARNING: could not read {path} ({e}); using built-in default cost model.")
            return cls(copy.deepcopy(_BUILTIN_DEFAULTS), source="<builtin>")

    def module_budget(self, module_name: str) -> "ModuleBudget":
        merged = _deep_merge(self._defaults, self._modules.get(module_name, {}))
        cm = merged.get("cost_model", {})
        ms = cm.get("magnitude_surcharge", {})
        return ModuleBudget(
            module_name=module_name,
            currency=self.currency,
            starting_funds=float(merged.get("starting_funds", 2000.0)),
            min_spend_fraction=float(merged.get("min_spend_fraction", 0.5)),
            request_fee=float(cm.get("request_fee", 50.0)),
            per_datapoint_cost=float(cm.get("per_datapoint_cost", 10.0)),
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


@dataclass
class ModuleBudget:
    module_name: str
    currency: str
    starting_funds: float
    min_spend_fraction: float
    request_fee: float
    per_datapoint_cost: float
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
        """Significant figures the sensor reports at this precision level.

        This is what makes precision matter even with zero measurement noise:
        a cheap (low-precision) reading is rounded hard, an expensive one is not.
        """
        if precision in self.precision_sig_figs:
            return self.precision_sig_figs[precision]
        if not self.precision_sig_figs:
            return 15
        nearest = min(self.precision_sig_figs, key=lambda k: abs(k - precision))
        return self.precision_sig_figs[nearest]

    def price_batch(self, experiments: List[Dict[str, Any]]) -> BatchPricing:
        cleaned: List[Dict[str, Any]] = []
        precisions: List[int] = []
        per_exp_cost: List[float] = []
        total = self.request_fee if experiments else 0.0
        for exp in experiments:
            exp = dict(exp) if isinstance(exp, dict) else {}
            precision = self._coerce_precision(exp.pop("precision", self.default_precision))
            surcharge = self._magnitude_surcharge_fraction(exp)
            cost = self.per_datapoint_cost * self._precision_multiplier(precision) * (1.0 + surcharge)
            cleaned.append(exp)
            precisions.append(precision)
            per_exp_cost.append(cost)
            total += cost
        return BatchPricing(round(total, 4), cleaned, precisions, [round(c, 4) for c in per_exp_cost])

    # ---- measurement -----------------------------------------------------
    def measure(self, module, exp: Dict[str, Any], precision: int, *,
                noise_level: float, difficulty: str, system: str, law_version: Optional[str]) -> Any:
        """One paid measurement.

        - Under measurement noise, precision k averages k replicate samples
          (error shrinks ~1/sqrt(k)).
        - Regardless of noise, the reading is then quantised to the number of
          significant figures the paid-for sensor resolves (sig_figs(precision)).
          This is why buying precision is worth something even at zero noise.
        """
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
    def min_spend_amount(self) -> float:
        return self.starting_funds * self.min_spend_fraction

    def system_note(self) -> str:
        c = self.currency
        pct = int(self.min_spend_fraction * 100)
        return (
            "**Research budget:** This mission is *budgeted*. You are the principal "
            "investigator on a grant with finite funds. Every `<run_experiment>` call "
            "is billed against the grant, and every `<experiment_output>` tells you the "
            f"funds remaining (in {c}). Running out of money does not end the mission, "
            f"but every {c} spent past zero is recorded and counts against you. Plan "
            "your experimental campaign the way a real lab lead must: batch measurements, "
            "buy precision only where it changes your conclusion, and keep enough in "
            "reserve to finish.\n\n"
            f"**Minimum spend rule:** You must spend at least {pct}% of your starting "
            f"funds ({c}{self.min_spend_amount():,.0f}) before submitting your "
            "`<final_law>`. Early submissions will be rejected. Use the budget to "
            "thoroughly explore the parameter space, verify your hypotheses with "
            "higher-precision measurements, and test edge cases."
        )

    def cost_explanation(self) -> str:
        c = self.currency
        pm = self.precision_multipliers or {1: 1.0}
        prec_lines = "\n".join(
            f"      precision {k}:  cost x{pm[k]:g},  reading given to {self.sig_figs(k)} significant figures"
            + ("   (default)" if k == self.default_precision else "")
            for k in sorted(pm)
        )
        pct = int(self.min_spend_fraction * 100)
        lines = [
            "**How experiments are billed (this run only):**",
            f"- Fixed setup fee: {c}{self.request_fee:g} per `<run_experiment>` call, whatever its size.",
            f"- Per data point: {c}{self.per_datapoint_cost:g} for each parameter set in your JSON array "
            "(a 10-row array is 10 data points).",
            "- Precision / sensor grade: add an optional integer `\"precision\"` field to any parameter set. "
            "A better sensor costs more but reports the reading to more significant figures:",
            prec_lines,
            f"    The reading is rounded to that many significant figures even when measurements are "
            f"noise-free, so buying precision always buys real information. When measurements are noisy, "
            f"precision k additionally averages k repeated samples of that point (error ~ 1/sqrt(k)).",
            f"- Exotic-range surcharge: any parameter whose magnitude falls outside "
            f"[{self.ref_low:g}, {self.ref_high:g}] adds {self.frac_per_decade * 100:g}% to that data point's "
            f"cost per order of magnitude beyond the window (specialised apparatus).",
            f"- Starting funds for this mission: {c}{self.starting_funds:g}.",
            f"- **Minimum spend:** You must use at least {pct}% of your budget "
            f"({c}{self.min_spend_amount():,.0f}) before you can submit `<final_law>`.",
        ]
        if self.notes:
            lines.append(f"- Note: {self.notes}")
        lines.append(
            "Example billed request (setup fee + 2 points, the second one high-precision):\n"
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

    def meets_min_spend(self) -> bool:
        """True if the agent has spent at least the required fraction of the budget."""
        return self.spent >= self.mb.min_spend_amount()

    def min_spend_rejection(self) -> str:
        """Message returned when the agent tries to submit before meeting the minimum spend."""
        c = self.currency
        pct = int(self.mb.min_spend_fraction * 100)
        needed = self.mb.min_spend_amount()
        shortfall = needed - self.spent
        return (
            f"**Submission rejected:** You have only spent {c}{self.spent:,.0f} of your "
            f"{c}{self.mb.starting_funds:,.0f} budget. You must spend at least {pct}% "
            f"({c}{needed:,.0f}) before submitting your final law. You still need to "
            f"spend {c}{shortfall:,.0f} more. Continue experimenting — explore additional "
            "parameter ranges, buy higher-precision measurements to refine your constants, "
            "or test edge cases to verify your hypothesis."
        )

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
