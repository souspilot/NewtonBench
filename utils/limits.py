# utils/limits.py
"""
Hard caps on experiment volume, applied to every trial (budgeted or not).

Without a cap an agent can put thousands of parameter sets in one
<run_experiment> array: thousands of physics calls, a huge <experiment_output>
block echoed back into the context every subsequent turn, and a trial that
takes far too long. These limits bound both.

Values come from configs/experiment_limits.json so they can be tuned without
touching prompt text or agent code. Set a limit very high to disable it.
"""

import os
import json
from dataclasses import dataclass
from typing import Optional

_DEFAULT_PATH = os.path.join("configs", "experiment_limits.json")

_BUILTIN = {
    "max_datapoints_per_request": 20,
    "max_datapoints_per_trial": 200,
}


@dataclass
class ExperimentLimits:
    max_datapoints_per_request: int
    max_datapoints_per_trial: int

    @classmethod
    def load(cls, path: Optional[str] = None) -> "ExperimentLimits":
        path = path or _DEFAULT_PATH
        cfg = dict(_BUILTIN)
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for k in _BUILTIN:
                if isinstance(raw.get(k), (int, float)):
                    cfg[k] = int(raw[k])
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[limits] WARNING: could not read {path} ({e}); using built-in defaults {cfg}.")
        return cls(**cfg)

    def note(self) -> str:
        """One short paragraph for the system prompt."""
        return (
            "**Experiment limits:** A single `<run_experiment>` call may contain at most "
            f"{self.max_datapoints_per_request} parameter sets -- any extras are dropped and not "
            f"measured. Across the entire mission you may collect at most "
            f"{self.max_datapoints_per_trial} data points; once that is reached, further "
            "`<run_experiment>` calls are refused and you must submit your `<final_law>`. "
            "Design a compact, informative set of experiments."
        )
