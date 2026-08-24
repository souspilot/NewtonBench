# Planned Agent for NewtonBench

A self-contained agent that separates **experiment design** (deterministic)
from **equation discovery** (LLM).  All code lives in `planned_agent/` — the
only change to the original NewtonBench repo is a small patch in
`run_experiments.py` to register the new backend.

## Directory layout

```
NewtonBench/                        ← original repo (unchanged except patch below)
├── planned_agent/                  ← ★ your code, self-contained
│   ├── __init__.py                 ← exports conduct_planned_exploration
│   ├── planner.py                  ← ExperimentPlanner class
│   ├── agent.py                    ← agent loop (plan → execute → LLM discover)
│   └── test_planner.py             ← validation (no LLM needed)
├── utils/                          ← original benchmark code (untouched)
├── modules/                        ← original benchmark code (untouched)
└── run_experiments.py              ← one small patch (see below)
```

## Installation

1. Copy the `planned_agent/` folder into your NewtonBench repo root.

2. Apply this patch to `run_experiments.py` (3 edits):

**Edit 1** — Add import (after the existing `code_assisted_agent` import block):

```python
try:
    from planned_agent import conduct_planned_exploration
    _WITH_PLANNED_AGENT = True
except Exception:
    _WITH_PLANNED_AGENT = False
```

**Edit 2** — Add dispatch branch in `run_trial()` (insert *before* the
`if agent_backend == "code_assisted_agent"` line):

```python
            if agent_backend == "planned_agent" and _WITH_PLANNED_AGENT:
                exploration_result = conduct_planned_exploration(
                    module=module,
                    model_name=model_name,
                    noise_level=noise_level,
                    difficulty=difficulty,
                    system=system,
                    law_version=law_version,
                    trial_info=trial_info
                )
            elif agent_backend == "code_assisted_agent" and _WITH_CODE_ASSISTANCE:
```

(Change the existing `if` to `elif`.)

**Edit 3** — Add `"planned_agent"` to the argparse choices:

```python
choices=["vanilla_agent", "code_assisted_agent", "planned_agent"]
```

Optionally apply the same choices change to `run_all_evaluations.py`, and
add `"planned_agent"` to the backend loop in `run_master.py`.

## Quick test (no LLM)

```bash
cd NewtonBench
python planned_agent/test_planner.py    # or: python -m planned_agent.test_planner
```

Validates prompt parsing + experiment execution for all 12 modules × 3
system types = 36 configurations.

## Running experiments

```bash
# Single config
python run_experiments.py \
    --module m0_gravity \
    --model_name qwq-32b \
    -d easy -m vanilla_equation \
    -b planned_agent -t 4

# All difficulties + system types for one module
python run_all_evaluations.py \
    --module m0_gravity \
    --model_name qwq-32b \
    --agent_backend planned_agent
```

Results land in `evaluation_results/{model}/{module}/planned_agent/…` —
same structure as the existing agents, so all analysis scripts work unchanged.

## How it works

| Phase | Uses LLM? | What happens |
|-------|-----------|-------------|
| 1. Plan | No | Parse task prompt → extract variables + scales → generate controlled-variable sweeps |
| 2. Execute | No | Run all planned experiments via `module.run_experiment_for_module()` |
| 3. Discover | Yes | Give LLM the curated dataset + preprocessing hints → LLM proposes equation |

The planner detects the system type from the prompt text and generates
appropriate strategies:

- **vanilla_equation** — log-spaced one-at-a-time sweeps + pairwise interaction sweep
- **trajectory systems** (gravity, coulomb, etc.) — short duration, small timestep,
  zero initial velocity → clean F = m·a extraction from first acceleration
- **spectrometer systems** — direct sweeps, with `n = R/ω³` inversion hint
- **calorimeter systems** — narrow bandwidth for point approximation,
  with `n = (P/Δω)/ω³` inversion hint
- **generic fallback** — sweep all variables, hint to use assisting equations