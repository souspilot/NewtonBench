# Budgeted ("principal investigator") mode

Normal NewtonBench runs let the agent request unlimited experiments. Budgeted mode
gives the agent a finite research grant: every `<run_experiment>` call is billed,
the agent is told the cost model up front, and each `<experiment_output>` reports
the funds remaining. It tests whether a model plans its experimental campaign the
way a real lab lead has to.

## Turning it on

Add `--budget` to any of the runners:

```bash
python run_experiments.py --module m11_heat_transfer --model_name qwen38-27b --budget
python run_all_evaluations.py --model_name qwen38-27b --budget
python run_master.py --models qwen38-27b --budget
```

Or set `NEWTONBENCH_BUDGET=1` in the environment. Without the flag the benchmark
behaves exactly as before.

Budgeted results are written to a **separate** directory tree (default
`budget_evaluation_results/`, set by `results_dir` below) so they never mix with
the standard `evaluation_results/`. The analysis scripts take a matching
`--budget` flag:

```bash
python analysis/scoreboard.py  --model qwen38-27b --budget          # + a grant-economics section
python analysis/diagnostics.py trace --model qwen38-27b --budget    # + budget-vs-outcome block
python analysis/rejudge.py     --model qwen38-27b --budget --judge gpt41
```

`--budget` points each script at the budget tree, keeps its bookkeeping in
separate files (`results_by_trial_budget.csv`, `verdicts_<model>_budget.csv`), and
surfaces the per-trial spend / funds-remaining / overspent numbers.

## Editing costs — `budget.json`

This file is **data only**. No prompt wording or experiment code lives here; the
agent-facing explanation is generated from these numbers by `utils/budget.py`.

| Key | Meaning |
| --- | --- |
| `results_dir` | Directory tree for budgeted runs. |
| `currency` | Symbol shown in prompts / logs. |
| `defaults.starting_funds` | Grant size per trial. |
| `defaults.cost_model.request_fee_per_param` | Setup fee charged **per varied input parameter** in a request. A controlled sweep that changes one variable pays 1×; a request where points differ on 3 inputs pays 3×. |
| `defaults.cost_model.per_datapoint_cost` | Base charge per parameter set in the JSON array (before tier and precision multipliers). |
| `defaults.cost_model.batch_tiers` | Step-wise convex pricing tiers as `[[threshold, multiplier], ...]`. Points are priced in marginal brackets: the first *threshold₁* points cost `per_datapoint_cost × multiplier₁` each, the next band costs `per_datapoint_cost × multiplier₂`, etc. Larger batches cost progressively more per point. |
| `defaults.cost_model.default_precision` | Precision assumed when the agent omits the field. |
| `defaults.cost_model.precision_multipliers` | `precision` value (1–5) → per-datapoint cost multiplier. |
| `defaults.cost_model.precision_sig_figs` | `precision` value (1–5) → significant figures the reading is rounded to. This is what makes precision matter **even at zero noise**: a cheap sensor gives you a coarse (e.g. 1-sig-fig) number, an expensive one gives you more digits. With measurement noise, higher precision *additionally* averages that many replicate samples (error ~ 1/√k). |
| `defaults.cost_model.magnitude_surcharge.reference_low` / `reference_high` | The "ordinary equipment" window for parameter values. |
| `defaults.cost_model.magnitude_surcharge.fraction_per_decade_outside` | Extra fraction of a datapoint's cost per order of magnitude any parameter falls outside that window. |
| `defaults.notes` | Optional extra sentence appended to every module's cost explanation. |

### Per-module overrides

Anything under `modules.<module_name>` is deep-merged onto `defaults` for that
module only. Add a block for any module that needs different economics; omit it
and the module uses `defaults`.

## Experiment volume caps

Independent of budgeted mode, **every** run (budgeted or not) is bounded by
[`configs/experiment_limits.json`](../experiment_limits.json):

| Key | Meaning |
| --- | --- |
| `max_datapoints_per_request` | Max parameter sets in one `<run_experiment>` array; extras are dropped and the agent is told. |
| `max_datapoints_per_trial` | Max data points over the whole mission; once reached, `<run_experiment>` is refused and the agent must submit `<final_law>`. |

These stop a single trial from ballooning its runtime or overflowing the context
window. Set a value very high to effectively disable that cap.
