# NewtonBench pilot scoring protocol

This protocol separates scientific performance from runner reliability and
never turns missing evaluation data into a smaller accuracy denominator.

## Scientific trial set

- A task configuration is `(module, noise, difficulty, system, law version,
  agent backend)`.
- Score the latest four completed `trialN.json` trajectories per configuration.
- A malformed, missing, or non-executable submitted law is a completed
  scientific trajectory with symbolic accuracy 0. Missing RMSLE does not remove
  it from symbolic-accuracy calculations.
- Re-run directories are de-duplicated by logical trial identity. A completed
  record is preferred to a failure stub, then the latest `_vN` and mtime win.
- `trialN_fail.json` means the runner exhausted its retries. These records are
  excluded from scientific accuracy and reported separately as operational
  failures. They should not be silently discarded from the paper's reliability
  discussion.

## Symbolic label

The primary checker is deterministic and has three outcomes:

1. `constant_equivalent`: pass.
2. `structurally_different`: fail.
3. `not_checkable`: unresolved.

RMSLE and the original experiment-time LLM judge are audit signals only. They
do not resolve `not_checkable`. This matters because NewtonBench intentionally
ignores physical-constant values, while numerical fit depends on those values.

Unresolved cases require an explicit human or independent locally hosted LLM
adjudication. The judge must fit on at most two A100 80GB GPUs. Record the exact
model, revision, quantization, prompt, and inference settings. Do not let a model
judge its own trajectories for a publication label.

## Reproducible workflow

Run deterministic verdicts:

```bash
python analysis/diagnostics.py verdicts \
  --model MODEL \
  --budget-config configs/budget/BUDGET_CONFIG.json \
  --subset_file configs/representative_subset.json
```

If unresolved rows remain, the command writes a result-tree-specific file such
as `analysis/unresolved_MODEL_budget_evaluation_results_400_cap.csv`. Fill every `adjudicated_success` cell
with 0/1, set `adjudicator`, add notes for ambiguous cases, and rerun:

```bash
python analysis/diagnostics.py verdicts \
  --model MODEL \
  --budget-config configs/budget/BUDGET_CONFIG.json \
  --subset_file configs/representative_subset.json \
  --adjudications analysis/unresolved_MODEL_budget_evaluation_results_400_cap.csv
```

Then generate a publication-safe scoreboard:

```bash
python analysis/scoreboard.py \
  --model MODEL \
  --budget-config configs/budget/BUDGET_CONFIG.json \
  --subset_file configs/representative_subset.json \
  --verified
```

`--verified` fails closed if the verdict cache is stale or any displayed trial
is unresolved. Without it, the scoreboard is clearly labeled
`PROVISIONAL ORIGINAL-JUDGE`.

## Required pilot audit fields

Archive the per-trial CSV, verdict CSV, adjudication CSV, command line, git
commit, result-tree checksum, model server configuration, and operational
failure counts. Report both the scientific trial count and runner failure count.
