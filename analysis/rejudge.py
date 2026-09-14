#!/usr/bin/env python3
"""
Re-Judge NewtonBench Results
=============================
Re-evaluates existing trial results with a different LLM judge model
without re-running any experiments.

Walks the evaluation_results directory, loads each trial JSON, re-runs
the symbolic equivalence check with the specified judge model, and
writes updated results alongside the originals.

Usage:
    # Use an independently hosted local judge that fits the project's hardware.
    python rejudge.py --model muse-glimmer-30b --judge gemma4-31b

    # Re-judge only one module
    python rejudge.py --model muse-glimmer-30b --judge gemma4-31b --module m0_gravity

    # Dry run: show what would be re-judged
    python rejudge.py --model muse-glimmer-30b --judge gemma4-31b --dry-run

    # Overwrite original files instead of creating new ones
    python rejudge.py --model muse-glimmer-30b --judge gemma4-31b --in-place
"""

import argparse
import hashlib
import importlib
import json
import os
import sys
import numpy as np
import traceback
from pathlib import Path

# Executing ``python analysis/rejudge.py`` makes ``analysis/`` Python's import
# root, not necessarily the repository root.  Make imports such as
# ``modules.m8_sound_speed`` independent of the caller's PYTHONPATH/cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from newton_common import filter_to_subset, load_trial_failures, load_trials  # noqa: E402


def load_module(module_name: str):
    """Import a NewtonBench physics module."""
    return importlib.import_module(f"modules.{module_name}")


def rejudge_trial(
    trial_data: dict,
    judge_model_name: str,
) -> dict:
    """
    Re-evaluate a single trial's submitted law with a new judge.

    Returns updated evaluation dict.
    """
    module_name = trial_data.get("module_name")
    difficulty = trial_data.get("equation_difficulty", "easy")
    law_version = trial_data.get("law_version")
    submitted_law = trial_data.get("submitted_law", "")

    if not module_name or not submitted_law:
        return None

    mod = load_module(module_name)

    # Re-run the module's evaluate_law with the new judge
    new_eval = mod.evaluate_law(
        llm_function_str=submitted_law,
        param_description=mod.PARAM_DESCRIPTION,
        difficulty=difficulty,
        law_version=law_version,
        judge_model_name=judge_model_name,
        trial_info={"trial_id": trial_data.get("trial_id", 0)},
    )

    symbolic_msg = str(new_eval.get("symbolic_msg") or "")
    if symbolic_msg.startswith("Symbolic equivalence check failed:"):
        raise RuntimeError(symbolic_msg)

    return new_eval


def rejudge_aggregated(trial_results: list) -> dict:
    """Recompute aggregated_results.json from re-judged trials."""
    rmsles = []
    accuracies = []
    for t in trial_results:
        ev = t.get("evaluation", {})
        rmsle = ev.get("rmsle", float("nan"))
        acc = ev.get("exact_accuracy", 0.0)
        if not np.isnan(rmsle):
            rmsles.append(rmsle)
        accuracies.append(acc)

    return {
        "average_rmsle": float(np.mean(rmsles)) if rmsles else float("nan"),
        "average_exact_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "num_total_trials": len(trial_results),
    }


def main():
    parser = argparse.ArgumentParser(description="Re-judge NewtonBench results")
    parser.add_argument("--model", required=True, help="Model whose results to re-judge")
    parser.add_argument("--judge", required=True,
                        help="Independent locally hosted judge model. It must fit on at most two "
                             "80GB A100 GPUs under the project's evaluation policy.")
    parser.add_argument("--allow-self-judge", action="store_true",
                        help="Allow judge == evaluated model for diagnostics only; never use this "
                             "output as the publication label.")
    parser.add_argument("--base-dir", default=None,
                        help="default: evaluation_results, or the budgeted tree when --budget is set")
    parser.add_argument("--budget", action="store_true",
                        help="re-judge a budgeted run (read/write under the budget results tree)")
    parser.add_argument("--budget-config", default=None,
                        help="Budget run JSON whose results_dir should be re-judged; also enables --budget")
    parser.add_argument("--module", default=None, help="Restrict to one module")
    parser.add_argument("--agent", default=None, help="Restrict to one agent backend")
    parser.add_argument("--subset-file", default=None,
                        help="representative-subset JSON; applies the same trial selection, "
                             "rerun deduplication, and four-trial cap as diagnostics/scoreboard")
    parser.add_argument("--dry-run", action="store_true", help="Just count trials, don't re-judge")
    parser.add_argument("--in-place", action="store_true",
                        help="Overwrite original trial JSONs. Default: write to a parallel directory.")
    parser.add_argument("--resume", action="store_true",
                        help="reuse an existing parallel output only when its judge and source-file "
                             "hash match; useful after an interrupted run")
    parser.add_argument("--output-suffix", default=None,
                        help="Suffix for output dir (default: judge model name)")
    parser.add_argument("--judge-max-tokens", type=int, default=None,
                        help="optional maximum completion tokens for each judge call "
                             "(default: no explicit cap)")
    args = parser.parse_args()
    args.budget = args.budget or bool(args.budget_config)
    if args.judge == args.model and not args.allow_self_judge:
        raise SystemExit("Refusing self-judging. Choose an independent local judge, or pass "
                         "--allow-self-judge for a diagnostic-only run.")
    if args.resume and args.in_place:
        raise SystemExit("--resume cannot be combined with --in-place")
    if args.judge_max_tokens is not None and args.judge_max_tokens <= 0:
        raise SystemExit("--judge-max-tokens must be positive")
    if args.judge_max_tokens is not None:
        # Module imports happen lazily below, so the API helper will see this cap.
        os.environ["LLM_MAX_TOKENS"] = str(args.judge_max_tokens)

    if args.base_dir is None:
        if args.budget:
            try:
                from utils.budget import load_budget_config
                args.base_dir = load_budget_config(args.budget_config).results_dir
            except Exception:
                if args.budget_config:
                    raise
                args.base_dir = "budget_evaluation_results"
        else:
            args.base_dir = "evaluation_results"

    model_dir = Path(args.base_dir) / args.model
    if not model_dir.exists():
        print(f"ERROR: {model_dir} not found")
        return

    suffix = args.output_suffix or args.judge
    output_model_name = f"{args.model}_judged_by_{suffix}"
    output_base = Path(args.base_dir) / output_model_name

    # Select exactly the same completed scientific trials that diagnostics and
    # scoreboard will consume: completed-over-failure preference, latest rerun,
    # and at most four trials per full task configuration. Infrastructure
    # failures are retained separately for operational reporting.
    completed = filter_to_subset(
        load_trials(args.base_dir, args.model, include_fails=False),
        args.subset_file,
    )
    failures = filter_to_subset(
        load_trial_failures(args.base_dir, args.model),
        args.subset_file,
    )
    if args.module:
        completed = completed[completed["module"] == args.module]
        failures = failures[failures["module"] == args.module]
    if args.agent:
        completed = completed[completed["agent_backend"] == args.agent]
        failures = failures[failures["agent_backend"] == args.agent]

    trial_files = sorted((Path(p) for p in completed["path"]), key=lambda p: str(p.relative_to(model_dir)))
    fail_files = sorted((Path(p) for p in failures["path"]), key=lambda p: str(p.relative_to(model_dir)))

    print(f"Found {len(trial_files)} completed trial files to re-judge "
          f"(+ {len(fail_files)} operational failure files copied separately, not scored)")
    print(f"  Model: {args.model}")
    print(f"  Judge: {args.judge}")
    print("  Policy: judge must be independently hosted locally and fit on <=2x A100-80GB")
    if args.module:
        print(f"  Module filter: {args.module}")
    if args.agent:
        print(f"  Agent filter: {args.agent}")
    if args.subset_file:
        print(f"  Subset: {args.subset_file}")

    if args.dry_run:
        # Count by module/agent
        from collections import Counter
        by_module = Counter()
        by_agent = Counter()
        for tf in trial_files + fail_files:
            parts = tf.relative_to(model_dir).parts
            by_module[parts[0]] += 1
            by_agent[parts[1]] += 1
        print("\n  By module (re-judge + carried-over fails):")
        for m, c in sorted(by_module.items()):
            print(f"    {m}: {c} trials")
        print("\n  By agent:")
        for a, c in sorted(by_agent.items()):
            print(f"    {a}: {c} trials")
        return

    if not args.in_place:
        print(f"  Output: {output_base}/")

    # Process each trial
    success, fail, changed = 0, 0, 0
    # Group by config dir for aggregated results
    config_trials = {}

    for i, trial_path in enumerate(trial_files):
        rel = trial_path.relative_to(model_dir)
        config_dir = trial_path.parent.parent  # up from trials/

        print(f"  [{i+1}/{len(trial_files)}] {rel} ... ", end="", flush=True)

        try:
            source_bytes = trial_path.read_bytes()
            source_sha256 = hashlib.sha256(source_bytes).hexdigest()
            trial_data = json.loads(source_bytes)

            out_path = trial_path if args.in_place else output_base / rel
            if args.resume and out_path.exists():
                try:
                    existing = json.loads(out_path.read_bytes())
                except Exception:  # malformed partial output must be regenerated
                    existing = None
                if (existing is not None
                        and existing.get("LLM judge") == args.judge
                        and existing.get("rejudge_source_sha256") == source_sha256):
                    config_key = str(config_dir)
                    config_trials.setdefault(config_key, []).append(existing)
                    success += 1
                    print("RESUMED")
                    continue

            old_acc = trial_data.get("evaluation", {}).get("exact_accuracy", 0.0)

            new_eval = rejudge_trial(trial_data, args.judge)
            if new_eval is None:
                print("SKIP (missing data)")
                fail += 1
                continue

            new_acc = new_eval.get("exact_accuracy", 0.0)

            # Update the trial data (capture the ORIGINAL judge before overwriting it)
            original_judge = trial_data.get("LLM judge", "unknown")
            trial_data["evaluation"] = new_eval
            trial_data["original_judge"] = original_judge
            trial_data["LLM judge"] = args.judge
            if not args.in_place:
                # The directory name is the analysis run identity. Preserve
                # the evaluated checkpoint separately so scoreboard can load
                # the parallel tree by its directory/model name without
                # filtering every row back out as the original model.
                trial_data["evaluated_model"] = trial_data.get("model_name", args.model)
                trial_data["model_name"] = output_model_name
                trial_data["rejudge_source_sha256"] = source_sha256

            # Write output
            if not args.in_place:
                out_path.parent.mkdir(parents=True, exist_ok=True)

            with open(out_path, "w") as f:
                json.dump(trial_data, f, indent=2)

            # Track for aggregation
            config_key = str(config_dir)
            if config_key not in config_trials:
                config_trials[config_key] = []
            config_trials[config_key].append(trial_data)

            status = "OK"
            if old_acc != new_acc:
                status = f"CHANGED {old_acc:.0f}→{new_acc:.0f}"
                changed += 1
            print(status)
            success += 1

        except Exception as e:
            print(f"FAIL ({e})")
            traceback.print_exc()
            fail += 1

    # Preserve failure records in parallel output for operational audits, but do
    # not mix runner/API failures into scientific accuracy.
    for trial_path in fail_files:
        rel = trial_path.relative_to(model_dir)
        config_dir = trial_path.parent.parent
        try:
            with open(trial_path) as f:
                trial_data = json.load(f)
        except Exception as e:
            print(f"  FAIL carrying over {rel}: {e}")
            fail += 1
            continue

        if not args.in_place:
            trial_data["evaluated_model"] = trial_data.get("model_name", args.model)
            trial_data["model_name"] = output_model_name
            out_path = output_base / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(trial_data, f, indent=2)

    if fail_files:
        print(f"Copied {len(fail_files)} operational failure record(s) without scoring them.")

    # Write updated aggregated_results.json for each config
    for config_key, trials_list in config_trials.items():
        agg = rejudge_aggregated(trials_list)
        if args.in_place:
            agg_path = Path(config_key) / "aggregated_results.json"
        else:
            rel_config = Path(config_key).relative_to(model_dir)
            agg_path = output_base / rel_config / "aggregated_results.json"
            agg_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing aggregated if present, update just the re-judged fields
        existing_agg = {}
        orig_agg = Path(config_key) / "aggregated_results.json"
        if orig_agg.exists():
            with open(orig_agg) as f:
                existing_agg = json.load(f)

        if "aggregate" not in existing_agg:
            existing_agg["aggregate"] = {"all_trials": {}}
        existing_agg["aggregate"]["all_trials"].update(agg)
        existing_agg["config"] = existing_agg.get("config", {})
        existing_agg["config"]["LLM judge"] = args.judge
        if not args.in_place:
            existing_agg["config"]["evaluated_model"] = args.model
            existing_agg["config"]["model_name"] = output_model_name

        with open(agg_path, "w") as f:
            json.dump(existing_agg, f, indent=2)

    print(f"\nDone: {success} succeeded, {fail} failed, {changed} changed verdict")
    if not args.in_place:
        print(f"Re-judged results written to: {output_base}/")
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
