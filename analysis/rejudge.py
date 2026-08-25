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
    # Re-judge all qwq-32b results with the paper's judge (gpt41) -- this is the
    # judge run_experiments.py used before your self-judge hack, and is what you
    # need for numbers to be comparable to Table 2 / Appendix B.1. Pass a
    # different --judge only if you deliberately want a non-paper comparison.
    python rejudge.py --model qwq-32b --judge gpt41

    # Re-judge only one module
    python rejudge.py --model qwq-32b --judge gpt41 --module m0_gravity

    # Dry run: show what would be re-judged
    python rejudge.py --model qwq-32b --judge gpt41 --dry-run

    # Overwrite original files instead of creating new ones
    python rejudge.py --model qwq-32b --judge gpt41 --in-place
"""

import argparse
import importlib
import json
import os
import glob
import numpy as np
import traceback
from pathlib import Path


PAPER_JUDGE_MODEL = "gpt41"  # judge_model_name used by upstream run_experiments.py / the paper's Table 2 & Appendix B.1


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
    parser.add_argument("--judge", default=PAPER_JUDGE_MODEL,
                        help=f"LLM model to use as judge (default: '{PAPER_JUDGE_MODEL}', matching what "
                             f"upstream run_experiments.py / the paper's Table 2 and Appendix B.1 used -- "
                             f"pass a different value only if you deliberately want a non-paper-comparable judge).")
    parser.add_argument("--base-dir", default="evaluation_results")
    parser.add_argument("--module", default=None, help="Restrict to one module")
    parser.add_argument("--agent", default=None, help="Restrict to one agent backend")
    parser.add_argument("--dry-run", action="store_true", help="Just count trials, don't re-judge")
    parser.add_argument("--in-place", action="store_true",
                        help="Overwrite original trial JSONs. Default: write to a parallel directory.")
    parser.add_argument("--output-suffix", default=None,
                        help="Suffix for output dir (default: judge model name)")
    args = parser.parse_args()

    model_dir = Path(args.base_dir) / args.model
    if not model_dir.exists():
        print(f"ERROR: {model_dir} not found")
        return

    suffix = args.output_suffix or args.judge
    output_base = Path(args.base_dir) / f"{args.model}_judged_by_{suffix}"

    # Find all trial JSON files, split into ones that need re-judging vs. failed
    # trials that should still count toward the aggregate but don't need an API
    # call (their submitted_law stub, "return float('nan')", can never be
    # symbolically equivalent to anything -- re-judging would just re-confirm
    # exact_accuracy == 0.0 at the cost of a wasted judge call).
    trial_files = []
    fail_files = []
    for trial_path in sorted(model_dir.rglob("trial*.json")):
        if "_chat_history" in trial_path.name:
            continue
        # Filter by module if specified
        if args.module and args.module not in str(trial_path):
            continue
        # Filter by agent if specified
        if args.agent and args.agent not in str(trial_path):
            continue
        if trial_path.name.endswith("_fail.json"):
            fail_files.append(trial_path)
        else:
            trial_files.append(trial_path)

    print(f"Found {len(trial_files)} trial files to re-judge "
          f"(+ {len(fail_files)} already-failed trials carried into the aggregate unchanged, not re-judged)")
    print(f"  Model: {args.model}")
    print(f"  Judge: {args.judge}")
    if args.judge != PAPER_JUDGE_MODEL:
        print(f"  WARNING: judge '{args.judge}' != paper's judge '{PAPER_JUDGE_MODEL}' -- results won't be "
              f"directly comparable to Table 2 / Appendix B.1 (you'll have swapped a self-judging bias for "
              f"a different judge-model mismatch). Use --judge {PAPER_JUDGE_MODEL} for paper-comparable numbers.")
    if args.module:
        print(f"  Module filter: {args.module}")
    if args.agent:
        print(f"  Agent filter: {args.agent}")

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
            with open(trial_path) as f:
                trial_data = json.load(f)

            old_acc = trial_data.get("evaluation", {}).get("exact_accuracy", 0.0)

            new_eval = rejudge_trial(trial_data, args.judge)
            if new_eval is None:
                print("SKIP (missing data)")
                continue

            new_acc = new_eval.get("exact_accuracy", 0.0)

            # Update the trial data (capture the ORIGINAL judge before overwriting it)
            original_judge = trial_data.get("LLM judge", "unknown")
            trial_data["evaluation"] = new_eval
            trial_data["original_judge"] = original_judge
            trial_data["LLM judge"] = args.judge

            # Write output
            if args.in_place:
                out_path = trial_path
            else:
                out_path = output_base / rel
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

    # Carry already-failed trials into the aggregate unchanged -- they still count
    # toward num_total_trials / average_exact_accuracy (as 0.0), matching how
    # run_experiments.py's original aggregation includes them, but they were never
    # re-judged so no judge/API call was spent on them.
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
            out_path = output_base / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(trial_data, f, indent=2)

        config_key = str(config_dir)
        config_trials.setdefault(config_key, []).append(trial_data)

    if fail_files:
        print(f"Carried over {len(fail_files)} already-failed trial(s) into the aggregate unchanged "
              f"(no judge call spent on them).")

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

        with open(agg_path, "w") as f:
            json.dump(existing_agg, f, indent=2)

    print(f"\nDone: {success} succeeded, {fail} failed, {changed} changed verdict")
    if not args.in_place:
        print(f"Re-judged results written to: {output_base}/")


if __name__ == "__main__":
    main()