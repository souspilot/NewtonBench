"""
Judge-independent failure analysis for a model's NewtonBench trials.

Combines three independent signals per trial, so no single one has to be
trusted blindly:

  1. judge_verdict      -- symbolic_equivalent, as decided by the LLM judge
                            (self-judged for models where judge_model_name ==
                            model_name -- the thing you don't fully trust).
  2. rmsle_verdict       -- purely numeric: is RMSLE below a "this is
                            essentially an exact fit" threshold? Independent of
                            any LLM. Outlier-cleaned the same way
                            summarize_results.py does (Modified Z-Score per
                            (module, difficulty, system, agent_backend) group)
                            so a single bad trial doesn't get miscompared
                            against a threshold tuned for typical trials.
  3. structural_verdict  -- from structural_equivalence.py: deterministic
                            sympy check of whether submitted_law has the same
                            functional form as ground_truth_law up to a
                            constant factor. No LLM, no numeric threshold --
                            pure symbolic manipulation. Falls back to
                            'not_checkable' for anything with control flow
                            rather than guessing.

The interesting rows are where these DISAGREE:

  - judge=True,  rmsle=False -> "judge_lenient": judge said correct but the
    numbers don't back it up (e.g. right form, catastrophically wrong
    constant -- see the qwen38-27b 6.674e-55 case earlier in this project).
    structural_verdict helps disambiguate: constant_equivalent here CONFIRMS
    "right form, bad constant" (numerically bad, but arguably still a
    legitimate symbolic-accuracy credit per the paper's own "ignore
    constants" rule); structurally_different here is a much more serious
    judge error -- the judge called something equivalent that sympy says
    genuinely isn't.

  - judge=False, rmsle=True  -> "judge_strict": judge said wrong but RMSLE is
    essentially exact. Possible false negative -- an algebraically equivalent
    but differently-written formula the judge failed to recognize.

Usage:
    python analysis/failure_analysis.py --model qwq-32b
    python analysis/failure_analysis.py --model qwq-32b --module m0_gravity
    python analysis/failure_analysis.py --model qwq-32b --rmsle_threshold 1e-3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "result_analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_results import detect_outliers_modified_zscore_column  # noqa: E402
from structural_equivalence import check_constant_equivalence  # noqa: E402

DEFAULT_RMSLE_THRESHOLD = 1e-3  # well above float-precision noise (~1e-16), well below a
                                  # genuinely-wrong-constant trial's RMSLE (order 1+ in our examples)


def load_trials(result_dir: str, model: str) -> pd.DataFrame:
    model_dir = Path(result_dir) / model
    if not model_dir.is_dir():
        raise SystemExit(f"No such directory: {model_dir}")

    rows = []
    for trials_dir in model_dir.rglob("trials"):
        for trial_path in sorted(trials_dir.glob("trial*.json")):
            if trial_path.name.endswith("_fail.json"):
                continue
            try:
                with open(trial_path) as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Skipping unreadable {trial_path}: {e}")
                continue

            ev = data.get("evaluation", {}) or {}
            rows.append(dict(
                path=str(trial_path),
                trial_id=data.get("trial_id"),
                module=data.get("module_name"),
                equation_difficulty=data.get("equation_difficulty"),
                model_system=data.get("model_system"),
                law_version=data.get("law_version"),
                agent_backend=data.get("agent_backend"),
                rmsle=ev.get("rmsle"),
                exact_accuracy=ev.get("exact_accuracy"),
                symbolic_equivalent=ev.get("symbolic_equivalent"),
                symbolic_msg=ev.get("symbolic_msg"),
                submitted_law=data.get("submitted_law"),
                ground_truth_law=ev.get("ground_truth_law"),
                rounds=data.get("rounds"),
                num_experiments=data.get("num_experiments"),
                total_tokens=data.get("total_tokens"),
            ))
    if not rows:
        raise SystemExit(f"No trial files found under {model_dir}")
    df = pd.DataFrame(rows)
    df["rmsle"] = df["rmsle"].replace([np.inf, -np.inf], np.nan)
    return df


def clean_rmsle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rmsle_cleaned"] = df["rmsle"]
    groups = df.groupby(["module", "equation_difficulty", "model_system", "agent_backend"])
    cleaned = []
    for _, g in groups:
        g2 = g.copy()
        g2["rmsle_cleaned"] = g2["rmsle"]
        g2 = detect_outliers_modified_zscore_column(g2, "rmsle_cleaned")
        cleaned.append(g2)
    return pd.concat(cleaned).reset_index(drop=True)


def compute_verdicts(df: pd.DataFrame, rmsle_threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["judge_verdict"] = df["symbolic_equivalent"].fillna(False).astype(bool)
    # Use the UN-cleaned rmsle for this trial's own verdict -- outlier cleaning is for
    # aggregate stats, not for judging one trial's own fit quality.
    df["rmsle_verdict"] = df["rmsle"] < rmsle_threshold

    structural = []
    for _, row in df.iterrows():
        if not isinstance(row["submitted_law"], str) or not isinstance(row["ground_truth_law"], str):
            structural.append("not_checkable")
            continue
        structural.append(check_constant_equivalence(row["submitted_law"], row["ground_truth_law"]))
    df["structural_verdict"] = structural

    def bucket(row):
        j, r = row["judge_verdict"], row["rmsle_verdict"]
        if j and r:
            return "consistent_pass"
        if not j and not r:
            return "consistent_fail"
        if j and not r:
            return "judge_lenient"
        return "judge_strict"
    df["agreement_bucket"] = df.apply(bucket, axis=1)
    return df


def print_summary(df: pd.DataFrame):
    print(f"\nTotal trials analyzed: {len(df)}\n")

    print("=== Agreement bucket counts ===")
    print(df["agreement_bucket"].value_counts().to_string())

    print("\n=== judge_lenient trials, broken down by structural_verdict ===")
    lenient = df[df["agreement_bucket"] == "judge_lenient"]
    if lenient.empty:
        print("(none)")
    else:
        print(lenient["structural_verdict"].value_counts().to_string())
        print("\n  constant_equivalent = right form, bad constant (numerically off, arguably still")
        print("    deserves symbolic credit per the paper's own 'ignore constants' rule)")
        print("  structurally_different = judge called it equivalent but sympy disagrees --")
        print("    the more concerning kind of judge error")
        print("  not_checkable = control flow / unsupported ops in submitted_law, needs manual review")

    print("\n=== judge_strict trials (possible judge false negatives) ===")
    strict = df[df["agreement_bucket"] == "judge_strict"]
    if strict.empty:
        print("(none)")
    else:
        print(strict["structural_verdict"].value_counts().to_string())

    print("\n=== Breakdown by module ===")
    print(pd.crosstab(df["module"], df["agreement_bucket"]).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--result_dir", default="evaluation_results")
    parser.add_argument("--module", default=None)
    parser.add_argument("--agent", default=None)
    parser.add_argument("--rmsle_threshold", type=float, default=DEFAULT_RMSLE_THRESHOLD)
    parser.add_argument("-o", "--output_csv", default="analysis/failure_analysis.csv")
    args = parser.parse_args()

    df = load_trials(args.result_dir, args.model)
    if args.module:
        df = df[df["module"] == args.module]
    if args.agent:
        df = df[df["agent_backend"] == args.agent]

    df = clean_rmsle_outliers(df)
    df = compute_verdicts(df, args.rmsle_threshold)

    print_summary(df)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    cols = ["path", "module", "equation_difficulty", "model_system", "law_version", "agent_backend",
            "trial_id", "rmsle", "rmsle_cleaned", "exact_accuracy", "judge_verdict", "rmsle_verdict",
            "structural_verdict", "agreement_bucket", "symbolic_msg", "submitted_law", "ground_truth_law"]
    df[cols].to_csv(args.output_csv, index=False)
    print(f"\nFull per-trial table written to {args.output_csv}")
    print("Sort/filter that CSV by agreement_bucket to pull up specific trials for manual review "
          "(the 'path' column points straight at the trial JSON, and 'submitted_law'/'ground_truth_law' "
          "are inlined so you often won't even need to open it).")