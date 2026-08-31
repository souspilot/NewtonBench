"""
Generate a collaborator-facing report: for a model's genuinely-wrong trials
(both the LLM judge and the deterministic sympy check agree the law is
wrong -- see failure_analysis.py), classify HOW each one is wrong via
mismatch_classifier.py, and sample 5-10 representative examples per mistake
type along with how many trials fall into that bucket.

Reuses failure_analysis.py's trial loading (with the same version-directory
deduplication) and agreement-bucket logic, so counts here are consistent with
what failure_analysis.py reports.

Mistake types (see mismatch_classifier.py for the exact definitions):
  missing_variable   - ground truth depends on a parameter the submission doesn't
  sign_flip          - a parameter's exponent has the wrong sign (e.g. 1/d^2 vs d^2)
  extra_variable      - submission depends on a parameter ground truth doesn't
  wrong_exponent     - right variables, wrong magnitude exponent
  other_structural   - no single-parameter elasticity issue found; likely an
                        operator-level difference (e.g. additive vs multiplicative
                        form) that this lens can't isolate -- needs a human look
  not_checkable      - submitted_law has control flow / unsupported ops

Usage:
    python analysis/mistake_taxonomy_report.py --model qwq-32b
    python analysis/mistake_taxonomy_report.py --model qwq-32b --module m0_gravity
    python analysis/mistake_taxonomy_report.py --model qwq-32b --samples 8
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from failure_analysis import load_trials, clean_rmsle_outliers, compute_verdicts, DEFAULT_RMSLE_THRESHOLD  # noqa: E402
from mismatch_classifier import classify_mismatch  # noqa: E402

MISTAKE_TYPE_ORDER = ["missing_variable", "sign_flip", "extra_variable", "wrong_exponent",
                       "other_structural", "not_checkable"]

MISTAKE_TYPE_BLURB = {
    "missing_variable": "Ground truth depends on a parameter the submitted law ignores entirely.",
    "sign_flip": "A parameter's exponent has the wrong SIGN (e.g. law should increase with distance, "
                 "submission has it decreasing, or vice versa) -- not just the wrong magnitude.",
    "extra_variable": "Submitted law depends on a parameter that ground truth doesn't.",
    "wrong_exponent": "Right variables, right sign, but the wrong exponent magnitude "
                       "(e.g. mass^1 submitted where ground truth is mass^2).",
    "other_structural": "No single-parameter exponent mismatch found, but the law is still wrong -- "
                         "likely an operator-level difference (e.g. additive vs multiplicative combination "
                         "of terms) that a per-parameter power-law comparison can't isolate. Needs a manual read.",
    "not_checkable": "Submitted law uses control flow (if/try/etc.) our automated checker can't parse -- "
                      "needs a manual read.",
}


def classify_all(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mistake_types = []
    detail_strs = []
    for _, row in df.iterrows():
        if not isinstance(row["submitted_law"], str) or not isinstance(row["ground_truth_law"], str):
            mistake_types.append("not_checkable")
            detail_strs.append("")
            continue
        result = classify_mismatch(row["submitted_law"], row["ground_truth_law"])
        mistake_types.append(result["mistake_type"])
        issues = [f"{p}: sub={d['sub']}, gt={d['gt']} ({d['issue']})"
                  for p, d in result["details"].items() if d["issue"] != "matches"]
        detail_strs.append("; ".join(issues))
    df["mistake_type"] = mistake_types
    df["mistake_detail"] = detail_strs
    return df


def print_report(df: pd.DataFrame, n_samples: int, model: str):
    print(f"\n{'='*70}\nMistake taxonomy for model: {model}\n{'='*70}")
    print(f"Total consistent_fail trials analyzed: {len(df)}\n")

    counts = df["mistake_type"].value_counts()
    print("=== Counts by mistake type ===")
    for mt in MISTAKE_TYPE_ORDER:
        if mt in counts.index:
            print(f"  {mt:20s} {counts[mt]:4d}   {MISTAKE_TYPE_BLURB[mt]}")
    print()

    print("=== By module x mistake type ===")
    print(pd.crosstab(df["module"], df["mistake_type"]).reindex(
        columns=[m for m in MISTAKE_TYPE_ORDER if m in df["mistake_type"].unique()]).to_string())

    for mt in MISTAKE_TYPE_ORDER:
        subset = df[df["mistake_type"] == mt]
        if subset.empty:
            continue
        print(f"\n{'-'*70}\n{mt.upper()}  ({len(subset)} trials)\n{MISTAKE_TYPE_BLURB[mt]}\n{'-'*70}")
        sample = subset.sample(n=min(n_samples, len(subset)), random_state=0).sort_values(
            ["module", "equation_difficulty", "model_system"])
        for _, row in sample.iterrows():
            print(f"\n  [{row['module']} / {row['equation_difficulty']} / {row['model_system']} / "
                  f"{row['law_version']} / {row['agent_backend']} / trial{row['trial_id']}]")
            if row["mistake_detail"]:
                print(f"    Mismatch: {row['mistake_detail']}")
            print(f"    Ground truth: {row['ground_truth_law']}")
            submitted_oneline = " ".join(str(row["submitted_law"]).split())
            print(f"    Submitted:    {submitted_oneline}")
            print(f"    Path: {row['path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--result_dir", default="evaluation_results")
    parser.add_argument("--module", default=None)
    parser.add_argument("--agent", default=None)
    parser.add_argument("--rmsle_threshold", type=float, default=DEFAULT_RMSLE_THRESHOLD)
    parser.add_argument("--samples", type=int, default=8, help="Examples to sample per mistake type (5-10 typical)")
    parser.add_argument("-o", "--output_csv", default=None,
                         help="Default: analysis/mistake_taxonomy_<model>.csv (full table, not just samples)")
    args = parser.parse_args()

    output_csv = args.output_csv or f"analysis/mistake_taxonomy_{args.model}.csv"

    df = load_trials(args.result_dir, args.model)
    if args.module:
        df = df[df["module"] == args.module]
    if args.agent:
        df = df[df["agent_backend"] == args.agent]

    df = clean_rmsle_outliers(df)
    df = compute_verdicts(df, args.rmsle_threshold)

    fails = df[df["agreement_bucket"] == "consistent_fail"].copy()
    if fails.empty:
        raise SystemExit("No consistent_fail trials found for this model/filter -- nothing to classify.")

    fails = classify_all(fails)
    print_report(fails, args.samples, args.model)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    cols = ["path", "module", "equation_difficulty", "model_system", "law_version", "agent_backend",
            "trial_id", "rmsle", "mistake_type", "mistake_detail", "submitted_law", "ground_truth_law"]
    fails[cols].to_csv(output_csv, index=False)
    print(f"\n\nFull classified table ({len(fails)} rows) written to {output_csv}")