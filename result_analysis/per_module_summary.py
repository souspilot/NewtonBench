"""
Per-module results summary, in the same shape as the paper's Appendix B.1
per-domain tables (e.g. Figure 8 for Gravitation, Figure 10 for Magnetostatics).

WHY THIS EXISTS: result_analysis/summarize_results.py's aggregate_results() only
groups by (model_name, agent_backend) -- every module present in
results_by_trial.csv gets pooled into a single acc_<difficulty>_<system> cell.
That reproduces Table 2's shape (one row per model, averaged across all domains),
not Appendix B.1's shape (one table per domain). Since a representative-subset run
covers a DIFFERENT pair of (difficulty, system) cells per module (see
configs/representative_subset.json), pooling across modules would silently mix
cells from different domains into the same column -- not comparable to any single
number in the paper.

This script reuses the exact same methodology as summarize_results.py (the same
Modified Z-Score RMSLE outlier filtering, grouped the same way, and the same
trial-then-mean/std aggregation via calculate_trial_stats) but adds `module` to
the final grouping, so you get one row per (module, agent_backend) -- directly
comparable to a single cell in the matching Appendix B.1 figure.

Usage:
    # First, make sure results_by_trial.csv is up to date (same as summarize_results.py)
    python result_analysis/summarize_results.py -m qwq-32b

    # Then run this to get the per-module breakdown
    python result_analysis/per_module_summary.py --model qwq-32b

    # Restrict to one module, or one backend
    python result_analysis/per_module_summary.py --model qwq-32b --module m0_gravity
    python result_analysis/per_module_summary.py --model qwq-32b --agent vanilla_agent

    # Only show cells your subset actually ran (skips all-N/A rows/columns)
    python result_analysis/per_module_summary.py --model qwq-32b --subset_file configs/representative_subset.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse summarize_results.py's exact aggregation logic rather than reimplementing it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "result_analysis"))
from summarize_results import detect_outliers_modified_zscore_column, calculate_trial_stats  # noqa: E402

DIFFICULTIES = ["easy", "medium", "hard"]
SYSTEMS = ["vanilla_equation", "simple_system", "complex_system"]


def load_trials(csv_path: str, model: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df["model_name"] == model]
    if df.empty:
        raise SystemExit(f"No rows for model_name == '{model}' in {csv_path}. "
                          f"Did you run result_analysis/summarize_results.py -m {model} first?")
    return df


def clean_rmsle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Same grouping as summarize_results.py's outlier removal: per
    (module, equation_difficulty, model_system, agent_backend) cell."""
    groups = df.groupby(["module", "equation_difficulty", "model_system", "agent_backend"])
    cleaned = [detect_outliers_modified_zscore_column(g.copy(), "rmsle") for _, g in groups]
    return pd.concat(cleaned).reset_index(drop=True) if cleaned else df


def load_subset_cells(subset_file: str):
    """Return {module: {(difficulty, system), ...}} from a representative_subset.json,
    or None if not provided/found (meaning: show every cell)."""
    if not subset_file or not Path(subset_file).exists():
        return None
    with open(subset_file) as f:
        raw = json.load(f)
    return {m: {(c["difficulty"], c["system"]) for c in cells} for m, cells in raw.items()}


def build_table(df: pd.DataFrame, subset_cells) -> pd.DataFrame:
    rows = []
    modules = sorted(df["module"].dropna().unique())
    backends = sorted(df["agent_backend"].dropna().unique())

    for module in modules:
        for backend in backends:
            mb_df = df[(df["module"] == module) & (df["agent_backend"] == backend)]
            if mb_df.empty:
                continue
            allowed_cells = subset_cells.get(module) if subset_cells else None

            row = {"module": module, "agent_backend": backend}
            for system in SYSTEMS:
                for difficulty in DIFFICULTIES:
                    col = f"acc_{difficulty}_{system}"
                    if allowed_cells is not None and (difficulty, system) not in allowed_cells:
                        row[col] = "-"  # not part of the subset for this module -- not just "no data"
                        continue
                    cell_df = mb_df[(mb_df["equation_difficulty"] == difficulty) & (mb_df["model_system"] == system)]
                    mean_acc, std_acc, mean_rmsle, _ = calculate_trial_stats(cell_df)
                    if pd.notna(mean_acc):
                        row[col] = f"{mean_acc*100:.1f}"
                        row[col.replace("acc_", "rmsle_")] = f"{mean_rmsle:.4f}" if pd.notna(mean_rmsle) else "N/A"
                    else:
                        row[col] = "N/A"
                        row[col.replace("acc_", "rmsle_")] = "N/A"

            overall_acc, overall_std, overall_rmsle, _ = calculate_trial_stats(mb_df)
            row["overall_acc"] = f"{overall_acc*100:.1f} (±{(overall_std or 0)*100:.3f})" if pd.notna(overall_acc) else "N/A"
            row["overall_rmsle"] = f"{overall_rmsle:.4f}" if pd.notna(overall_rmsle) else "N/A"
            row["n_trials"] = len(mb_df)
            rows.append(row)

    acc_cols = [f"acc_{d}_{s}" for s in SYSTEMS for d in DIFFICULTIES]
    rmsle_cols = [f"rmsle_{d}_{s}" for s in SYSTEMS for d in DIFFICULTIES]
    col_order = ["module", "agent_backend"] + acc_cols + rmsle_cols + ["overall_acc", "overall_rmsle", "n_trials"]
    return pd.DataFrame(rows).reindex(columns=col_order)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model_name to summarize (matches the 'model_name' column in results_by_trial.csv)")
    parser.add_argument("--module", default=None, help="Restrict to one module (e.g. m0_gravity)")
    parser.add_argument("--agent", default=None, help="Restrict to one agent_backend (vanilla_agent / code_assisted_agent)")
    parser.add_argument("--csv", default="result_analysis/results_by_trial.csv", help="Path to results_by_trial.csv")
    parser.add_argument("--subset_file", default=None,
                         help="Path to representative_subset.json -- when given, cells outside the subset "
                              "for a given module are marked '-' instead of 'N/A' (distinguishing 'not run "
                              "by design' from 'ran but produced no valid trials').")
    parser.add_argument("-o", "--output_csv", default="result_analysis/per_module_summary.csv")
    args = parser.parse_args()

    df = load_trials(args.csv, args.model)
    if args.module:
        df = df[df["module"] == args.module]
        if df.empty:
            raise SystemExit(f"No rows for module '{args.module}' and model '{args.model}'.")
    if args.agent:
        df = df[df["agent_backend"] == args.agent]
        if df.empty:
            raise SystemExit(f"No rows for agent_backend '{args.agent}'.")

    df = clean_rmsle_outliers(df)
    subset_cells = load_subset_cells(args.subset_file)
    table = build_table(df, subset_cells)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(table.to_string(index=False))
    print(f"\nWrote {args.output_csv}")