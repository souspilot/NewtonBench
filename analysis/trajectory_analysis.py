"""
Trial-level "resource vs. outcome" analysis: does accuracy correlate with
tokens used, number of real experiments run, or rounds used before
submitting <final_law> -- and does the model tend to answer prematurely
rather than exhausting its round budget?

Deliberately trial-level, not config-averaged: reuses failure_analysis.py's
load_trials()/compute_verdicts() (same-directory import, same version-
directory dedup) so every row is one real trial, not an already-averaged
per-config number from the benchmark-completion logs.

Two things this does differently from a naive version of this analysis:

  1. include_fails=True. failure_analysis.py's load_trials() normally skips
     *_fail.json (round-limit-exhausted trials) -- reasonable for structural
     mismatch classification, since their stub "return float('nan')"
     submissions are noise there, but wrong here: a trial that burned its
     full round budget and still failed is exactly the data point a "does
     hitting the round limit predict failure" question needs. Dropping it
     would bias the whole analysis toward only trials that finished cleanly.

  2. "success" has both a raw and a verified version. `raw_success` is just
     the LLM judge's own exact_accuracy (self-judged for models where
     judge_model_name == model_name). `verified_success` additionally
     requires sympy's structural check to agree (agreement_bucket ==
     'consistent_pass') -- catching judge-lenient false credit that would
     otherwise inflate the correlation between "used more resources" and
     "succeeded." Both are reported so you can see how much they diverge,
     but verified_success is what drives the tables and correlations by
     default (--success_metric raw to switch).

Termination is read from the trial JSON's own `status` field
('completed' vs 'max_turns_reached') rather than inferred from rounds alone
-- a trial can legitimately take exactly MAX_TURNS rounds and finish
voluntarily, which a bare `rounds >= MAX_TURNS` check can't distinguish from
being cut off.

Usage:
    python analysis/trajectory_analysis.py --model qwq-32b
    python analysis/trajectory_analysis.py --model qwq-32b --agent vanilla_agent
    python analysis/trajectory_analysis.py --model qwq-32b --module m5_radioactive_decay
    python analysis/trajectory_analysis.py --model qwq-32b --success_metric raw
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from failure_analysis import load_trials, clean_rmsle_outliers, compute_verdicts, DEFAULT_RMSLE_THRESHOLD  # noqa: E402

MAX_TURNS = 10  # every module's system prompt states "up to 10 rounds"


def bin_and_report(df: pd.DataFrame, col: str, bins, labels, success_col: str) -> pd.DataFrame:
    d = df.dropna(subset=[col]).copy()
    d["_bin"] = pd.cut(d[col], bins=bins, labels=labels, include_lowest=True)
    out = d.groupby("_bin", observed=True).agg(
        n=(success_col, "size"),
        success_rate_pct=(success_col, "mean"),
    )
    out["success_rate_pct"] = (out["success_rate_pct"] * 100).round(1)
    return out


def report(df: pd.DataFrame, label: str, success_col: str):
    print(f"\n{'='*78}\n{label}   (n={len(df)}, {success_col}={100*df[success_col].mean():.1f}%)\n{'='*78}")

    # --- raw vs verified divergence, if both are available ---
    if "raw_success" in df.columns and "verified_success" in df.columns:
        flipped = (df["raw_success"] != df["verified_success"]).sum()
        if flipped:
            print(f"\n[Note: {flipped}/{len(df)} trials differ between raw judge accuracy and the "
                  f"sympy-verified success label -- see failure_analysis.py's agreement_bucket for why.]")

    # --- 1. num_experiments, including a zero-experiment call-out ---
    print("\n--- Success rate vs. number of real experiments run ---")
    exp_bins = [-0.5, 0.5, 2.5, 5.5, 1e9]
    exp_labels = ["0", "1-2", "3-5", "6+"]
    print(bin_and_report(df, "num_experiments", exp_bins, exp_labels, success_col).to_string())

    zero_exp = df[df["num_experiments"] == 0]
    if len(zero_exp):
        print(f"\nZero-experiment trials (never ran a single <run_experiment> before answering): "
              f"{len(zero_exp)}/{len(df)} ({100*len(zero_exp)/len(df):.1f}%), "
              f"{success_col} among them: {100*zero_exp[success_col].mean():.1f}%")
        print(zero_exp.groupby(["module", "agent_backend"], observed=True).size().to_string())
    else:
        print("\nNo zero-experiment trials found.")

    # --- 2. rounds used + termination status (from the trial's own recorded status, not inferred) ---
    print("\n--- Success rate vs. rounds used ---")
    round_bins = [-0.5, 1.5, 3.5, 6.5, 9.5, MAX_TURNS + 0.5]
    round_labels = ["1", "2-3", "4-6", "7-9", f"{MAX_TURNS} (max)"]
    print(bin_and_report(df, "rounds", round_bins, round_labels, success_col).to_string())

    print("\n--- Success rate by termination status (trial JSON's own 'status' field) ---")
    if df["status"].notna().any():
        st = df.groupby("status", dropna=False, observed=True).agg(
            n=(success_col, "size"),
            success_rate_pct=(success_col, "mean"),
            avg_tokens=("total_tokens", "mean"),
            avg_experiments=("num_experiments", "mean"),
        )
        st["success_rate_pct"] = (st["success_rate_pct"] * 100).round(1)
        print(st.to_string())
        print("  max_turns_reached + LOWER success: the round budget is a real constraint --")
        print("    these trials were still searching when cut off, more rounds might help.")
        print("  max_turns_reached + success NOT lower: these trials use every round regardless")
        print("    of whether they're converging -- more rounds alone likely won't fix them.")
    else:
        print("  ('status' field not present in these trial JSONs -- falling back to rounds>=max)")
        st = df.copy()
        st["hit_limit"] = st["rounds"] >= MAX_TURNS
        t = st.groupby("hit_limit", observed=True).agg(n=(success_col, "size"), success_rate_pct=(success_col, "mean"))
        t["success_rate_pct"] = (t["success_rate_pct"] * 100).round(1)
        print(t.to_string())

    premature = df[(df["status"] != "max_turns_reached") & (df["rounds"] <= 2)]
    if len(premature):
        print(f"\nSubmitted <final_law> within the first 2 rounds, without being cut off: "
              f"{len(premature)}/{len(df)} ({100*len(premature)/len(df):.1f}%), "
              f"{success_col} among them: {100*premature[success_col].mean():.1f}%")
        print(premature.groupby(["module", "agent_backend"], observed=True).size().to_string())
        rest = df[~df.index.isin(premature.index)]
        print(f"  vs. everyone else: {100*rest[success_col].mean():.1f}% (n={len(rest)})")
        print("  Much lower here: premature submission is a real failure mode -- worth a")
        print("    minimum-rounds nudge or a self-verification gate before <final_law>.")
        print("  Comparable or higher here: quick-and-correct is common, not premature guessing.")

    # --- 3. tokens used (quantile bins -- adapts to this model's actual usage range) ---
    print("\n--- Success rate vs. total tokens used (quartiles) ---")
    try:
        tok_q = df["total_tokens"].quantile([0, .25, .5, .75, 1.0]).tolist()
        if len(set(tok_q)) < 5:
            raise ValueError("degenerate quantiles")
        tok_labels = ["Q1 (fewest)", "Q2", "Q3", "Q4 (most)"]
        print(bin_and_report(df, "total_tokens", tok_q, tok_labels, success_col).to_string())
    except (ValueError, IndexError):
        print("Not enough spread in total_tokens to quartile-bin (falling back to fixed bins).")
        fixed_bins = [0, 3000, 6000, 10000, 20000, 40000, 1e9]
        fixed_labels = ["0-3k", "3-6k", "6-10k", "10-20k", "20-40k", "40k+"]
        print(bin_and_report(df, "total_tokens", fixed_bins, fixed_labels, success_col).to_string())

    # --- 4. correlations, with the linear-only caveat spelled out ---
    print(f"\n--- Correlation of trajectory features with {success_col} (Pearson r vs. 0/1 outcome) ---")
    for col in ["num_experiments", "rounds", "total_tokens"]:
        sub = df[[col, success_col]].dropna()
        if len(sub) < 3 or sub[col].std() == 0:
            print(f"  {col:16s}: (insufficient variation)")
            continue
        r = sub[col].corr(sub[success_col])
        print(f"  {col:16s}: r = {r:+.3f}   (n={len(sub)})")
    print("  CAVEAT: Pearson r only captures LINEAR relationships. A pattern where success")
    print("  rises then falls (e.g. peaks mid-range, drops for trials that hit the round")
    print("  limit) can show a weak/near-zero r despite a real, strong effect -- always")
    print("  check the binned tables above before concluding a resource 'doesn't matter'.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--result_dir", default="evaluation_results")
    ap.add_argument("--agent", choices=["vanilla_agent", "code_assisted_agent"], default=None,
                     help="Restrict to one agent backend. Omit to see both, plus a per-agent split.")
    ap.add_argument("--module", default=None, help="Restrict to one module, e.g. m5_radioactive_decay.")
    ap.add_argument("--rmsle_threshold", type=float, default=DEFAULT_RMSLE_THRESHOLD)
    ap.add_argument("--success_metric", choices=["raw", "verified"], default="verified",
                     help="'verified' (default): judge AND sympy structural check both agree it's "
                           "correct -- filters out judge-lenient false credit. 'raw': the LLM judge's "
                           "own exact_accuracy, unfiltered -- matches what the live benchmark run recorded.")
    ap.add_argument("-o", "--output_csv", default=None,
                     help="Default: analysis/trajectory_analysis_<model>.csv")
    args = ap.parse_args()

    output_csv = args.output_csv or f"analysis/trajectory_analysis_{args.model}.csv"

    df = load_trials(args.result_dir, args.model, include_fails=True)
    df = clean_rmsle_outliers(df)
    df = compute_verdicts(df, args.rmsle_threshold)

    df["raw_success"] = df["exact_accuracy"].fillna(0.0) >= 0.5
    df["verified_success"] = df["agreement_bucket"] == "consistent_pass"
    success_col = "verified_success" if args.success_metric == "verified" else "raw_success"

    if args.module:
        df = df[df["module"] == args.module]
    if args.agent:
        df = df[df["agent_backend"] == args.agent]
    if df.empty:
        raise SystemExit("No trials match the given filters.")

    label = f"Trajectory analysis: {args.model}"
    if args.module:
        label += f" / {args.module}"
    if args.agent:
        label += f" / {args.agent}"
    report(df, label, success_col)

    if args.agent is None:
        print(f"\n\n{'#'*78}\n# Split by agent_backend\n{'#'*78}")
        for backend, g in df.groupby("agent_backend", observed=True):
            report(g, f"{args.model} / {backend}", success_col)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    cols = ["path", "module", "equation_difficulty", "model_system", "law_version", "agent_backend",
            "trial_id", "is_fail", "status", "rounds", "num_experiments", "total_tokens",
            "raw_success", "verified_success", "agreement_bucket", "structural_verdict"]
    df[cols].to_csv(output_csv, index=False)
    print(f"\n\nFull per-trial table written to {output_csv}")


if __name__ == "__main__":
    main()