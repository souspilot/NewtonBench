"""
Chat-history ("trace") mining for a model's NewtonBench trials.

trajectory_analysis.py only looks at the scalar summary fields a trial JSON
records (rounds, num_experiments, total_tokens, status). That tells you
*that* high-token trials fail, not *why*. This script re-opens each trial's
`chat_history` and classifies what actually happened turn by turn, so the
failure modes behind the correlations become visible and countable:

  * format_failures   -- turns where the harness had to reply "Invalid
                          response / Action Reminder" because the model's
                          <run_experiment> / <python> / <final_law> block
                          didn't parse. Wasted rounds that look like
                          "thinking" in the token count.
  * never_experimented -- submitted (or was cut off) without a single
                          successful <run_experiment>. Split into
                          "answered from priors" (no format failures, just
                          guessed) vs "locked out by format failures".
  * hypothesis_churn   -- number of DISTINCT `def discovered_law` bodies the
                          model floated across the whole trajectory. High
                          churn = never converged.
  * unverified_submit  -- the exact functional form it submitted never
                          appeared in an earlier turn that was followed by an
                          <experiment_output> (i.e. it was never checked
                          against data before being finalised).
  * reasoning_blowup   -- total reasoning-block characters, and the single
                          largest assistant message. QwQ-style runaway CoT.
  * nan_flood          -- fraction of returned experiment data points that
                          were `nan` (bad input ranges the model kept using).
  * python_errors      -- (code_assisted only) <python_output> blocks
                          containing a traceback.

Each feature is then cross-tabulated against verified_success, and a
per-feature mean-by-outcome table + point-biserial correlations are printed,
so you can see which behaviours actually separate wins from losses.

Reuses failure_analysis.load_trials/compute_verdicts (same dedup + verified-
success label as trajectory_analysis.py) and joins on the trial `path`.

Usage:
    python analysis/trajectory_trace_analysis.py --model qwq-32b
    python analysis/trajectory_trace_analysis.py --model qwq-32b --agent code_assisted_agent
    python analysis/trajectory_trace_analysis.py --model qwq-32b --dump-examples format_failures
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from failure_analysis import (  # noqa: E402
    load_trials, clean_rmsle_outliers, compute_verdicts, DEFAULT_RMSLE_THRESHOLD,
)

MAX_TURNS = 10
_INVALID_MARKERS = (
    "Invalid response",
    "Action Reminder",
    "exactly 1 action per turn",
    "Please use <run_experiment>",
)
_LAW_BODY_RE = re.compile(r"def\s+discovered_law\s*\([^)]*\)\s*:(.*?)(?=\ndef\s|\Z)", re.DOTALL)
_RETURN_RE = re.compile(r"return\s+(.+)")


def _split_reasoning(content: str):
    """Assistant messages are stored as '**Reasoning Process:**\\n...\\n\\n**Main Response:**\\n...'
    when the provider returned a separate reasoning field, else just the response."""
    if "**Main Response:**" in content:
        head, _, tail = content.partition("**Main Response:**")
        return head.replace("**Reasoning Process:**", "").strip(), tail.strip()
    return "", content.strip()


def _norm_law(body: str) -> str:
    """Normalise a discovered_law body to its return expression, whitespace-collapsed,
    so cosmetically-different restatements of the same formula collapse together."""
    m = _RETURN_RE.search(body)
    expr = m.group(1) if m else body
    return re.sub(r"\s+", "", expr).rstrip(")")[:400]


def analyse_trace(chat_history):
    assistant = [m.get("content", "") or "" for m in chat_history if m.get("role") == "assistant"]
    user = [m.get("content", "") or "" for m in chat_history if m.get("role") == "user"]

    reasonings, responses = [], []
    for c in assistant:
        r, resp = _split_reasoning(c)
        reasonings.append(r)
        responses.append(resp)

    format_failures = sum(any(mk in u for mk in _INVALID_MARKERS) for u in user)
    experiment_outputs = [u for u in user if "<experiment_output>" in u]
    python_outputs = [u for u in user if "<python_output>" in u]

    # nan flood: over all returned experiment data
    nan_count = tot_count = 0
    for u in experiment_outputs:
        toks = re.findall(r"-?\d+\.?\d*e?-?\d*|nan|NaN", u)
        tot_count += len(toks)
        nan_count += sum(t.lower() == "nan" for t in toks)

    python_errors = sum(("Traceback" in u or "Error:" in u) for u in python_outputs)

    # candidate laws floated anywhere (churn) + which turn each first appeared
    seen, first_turn = {}, {}
    for i, resp in enumerate(responses):
        for body in _LAW_BODY_RE.findall(resp):
            key = _norm_law(body)
            if key and key not in seen:
                seen[key] = body
                first_turn[key] = i

    # did the model ever successfully run an experiment?
    ran_experiment = len(experiment_outputs) > 0
    first_exp_turn = next((i for i, resp in enumerate(responses) if "<run_experiment>" in resp), np.nan)

    # unverified submit: final law's form never floated before an experiment came back
    final_key = None
    for resp in reversed(responses):
        bodies = _LAW_BODY_RE.findall(resp)
        if bodies:
            final_key = _norm_law(bodies[-1])
            break
    unverified_submit = True
    if final_key is not None and final_key in first_turn:
        # was there at least one experiment_output strictly after the turn it first appeared?
        appeared_turn = first_turn[final_key]
        exp_turns = [i for i, resp in enumerate(responses) if "<run_experiment>" in resp]
        unverified_submit = not any(t >= appeared_turn for t in exp_turns) if ran_experiment else True
    if not ran_experiment:
        unverified_submit = True

    reasoning_chars = sum(len(r) for r in reasonings)
    max_msg_chars = max((len(c) for c in assistant), default=0)
    empty_assistant = sum(len(c.strip()) == 0 for c in assistant)

    return dict(
        assistant_turns=len(assistant),
        format_failures=format_failures,
        had_format_failure=format_failures > 0,
        ran_experiment=ran_experiment,
        first_exp_turn=first_exp_turn,
        n_experiment_batches=len(experiment_outputs),
        n_python_calls=len(python_outputs),
        python_errors=python_errors,
        nan_fraction=(nan_count / tot_count) if tot_count else 0.0,
        hypothesis_churn=len(seen),
        unverified_submit=unverified_submit,
        reasoning_chars=reasoning_chars,
        max_msg_chars=max_msg_chars,
        empty_assistant_msgs=empty_assistant,
    )


def _load_traces(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for path in df["path"]:
        try:
            data = json.load(open(path))
            recs.append({"path": path, **analyse_trace(data.get("chat_history", []))})
        except Exception as e:  # noqa: BLE001
            print(f"  (skipped {path}: {e})")
            recs.append({"path": path})
    return pd.DataFrame(recs)


NUMERIC_FEATURES = [
    "assistant_turns", "format_failures", "first_exp_turn", "n_experiment_batches",
    "n_python_calls", "python_errors", "nan_fraction", "hypothesis_churn",
    "reasoning_chars", "max_msg_chars", "empty_assistant_msgs",
]
BOOL_FEATURES = ["had_format_failure", "ran_experiment", "unverified_submit"]


def report(df: pd.DataFrame, label: str):
    n = len(df)
    sr = 100 * df["verified_success"].mean()
    print(f"\n{'='*78}\n{label}   (n={n}, verified_success={sr:.1f}%)\n{'='*78}")

    print("\n--- Mean trace feature by outcome (fail vs success) ---")
    g = df.groupby("verified_success")[NUMERIC_FEATURES].mean().T
    g.columns = ["fail", "success"] if list(g.columns) == [False, True] else g.columns
    print(g.round(2).to_string())

    print("\n--- Binary behaviours: prevalence and success rate when present ---")
    rows = []
    for feat in BOOL_FEATURES:
        present = df[df[feat]]
        absent = df[~df[feat]]
        rows.append(dict(
            behaviour=feat,
            pct_of_trials=round(100 * len(present) / n, 1),
            success_when_present=round(100 * present["verified_success"].mean(), 1) if len(present) else np.nan,
            success_when_absent=round(100 * absent["verified_success"].mean(), 1) if len(absent) else np.nan,
        ))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n--- Point-biserial correlation of trace feature with verified_success ---")
    for feat in NUMERIC_FEATURES:
        sub = df[[feat, "verified_success"]].dropna()
        if len(sub) < 3 or sub[feat].std() == 0:
            print(f"  {feat:22s}: (insufficient variation)")
            continue
        r = sub[feat].corr(sub["verified_success"].astype(float))
        print(f"  {feat:22s}: r = {r:+.3f}   (n={len(sub)})")

    # never-experimented, split by cause
    ne = df[~df["ran_experiment"]]
    if len(ne):
        locked = ne[ne["had_format_failure"]]
        guessed = ne[~ne["had_format_failure"]]
        print(f"\n--- Never ran an experiment: {len(ne)}/{n} ({100*len(ne)/n:.1f}%), "
              f"success {100*ne['verified_success'].mean():.1f}% ---")
        print(f"  locked out by format failures : {len(locked):3d}  "
              f"(success {100*locked['verified_success'].mean():.1f}%)" if len(locked) else
              "  locked out by format failures :   0")
        print(f"  answered from priors, no error: {len(guessed):3d}  "
              f"(success {100*guessed['verified_success'].mean():.1f}%)" if len(guessed) else
              "  answered from priors, no error:   0")

    # format failure burden
    ff = df[df["had_format_failure"]]
    if len(ff):
        print(f"\n--- Format failures: {len(ff)}/{n} trials ({100*len(ff)/n:.1f}%) hit >=1, "
              f"{df['format_failures'].sum()} total wasted turns ---")
        print(f"  success with any format failure : {100*ff['verified_success'].mean():.1f}%")
        print(f"  success with none               : {100*df[~df['had_format_failure']]['verified_success'].mean():.1f}%")

    # churn
    print("\n--- Success rate vs. distinct candidate laws floated (hypothesis churn) ---")
    d = df.copy()
    d["_bin"] = pd.cut(d["hypothesis_churn"], [-0.5, 0.5, 1.5, 2.5, 3.5, 1e9],
                       labels=["0", "1", "2", "3", "4+"])
    t = d.groupby("_bin", observed=True)["verified_success"].agg(n="size", success=lambda x: round(100*x.mean(), 1))
    print(t.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--result_dir", default="evaluation_results")
    ap.add_argument("--agent", choices=["vanilla_agent", "code_assisted_agent"], default=None)
    ap.add_argument("--module", default=None)
    ap.add_argument("--rmsle_threshold", type=float, default=DEFAULT_RMSLE_THRESHOLD)
    ap.add_argument("--dump-examples", default=None,
                    help="Print trial paths for a behaviour: one of "
                         "had_format_failure, unverified_submit, or 'never_experimented'.")
    ap.add_argument("-o", "--output_csv", default=None)
    args = ap.parse_args()

    out_csv = args.output_csv or f"analysis/trajectory_trace_{args.model}.csv"

    df = load_trials(args.result_dir, args.model, include_fails=True)
    df = clean_rmsle_outliers(df)
    df = compute_verdicts(df, args.rmsle_threshold)
    df["verified_success"] = df["agreement_bucket"] == "consistent_pass"

    if args.module:
        df = df[df["module"] == args.module]
    if args.agent:
        df = df[df["agent_backend"] == args.agent]
    if df.empty:
        raise SystemExit("No trials match the given filters.")

    traces = _load_traces(df)
    merged = df.merge(traces, on="path", how="left")

    label = f"Trace analysis: {args.model}"
    if args.module:
        label += f" / {args.module}"
    if args.agent:
        label += f" / {args.agent}"
    report(merged, label)

    if args.agent is None:
        print(f"\n\n{'#'*78}\n# Split by agent_backend\n{'#'*78}")
        for backend, g in merged.groupby("agent_backend", observed=True):
            report(g, f"{args.model} / {backend}")

    if args.dump_examples:
        key = args.dump_examples
        if key == "never_experimented":
            sel = merged[~merged["ran_experiment"].fillna(True)]
        else:
            sel = merged[merged[key].fillna(False)]
        print(f"\n--- {len(sel)} trials where {key} ---")
        for _, r in sel.iterrows():
            print(f"  [{'OK ' if r['verified_success'] else 'FAIL'}] {r['module']:24s} "
                  f"{r['agent_backend']:20s} {r['path']}")

    cols = ["path", "module", "equation_difficulty", "model_system", "law_version",
            "agent_backend", "trial_id", "verified_success", "structural_verdict"] + \
        NUMERIC_FEATURES + BOOL_FEATURES
    merged[[c for c in cols if c in merged.columns]].to_csv(out_csv, index=False)
    print(f"\nPer-trial trace table written to {out_csv}")


if __name__ == "__main__":
    main()
