"""
Chat-history ("trace") mining for a model's NewtonBench trials.

trajectory_analysis.py only looks at the scalar summary fields a trial JSON
records (rounds, num_experiments, total_tokens, status). That tells you
*that* high-token trials fail, not *why*. This script re-opens each trial's
`chat_history` and measures what only the transcript can tell you, then
cross-tabs it against verified_success:

  * reasoning_chars / max_msg_chars -- total chain-of-thought characters, and
                          the single largest assistant message. Runaway CoT.
  * hypothesis_churn   -- number of DISTINCT `def discovered_law` return
                          expressions the model floated across the whole
                          trajectory. 0 = submission never parsed / gave up;
                          high = never converged.
  * unverified_submit  -- the exact functional form it finally submitted was
                          NOT tested against data: no <run_experiment> was
                          issued in or after the turn that form first
                          appeared.
  * format_failures    -- turns the harness had to answer "Invalid response /
                          Action Reminder" because a tag/JSON block didn't
                          parse. Wasted rounds that look like thinking in the
                          token count.
  * nan_fraction       -- fraction of returned experiment data points that
                          were `nan` (bad input ranges the model kept using).
  * python_errors      -- (code_assisted only) <python_output> blocks with a
                          traceback.

Authoritative scalars (rounds, num_experiments, total_tokens, status) are
taken from the trial JSON itself via failure_analysis.load_trials -- NOT
re-derived from the transcript, because instruction text in the task prompt
mentions `<run_experiment>` / `<experiment_output>` and would inflate any
naive tag count.

For each numeric feature: mean-by-outcome, point-biserial r, and -- because
"longer trajectory correlates with failure" is partly just "hard problems
take longer" -- the same means split by equation_difficulty so you can see
whether the effect survives that control.

Reuses failure_analysis.load_trials/compute_verdicts (same dedup + verified-
success label as trajectory_analysis.py) and joins on the trial `path`.

Usage:
    python analysis/trajectory_trace_analysis.py --model qwq-32b
    python analysis/trajectory_trace_analysis.py --model qwq-32b --agent code_assisted_agent
    python analysis/trajectory_trace_analysis.py --model qwq-32b --dump-examples unverified_submit
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

_INVALID_MARKERS = (
    "Invalid response",
    "Action Reminder",
    "exactly 1 action per turn",
)
_LAW_BODY_RE = re.compile(r"def\s+discovered_law\s*\([^)]*\)\s*:(.*?)(?=\ndef\s|\Z)", re.DOTALL)
_RETURN_RE = re.compile(r"return\s+(.+)")
# an <experiment_output> block that actually carries data (starts with a list/obj),
# as opposed to the task prompt merely naming the tag.
_EXP_OUTPUT_RE = re.compile(r"<experiment_output>\s*\n?\s*[\[{]")


def _split_reasoning(content: str):
    """Assistant messages are stored as
    '**Reasoning Process:**\\n...\\n\\n**Main Response:**\\n...' when the provider
    returned a separate reasoning field, else just the response text."""
    if "**Main Response:**" in content:
        head, _, tail = content.partition("**Main Response:**")
        return head.replace("**Reasoning Process:**", "").strip(), tail.strip()
    return "", content.strip()


def _norm_law(body: str) -> str:
    """Reduce a discovered_law body to its return expression, whitespace-stripped,
    so cosmetically-different restatements of one formula collapse together."""
    m = _RETURN_RE.search(body)
    expr = m.group(1) if m else body
    return re.sub(r"\s+", "", expr).rstrip(")")[:400]


def analyse_trace(chat_history):
    assistant = [m.get("content", "") or "" for m in chat_history if m.get("role") == "assistant"]
    user = [m.get("content", "") or "" for m in chat_history if m.get("role") == "user"]

    responses, reasonings = [], []
    for c in assistant:
        r, resp = _split_reasoning(c)
        reasonings.append(r)
        responses.append(resp)

    # -- format failures: harness nag messages (these strings never appear in a task prompt) --
    format_failures = sum(any(mk in u for mk in _INVALID_MARKERS) for u in user)

    # -- real experiment-output blocks (data-carrying, not the prompt naming the tag) --
    exp_output_turns = [i for i, u in enumerate(user) if _EXP_OUTPUT_RE.search(u)]
    python_outputs = [u for u in user if "<python_output>" in u]
    python_errors = sum(("Traceback" in u or "Error:" in u) for u in python_outputs)

    nan_count = tot_count = 0
    for i in exp_output_turns:
        toks = re.findall(r"-?\d+\.?\d*(?:e-?\d+)?|nan", user[i], flags=re.IGNORECASE)
        tot_count += len(toks)
        nan_count += sum(t.lower() == "nan" for t in toks)

    # -- turns in which the assistant actually issued <run_experiment> --
    exp_request_turns = [i for i, resp in enumerate(responses) if "<run_experiment>" in resp]

    # -- candidate laws floated across the trajectory (churn) + first appearance turn --
    seen, first_turn = {}, {}
    for i, resp in enumerate(responses):
        for body in _LAW_BODY_RE.findall(resp):
            key = _norm_law(body)
            if key and key not in seen:
                seen[key] = i
                first_turn[key] = i

    # -- final submitted form, and whether it was ever tested --
    final_key = None
    for resp in reversed(responses):
        bodies = _LAW_BODY_RE.findall(resp)
        if bodies:
            final_key = _norm_law(bodies[-1])
            break
    if final_key is None:
        unverified_submit = True                       # nothing parseable submitted
    elif final_key not in first_turn:
        unverified_submit = True                       # only appeared in the forced last message
    else:
        appeared = first_turn[final_key]
        unverified_submit = not any(t >= appeared for t in exp_request_turns)

    return dict(
        assistant_turns=len(assistant),
        format_failures=format_failures,
        had_format_failure=format_failures > 0,
        n_experiment_batches=len(exp_output_turns),
        n_python_calls=len(python_outputs),
        python_errors=python_errors,
        nan_fraction=(nan_count / tot_count) if tot_count else 0.0,
        hypothesis_churn=len(seen),
        no_parseable_law=final_key is None,
        unverified_submit=unverified_submit,
        reasoning_chars=sum(len(r) for r in reasonings),
        max_msg_chars=max((len(c) for c in assistant), default=0),
    )


NUMERIC_FEATURES = [
    "assistant_turns", "rounds", "num_experiments", "total_tokens",
    "format_failures", "n_experiment_batches", "n_python_calls", "python_errors",
    "nan_fraction", "hypothesis_churn", "reasoning_chars", "max_msg_chars",
]
BOOL_FEATURES = ["had_format_failure", "no_parseable_law", "unverified_submit"]


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


def report(df: pd.DataFrame, label: str):
    n = len(df)
    sr = 100 * df["verified_success"].mean()
    print(f"\n{'='*78}\n{label}   (n={n}, verified_success={sr:.1f}%)\n{'='*78}")

    print("\n--- Mean trace feature by outcome ---")
    grp = df.groupby("verified_success")
    g = grp[NUMERIC_FEATURES].mean().T
    g.columns = [f"fail (n={int((~df['verified_success']).sum())})",
                 f"success (n={int(df['verified_success'].sum())})"] if list(g.columns) == [False, True] else g.columns
    print(g.round(2).to_string())

    print("\n--- Point-biserial correlation with verified_success ---")
    for feat in NUMERIC_FEATURES:
        sub = df[[feat, "verified_success"]].dropna()
        if len(sub) < 3 or sub[feat].std() == 0:
            print(f"  {feat:20s}: (insufficient variation)")
            continue
        r = sub[feat].corr(sub["verified_success"].astype(float))
        print(f"  {feat:20s}: r = {r:+.3f}")

    print("\n--- Binary behaviours: prevalence and success rate ---")
    rows = []
    for feat in BOOL_FEATURES:
        p, a = df[df[feat]], df[~df[feat].fillna(False)]
        rows.append(dict(
            behaviour=feat,
            pct_of_trials=round(100 * len(p) / n, 1),
            success_when_present=round(100 * p["verified_success"].mean(), 1) if len(p) else np.nan,
            success_when_absent=round(100 * a["verified_success"].mean(), 1) if len(a) else np.nan,
        ))
    print(pd.DataFrame(rows).to_string(index=False))

    # confound control: do the "longer = worse" features survive splitting by difficulty?
    if df["equation_difficulty"].nunique() > 1:
        print("\n--- assistant_turns / reasoning_chars / max_msg_chars by (difficulty x outcome) ---")
        t = df.groupby(["equation_difficulty", "verified_success"]).agg(
            n=("path", "size"),
            turns=("assistant_turns", "mean"),
            reasoning_k=("reasoning_chars", lambda s: s.mean() / 1000),
            maxmsg_k=("max_msg_chars", lambda s: s.mean() / 1000),
            num_exp=("num_experiments", "mean"),
        )
        print(t.round(1).to_string())

    nl = df[df["no_parseable_law"].fillna(False)]
    if len(nl):
        print(f"\n--- No parseable <final_law> in transcript: {len(nl)}/{n} "
              f"({100*len(nl)/n:.1f}%), success {100*nl['verified_success'].mean():.1f}% ---")
        print(nl.groupby(["module", "agent_backend"], observed=True).size().to_string())

    ff = df[df["had_format_failure"].fillna(False)]
    if len(ff):
        print(f"\n--- >=1 format failure: {len(ff)}/{n} ({100*len(ff)/n:.1f}%), "
              f"{int(df['format_failures'].sum())} wasted turns total, "
              f"success {100*ff['verified_success'].mean():.1f}% (vs "
              f"{100*df[~df['had_format_failure'].fillna(False)]['verified_success'].mean():.1f}%) ---")

    print("\n--- Success rate vs. distinct candidate laws floated (hypothesis churn) ---")
    d = df.copy()
    d["_bin"] = pd.cut(d["hypothesis_churn"], [-0.5, 0.5, 1.5, 2.5, 1e9], labels=["0", "1", "2", "3+"])
    print(d.groupby("_bin", observed=True)["verified_success"]
            .agg(n="size", success_pct=lambda x: round(100 * x.mean(), 1)).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--result_dir", default="evaluation_results")
    ap.add_argument("--agent", choices=["vanilla_agent", "code_assisted_agent"], default=None)
    ap.add_argument("--module", default=None)
    ap.add_argument("--rmsle_threshold", type=float, default=DEFAULT_RMSLE_THRESHOLD)
    ap.add_argument("--dump-examples", default=None,
                    help="Print trial paths for a behaviour: had_format_failure, "
                         "no_parseable_law, or unverified_submit.")
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

    merged = df.merge(_load_traces(df), on="path", how="left")

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
        sel = merged[merged[key].fillna(False)]
        print(f"\n--- {len(sel)} trials where {key} ---")
        for _, r in sel.sort_values(["module", "agent_backend"]).iterrows():
            print(f"  [{'OK ' if r['verified_success'] else 'FAIL'}] "
                  f"{str(r['module']):24s} {str(r['agent_backend']):20s} "
                  f"turns={r.get('assistant_turns')} churn={r.get('hypothesis_churn')}  {r['path']}")

    cols = (["path", "module", "equation_difficulty", "model_system", "law_version",
             "agent_backend", "trial_id", "verified_success", "structural_verdict", "status"]
            + NUMERIC_FEATURES + BOOL_FEATURES)
    merged[[c for c in cols if c in merged.columns]].to_csv(out_csv, index=False)
    print(f"\nPer-trial trace table written to {out_csv}")


if __name__ == "__main__":
    main()
