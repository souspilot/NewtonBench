"""
NewtonBench deep-dive: where is the model going wrong, and can we do something
about it?  Slow (sympy + full-transcript parsing); run scoreboard.py first for
headline numbers.

Subcommands (all take --model, --result_dir, --subset_file, --agent, --module):

  verdicts   Judge vs. deterministic-sympy vs. RMSLE agreement. Finds
             judge-lenient credit (inflated success) and judge-strict misses.
             WRITES analysis/verdicts_<model>.csv -- the per-trial
             verified_success label scoreboard.py picks up.

  mistakes   For trials that are genuinely wrong (judge AND sympy agree),
             classify HOW (missing variable / wrong exponent / sign flip /
             ...), with sampled example laws per bucket.

  trace      Mine chat_history: reasoning blowup, format failures, hypothesis
             churn, unverified submissions, plus resource-vs-outcome tables.
             Prints an example trajectory for each failure signal.

  agents     vanilla vs code_assisted vs planned: per-cell SA, and the
             configs where they most disagree (with dumped case files).

  all        verdicts -> mistakes -> trace -> agents in sequence.

Usage:
    python analysis/diagnostics.py verdicts  --model qwen38-27b --subset_file configs/representative_subset.json
    python analysis/diagnostics.py trace     --model qwen38-27b --subset_file configs/representative_subset.json
    python analysis/diagnostics.py all       --model qwen38-27b --subset_file configs/representative_subset.json
"""
import argparse
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from newton_common import (  # noqa: E402
    load_trials, filter_to_subset, clean_rmsle_outliers, compute_verdicts, time_limit,
    verdicts_csv_path, load_verified_labels, analysis_path,
    DEFAULT_RMSLE_THRESHOLD, DEFAULT_SYMPY_TIMEOUT, MAX_TURNS,
    MODULE_SHORT, SYS_SHORT, AGENT_SHORT, MODULE_ORDER, DIFFICULTIES, SYSTEMS,
)


# ===========================================================================
# shared frame prep
# ===========================================================================

VERDICT_COLS = ["path", "module", "equation_difficulty", "model_system", "law_version",
                "agent_backend", "trial_id", "is_fail", "status", "rounds", "num_experiments",
                "total_tokens", "rmsle", "exact_accuracy", "judge_verdict", "rmsle_verdict",
                "structural_verdict", "agreement_bucket", "raw_success", "verified_success",
                "symbolic_msg", "submitted_law", "ground_truth_law"]


def _load_filtered(args, include_fails: bool) -> pd.DataFrame:
    df = load_trials(args.result_dir, args.model, include_fails=include_fails)
    df = filter_to_subset(df, args.subset_file)
    if args.module:
        df = df[df["module"] == args.module].reset_index(drop=True)
    if args.agent:
        df = df[df["agent_backend"] == args.agent].reset_index(drop=True)
    if df.empty:
        raise SystemExit("No trials match the given filters.")
    return clean_rmsle_outliers(df)


def compute_and_write_verdicts(args) -> pd.DataFrame:
    """The one expensive sympy pass. `verdicts` calls it directly; `mistakes`
    and `trace` go through verdict_frame() which reuses the written CSV."""
    df = compute_verdicts(_load_filtered(args, include_fails=True),
                          args.rmsle_threshold, getattr(args, "sympy_timeout", DEFAULT_SYMPY_TIMEOUT))
    out = verdicts_csv_path(args.model)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df[[c for c in VERDICT_COLS if c in df.columns]].to_csv(out, index=False)
    return df


def verdict_frame(args) -> pd.DataFrame:
    """Per-trial verdict table for `mistakes` / `trace`: reuse verdicts_<model>.csv
    if it exists and still covers every filtered trial, else run (and cache) the
    sympy pass. Avoids paying for structural checks 3x in `all`."""
    cached = load_verified_labels(args.model)
    if cached is not None:
        want = _load_filtered(args, include_fails=True)
        merged = want.drop(columns=[c for c in cached.columns if c != "path" and c in want.columns]) \
                     .merge(cached, on="path", how="left")
        if merged["agreement_bucket"].notna().all():
            print(f"(reusing {verdicts_csv_path(args.model)} -- run `diagnostics.py verdicts` to refresh)")
            return merged
        print("(verdicts_<model>.csv is stale / missing rows -- recomputing)")
    return compute_and_write_verdicts(args)


def cmd_verdicts(args):
    df = compute_and_write_verdicts(args)

    print(f"\n{'='*70}\nVerdicts: {args.model}   (n={len(df)})\n{'='*70}")
    print("agreement_bucket trusts the deterministic sympy structural check whenever it "
          "reached a verdict;\nthe RMSLE threshold only decides not_checkable rows.")

    print("\n=== Agreement bucket counts ===")
    print(df["agreement_bucket"].value_counts().to_string())
    n = len(df)
    print(f"\nraw success (judge/exact_accuracy): {100*df['raw_success'].mean():.1f}%")
    print(f"verified success (judge AND sympy): {100*df['verified_success'].mean():.1f}%")
    flipped = int((df["raw_success"] != df["verified_success"]).sum())
    print(f"{flipped}/{n} trials differ between the two labels.")

    lenient = df[df["agreement_bucket"] == "judge_lenient"]
    strict = df[df["agreement_bucket"] == "judge_strict"]

    print("\n=== judge_lenient (judge says equivalent, sympy disagrees) by structural_verdict ===")
    print(lenient["structural_verdict"].value_counts().to_string() if len(lenient) else "(none)")
    print("  constant_equivalent = right form, wrong constant (arguably still deserves credit)")
    print("  structurally_different = genuine judge error; not_checkable = needs a human read")

    print("\n=== Breakdown by module ===")
    print(pd.crosstab(df["module"], df["agreement_bucket"]).to_string())

    show = ["module", "equation_difficulty", "model_system", "law_version", "agent_backend",
            "trial_id", "rmsle", "path"]
    concerning = lenient[lenient["structural_verdict"] == "structurally_different"].sort_values(
        "rmsle", ascending=False)
    if len(concerning):
        print(f"\n=== Top {min(args.top, len(concerning))} concerning judge errors "
              f"(sorted by RMSLE; ~1e-16 = classifier miss, ~1e-3+ = real) ===")
        print(concerning[show].head(args.top).to_string(index=False))
    if len(strict):
        print(f"\n=== Top {min(args.top, len(strict))} judge_strict (lowest RMSLE = likely false negative) ===")
        print(strict.sort_values("rmsle")[show + ["structural_verdict"]].head(args.top).to_string(index=False))

    print(f"\nPer-trial verified_success labels written to {verdicts_csv_path(args.model)} "
          f"(scoreboard.py --verified and the other subcommands read this).")


# ===========================================================================
# mistakes
# ===========================================================================

MISTAKE_ORDER = ["missing_variable", "sign_flip", "extra_variable", "wrong_exponent",
                 "nonlinear_dependence", "other_structural", "not_checkable"]
MISTAKE_BLURB = {
    "missing_variable": "Ground truth depends on a parameter the submitted law ignores entirely.",
    "sign_flip": "A parameter's exponent has the wrong SIGN, not just the wrong magnitude.",
    "extra_variable": "Submitted law depends on a parameter ground truth doesn't.",
    "wrong_exponent": "Right variables and sign, wrong exponent magnitude.",
    "nonlinear_dependence": "Both depend on the parameter, but not as a comparable single power. Manual read.",
    "other_structural": "Operator-level difference (additive vs multiplicative etc). Manual read.",
    "not_checkable": "Submitted law uses control flow the checker can't parse. Manual read.",
}


def cmd_mistakes(args):
    from mismatch_classifier import classify_mismatch

    df = verdict_frame(args)
    fails = df[df["agreement_bucket"] == "consistent_fail"].copy()
    if fails.empty:
        raise SystemExit("No consistent_fail trials -- nothing to classify.")

    # classify once per unique (submitted, ground_truth) pair, each call time-limited
    # (mismatch_classifier's sp.simplify can hang on nested exp/log/power forms).
    timeout = getattr(args, "sympy_timeout", DEFAULT_SYMPY_TIMEOUT)
    cache, n_timeout = {}, 0
    for sub, gt in fails[["submitted_law", "ground_truth_law"]].drop_duplicates().itertuples(index=False):
        if not isinstance(sub, str) or not isinstance(gt, str):
            cache[(sub, gt)] = ("not_checkable", "")
            continue
        try:
            with time_limit(timeout):
                res = classify_mismatch(sub, gt)
            cache[(sub, gt)] = (res["mistake_type"],
                                "; ".join(f"{p}: sub={d['sub']}, gt={d['gt']} ({d['issue']})"
                                          for p, d in res["details"].items() if d["issue"] != "matches"))
        except Exception:  # noqa: BLE001  (TimeoutError included)
            cache[(sub, gt)] = ("not_checkable", "")
            n_timeout += 1
    if n_timeout:
        print(f"  ({n_timeout} law pair(s) hit the {timeout}s classify timeout -> not_checkable)")
    fails["mistake_type"] = [cache.get((s, g), ("not_checkable", ""))[0]
                             for s, g in zip(fails["submitted_law"], fails["ground_truth_law"])]
    fails["mistake_detail"] = [cache.get((s, g), ("not_checkable", ""))[1]
                               for s, g in zip(fails["submitted_law"], fails["ground_truth_law"])]

    print(f"\n{'='*70}\nMistake taxonomy: {args.model}   ({len(fails)} genuinely-wrong trials)\n{'='*70}")
    counts = fails["mistake_type"].value_counts()
    for mt in MISTAKE_ORDER:
        if mt in counts.index:
            print(f"  {mt:20s} {counts[mt]:4d}   {MISTAKE_BLURB[mt]}")

    print("\n=== module x mistake type ===")
    cols = [m for m in MISTAKE_ORDER if m in fails["mistake_type"].unique()]
    print(pd.crosstab(fails["module"], fails["mistake_type"]).reindex(columns=cols).to_string())

    for mt in MISTAKE_ORDER:
        sub = fails[fails["mistake_type"] == mt]
        if sub.empty:
            continue
        print(f"\n{'-'*70}\n{mt.upper()}  ({len(sub)} trials) -- {MISTAKE_BLURB[mt]}\n{'-'*70}")
        sample = sub.sample(n=min(args.samples, len(sub)), random_state=0).sort_values(
            ["module", "equation_difficulty", "model_system"])
        for _, row in sample.iterrows():
            print(f"\n  [{row['module']} / {row['equation_difficulty']} / {row['model_system']} / "
                  f"{row['law_version']} / {row['agent_backend']} / trial{row['trial_id']}]")
            if row["mistake_detail"]:
                print(f"    Mismatch:  {row['mistake_detail']}")
            print(f"    Truth:     {row['ground_truth_law']}")
            print(f"    Submitted: {' '.join(str(row['submitted_law']).split())}")
            print(f"    Path:      {row['path']}")

    out = analysis_path(f"mistake_taxonomy_{args.model}.csv")
    fails[["path", "module", "equation_difficulty", "model_system", "law_version", "agent_backend",
           "trial_id", "rmsle", "mistake_type", "mistake_detail", "submitted_law",
           "ground_truth_law"]].to_csv(out, index=False)
    print(f"\nFull classified table written to {out}")


# ===========================================================================
# trace  (chat_history mining + resource-vs-outcome)
# ===========================================================================

_INVALID_MARKERS = ("Invalid response", "Action Reminder", "exactly 1 action per turn")
_LAW_BODY_RE = re.compile(r"def\s+discovered_law\s*\([^)]*\)\s*:(.*?)(?=\ndef\s|\Z)", re.DOTALL)
_RETURN_RE = re.compile(r"return\s+(.+)")
_EXP_OUTPUT_RE = re.compile(r"<experiment_output>\s*\n?\s*[\[{]")


def _split_reasoning(content: str):
    if "**Main Response:**" in content:
        head, _, tail = content.partition("**Main Response:**")
        return head.replace("**Reasoning Process:**", "").strip(), tail.strip()
    return "", content.strip()


def _norm_law(body: str) -> str:
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

    format_failures = sum(any(mk in u for mk in _INVALID_MARKERS) for u in user)
    exp_output_turns = [i for i, u in enumerate(user) if _EXP_OUTPUT_RE.search(u)]
    python_outputs = [u for u in user if "<python_output>" in u]
    python_errors = sum(("Traceback" in u or "Error:" in u) for u in python_outputs)

    nan_count = tot_count = 0
    for i in exp_output_turns:
        toks = re.findall(r"-?\d+\.?\d*(?:e-?\d+)?|nan", user[i], flags=re.IGNORECASE)
        tot_count += len(toks)
        nan_count += sum(t.lower() == "nan" for t in toks)

    exp_request_turns = [i for i, resp in enumerate(responses) if "<run_experiment>" in resp]
    seen, first_turn = {}, {}
    for i, resp in enumerate(responses):
        for body in _LAW_BODY_RE.findall(resp):
            key = _norm_law(body)
            if key and key not in seen:
                seen[key] = i
                first_turn[key] = i

    final_key = None
    for resp in reversed(responses):
        bodies = _LAW_BODY_RE.findall(resp)
        if bodies:
            final_key = _norm_law(bodies[-1])
            break
    if final_key is None or final_key not in first_turn:
        unverified_submit = True
    else:
        unverified_submit = not any(t >= first_turn[final_key] for t in exp_request_turns)

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


TRACE_NUMERIC = ["assistant_turns", "rounds", "num_experiments", "total_tokens", "format_failures",
                 "n_experiment_batches", "n_python_calls", "python_errors", "nan_fraction",
                 "hypothesis_churn", "reasoning_chars", "max_msg_chars"]
TRACE_BOOL = ["had_format_failure", "no_parseable_law", "unverified_submit"]


def _bin(df, col, bins, labels, tgt="verified_success"):
    d = df.dropna(subset=[col]).copy()
    d["_bin"] = pd.cut(d[col], bins=bins, labels=labels, include_lowest=True)
    g = d.groupby("_bin", observed=True)[tgt].agg(n="size", success_pct=lambda x: round(100 * x.mean(), 1))
    return g


def _trace_report(df, label, example_sink):
    n = len(df)
    print(f"\n{'='*78}\n{label}   (n={n}, verified_success={100*df['verified_success'].mean():.1f}%)\n{'='*78}")

    print("\n--- Mean trace feature by outcome ---")
    g = df.groupby("verified_success")[TRACE_NUMERIC].mean().T
    if list(g.columns) == [False, True]:
        g.columns = [f"fail(n={int((~df['verified_success']).sum())})",
                     f"success(n={int(df['verified_success'].sum())})"]
    print(g.round(2).to_string())

    print("\n--- Point-biserial correlation with verified_success ---")
    for feat in TRACE_NUMERIC:
        sub = df[[feat, "verified_success"]].dropna()
        if len(sub) < 3 or sub[feat].std() == 0:
            print(f"  {feat:20s}: (insufficient variation)")
        else:
            print(f"  {feat:20s}: r = {sub[feat].corr(sub['verified_success'].astype(float)):+.3f}")

    print("\n--- Binary behaviours: prevalence and success rate ---")
    rows = []
    for feat in TRACE_BOOL:
        p, a = df[df[feat].fillna(False)], df[~df[feat].fillna(False)]
        rows.append(dict(behaviour=feat, pct=round(100 * len(p) / n, 1),
                         success_present=round(100 * p["verified_success"].mean(), 1) if len(p) else np.nan,
                         success_absent=round(100 * a["verified_success"].mean(), 1) if len(a) else np.nan))
    print(pd.DataFrame(rows).to_string(index=False))

    if df["equation_difficulty"].nunique() > 1:
        print("\n--- turns / reasoning(k) / max_msg(k) / experiments by (difficulty x outcome) ---")
        t = df.groupby(["equation_difficulty", "verified_success"]).agg(
            n=("path", "size"), turns=("assistant_turns", "mean"),
            reasoning_k=("reasoning_chars", lambda s: s.mean() / 1000),
            maxmsg_k=("max_msg_chars", lambda s: s.mean() / 1000),
            num_exp=("num_experiments", "mean"))
        print(t.round(1).to_string())

    print("\n--- Resource vs. outcome (binned) ---")
    print("experiments:")
    print(_bin(df, "num_experiments", [-0.5, 0.5, 2.5, 5.5, 1e9], ["0", "1-2", "3-5", "6+"]).to_string())
    print("rounds:")
    print(_bin(df, "rounds", [-0.5, 1.5, 3.5, 6.5, MAX_TURNS + 0.5],
               ["1", "2-3", "4-6", f"7-{MAX_TURNS}"]).to_string())
    try:
        q = df["total_tokens"].quantile([0, .25, .5, .75, 1.0]).tolist()
        if len(set(q)) == 5:
            print("total_tokens quartiles:")
            print(_bin(df, "total_tokens", q, ["Q1(few)", "Q2", "Q3", "Q4(most)"]).to_string())
    except (ValueError, IndexError):
        pass

    zero = df[df["num_experiments"] == 0]
    if len(zero):
        print(f"\nzero-experiment trials: {len(zero)}/{n} ({100*len(zero)/n:.1f}%), "
              f"verified_success {100*zero['verified_success'].mean():.1f}%")
    prem = df[(df["status"] != "max_turns_reached") & (df["rounds"] <= 2)]
    if len(prem):
        print(f"submitted within 2 rounds (not cut off): {len(prem)}/{n} ({100*len(prem)/n:.1f}%), "
              f"verified_success {100*prem['verified_success'].mean():.1f}% "
              f"(everyone else {100*df[~df.index.isin(prem.index)]['verified_success'].mean():.1f}%)")

    nl = df[df["no_parseable_law"].fillna(False)]
    if len(nl):
        print(f"\nno parseable <final_law>: {len(nl)}/{n}, success {100*nl['verified_success'].mean():.1f}%")
        print(nl.groupby(["module", "agent_backend"], observed=True).size().to_string())
    ff = df[df["had_format_failure"].fillna(False)]
    if len(ff):
        print(f"\n>=1 format failure: {len(ff)}/{n} ({100*len(ff)/n:.1f}%), "
              f"{int(df['format_failures'].sum())} wasted turns, "
              f"success {100*ff['verified_success'].mean():.1f}% "
              f"(vs {100*df[~df['had_format_failure'].fillna(False)]['verified_success'].mean():.1f}%)")

    # stash an example failing trajectory for each signal (printed once, at top level)
    for sig, mask in [
        ("reasoning blowup (longest msg, a failure)",
         (~df["verified_success"]) & df["max_msg_chars"].notna()),
        ("format failure (a failure)", (~df["verified_success"]) & df["had_format_failure"].fillna(False)),
        ("unverified submit (a failure)", (~df["verified_success"]) & df["unverified_submit"].fillna(False)),
        ("no parseable final_law", df["no_parseable_law"].fillna(False)),
    ]:
        cand = df[mask]
        if len(cand) and sig not in example_sink:
            r = cand.sort_values("max_msg_chars", ascending=False).iloc[0]
            example_sink[sig] = f"{r['module']}/{r['agent_backend']}/{r['equation_difficulty']}  {r['path']}"


def cmd_trace(args):
    df = verdict_frame(args)

    recs = []
    for p in df["path"]:
        if not os.path.exists(p):
            recs.append({"path": p})
            continue
        try:
            with open(p) as f:
                recs.append({"path": p, **analyse_trace(json.load(f).get("chat_history", []))})
        except Exception as e:  # noqa: BLE001
            print(f"  (skipped {p}: {e})")
            recs.append({"path": p})

    traces = pd.DataFrame(recs)
    merged = df.merge(traces, on="path", how="left")

    examples = {}
    _trace_report(merged, f"Trace: {args.model}", examples)
    if args.agent is None:
        for backend, g in merged.groupby("agent_backend", observed=True):
            _trace_report(g, f"{args.model} / {backend}", examples)

    if examples:
        print(f"\n{'='*78}\nExample trajectories to open (one per failure signal)\n{'='*78}")
        for sig, loc in examples.items():
            print(f"  {sig}:\n    {loc}")

    out = analysis_path(f"trajectory_trace_{args.model}.csv")
    keep = (["path", "module", "equation_difficulty", "model_system", "law_version", "agent_backend",
             "trial_id", "verified_success", "structural_verdict", "status"] + TRACE_NUMERIC + TRACE_BOOL)
    merged[[c for c in keep if c in merged.columns]].to_csv(out, index=False)
    print(f"\nPer-trial trace table written to {out}")


# ===========================================================================
# agents  (divergence)
# ===========================================================================

def _load_agent_trials(base_dir, model):
    """(agent, module, difficulty, version, system) -> list of trial dicts, raw
    (no dedup -- versions are a real axis here)."""
    trials = defaultdict(list)
    model_dir = Path(base_dir) / model
    if not model_dir.exists():
        raise SystemExit(f"{model_dir} not found")
    for tf in model_dir.rglob("trials/trial*.json"):
        if "_chat_history" in tf.name or tf.name.endswith("_fail.json"):
            continue
        try:
            data = json.load(open(tf))
        except Exception:  # noqa: BLE001
            continue
        data["_chat_log"] = str(tf).replace(".json", "_chat_history.log")
        key = (data.get("agent_backend"), data.get("module_name"), data.get("equation_difficulty"),
               data.get("law_version"), data.get("model_system"))
        trials[key].append(data)
    return trials


def _acc(t):
    ev = t.get("evaluation", {})
    return float(ev.get("exact_accuracy", 0.0)) if isinstance(ev, dict) else 0.0


def cmd_agents(args):
    trials = _load_agent_trials(args.result_dir, args.model)
    if args.subset_file:
        from newton_common import load_subset_cells
        cells = load_subset_cells(args.subset_file)
        trials = {k: v for k, v in trials.items()
                  if cells and (k[2], k[4]) in cells.get(k[1], set())}
    agents = sorted({k[0] for k in trials if k[0]})
    print(f"\n{'='*78}\nAgent comparison: {args.model}   agents={agents}\n{'='*78}")

    # per (system, difficulty) SA per agent
    agg = defaultdict(list)
    for (agent, module, diff, ver, sysd), tl in trials.items():
        for t in tl:
            agg[(agent, sysd, diff)].append(_acc(t))
    print(f"\n  {'System':<7} {'Diff':<7} " + " ".join(f"{AGENT_SHORT.get(a, a):>9}" for a in agents))
    for sysd in SYSTEMS:
        for diff in DIFFICULTIES:
            cells = [agg.get((a, sysd, diff), []) for a in agents]
            if not any(cells):
                continue
            print(f"  {SYS_SHORT[sysd]:<7} {diff:<7} " +
                  " ".join(f"{100*statistics.mean(c):>9.1f}" if c else f"{'-':>9}" for c in cells))
    for a in agents:
        allv = [x for k, v in agg.items() if k[0] == a for x in v]
        if allv:
            print(f"  overall {AGENT_SHORT.get(a, a):<15} {100*statistics.mean(allv):.1f}%  (n={len(allv)})")

    # divergences: same config, agents disagree
    by_config = defaultdict(dict)
    for (agent, module, diff, ver, sysd), tl in trials.items():
        accs = [_acc(t) for t in tl]
        by_config[(module, diff, ver, sysd)][agent] = (statistics.mean(accs) if accs else 0.0, tl)
    divs = []
    for (module, diff, ver, sysd), per_agent in by_config.items():
        names = list(per_agent)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a1, a2 = names[i], names[j]
                s1, s2 = per_agent[a1][0], per_agent[a2][0]
                if abs(s1 - s2) > 0.01:
                    w, l = (a1, a2) if s1 > s2 else (a2, a1)
                    divs.append(dict(module=module, diff=diff, ver=ver, sys=sysd, winner=w, loser=l,
                                     w_sa=max(s1, s2), l_sa=min(s1, s2),
                                     w_tl=per_agent[w][1], l_tl=per_agent[l][1]))
    divs.sort(key=lambda d: d["w_sa"] - d["l_sa"], reverse=True)

    print(f"\n--- {len(divs)} divergent configs (agents disagree) ---")
    wins = defaultdict(lambda: defaultdict(int))
    for d in divs:
        wins[d["winner"]][d["loser"]] += 1
    for w in agents:
        for l in agents:
            if wins[w][l]:
                print(f"  {AGENT_SHORT.get(w, w)} beats {AGENT_SHORT.get(l, l)}: {wins[w][l]}")
    for d in divs[:args.top]:
        print(f"  {MODULE_SHORT.get(d['module'], d['module']):<9} {SYS_SHORT[d['sys']]:<6} "
              f"{d['diff']:<7} {d['ver']:<3} | {AGENT_SHORT.get(d['winner'], d['winner']):>8} "
              f"{100*d['w_sa']:>5.1f}  {AGENT_SHORT.get(d['loser'], d['loser']):>8} {100*d['l_sa']:>5.1f} "
              f"| {100*(d['w_sa']-d['l_sa']):>+5.1f}")

    outdir = Path(analysis_path(f"divergent_cases/{args.model}"))
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for idx, d in enumerate(divs[:args.max_examples]):
        fn = f"{idx:02d}_{d['module']}_{d['sys']}_{d['diff']}_{d['ver']}_{AGENT_SHORT.get(d['winner'],d['winner'])}vs{AGENT_SHORT.get(d['loser'],d['loser'])}.txt"
        with open(outdir / fn, "w") as f:
            f.write(f"{d['module']} / {d['sys']} / {d['diff']} / {d['ver']}\n")
            f.write(f"WINNER {d['winner']} SA={100*d['w_sa']:.1f}   LOSER {d['loser']} SA={100*d['l_sa']:.1f}\n\n")
            for tag, tl in [("WINNER", d["w_tl"]), ("LOSER", d["l_tl"])]:
                f.write(f"--- {tag} laws ---\n")
                for t in tl:
                    f.write(f"  trial{t.get('trial_id')}: acc={_acc(t)}  {t.get('submitted_law')}\n")
            for t in d["l_tl"]:
                cl = t.get("_chat_log", "")
                if cl and os.path.exists(cl):
                    f.write(f"\n--- LOSER trial{t.get('trial_id')} chat ---\n")
                    c = open(cl).read()
                    f.write(c if len(c) < 20000 else c[:10000] + "\n...[TRUNCATED]...\n" + c[-10000:])
                    break
        manifest.append({"file": fn, **{k: d[k] for k in ("module", "sys", "diff", "ver")},
                         "winner": d["winner"], "loser": d["loser"],
                         "gap": f"{100*(d['w_sa']-d['l_sa']):+.1f}"})
    json.dump(manifest, open(outdir / "manifest.json", "w"), indent=2)
    print(f"\nWrote {len(manifest)} divergence case files to {outdir}/")


# ===========================================================================
# main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("verdicts", "mistakes", "trace", "agents", "all"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--result_dir", default="evaluation_results")
        p.add_argument("--subset_file", default=None,
                       help="representative_subset.json -- pin analysis to one cell set.")
        p.add_argument("--agent", choices=["vanilla_agent", "code_assisted_agent", "planned_agent"],
                       default=None)
        p.add_argument("--module", default=None)
        p.add_argument("--rmsle_threshold", type=float, default=DEFAULT_RMSLE_THRESHOLD)
        p.add_argument("--sympy_timeout", type=float, default=DEFAULT_SYMPY_TIMEOUT,
                       help="seconds per unique law pair before the structural / classify "
                            "check bails to not_checkable (0 disables)")
        p.add_argument("--top", type=int, default=15)
        if name in ("mistakes", "all"):
            p.add_argument("--samples", type=int, default=8)
        if name in ("agents", "all"):
            p.add_argument("--max-examples", type=int, default=15, dest="max_examples")
    args = ap.parse_args()
    if not hasattr(args, "samples"):
        args.samples = 8
    if not hasattr(args, "max_examples"):
        args.max_examples = 15

    steps = {"verdicts": cmd_verdicts, "mistakes": cmd_mistakes,
             "trace": cmd_trace, "agents": cmd_agents}
    if args.cmd == "all":
        for name in ("verdicts", "mistakes", "trace", "agents"):
            print(f"\n\n{'#'*78}\n# {name}\n{'#'*78}")
            try:
                steps[name](args)
            except SystemExit as e:
                print(f"({name} skipped: {e})")
    else:
        steps[args.cmd](args)


if __name__ == "__main__":
    main()
