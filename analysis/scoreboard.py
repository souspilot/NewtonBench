"""
NewtonBench headline numbers -- fast. Refreshes analysis/results_by_trial.csv
from the trial JSONs on disk, then prints symbolic accuracy by cell, by
module, and by agent, plus coverage and a resource quick-look.

No sympy, no transcript parsing. For "where is the model going wrong",
"is that 90% judge-inflated", mistake taxonomy, etc. -> diagnostics.py.

verified_success: raw exact-accuracy is what this reads by default. If you've
run `diagnostics.py verdicts --model X` it wrote analysis/verdicts_X.csv;
pass --verified here to show the sympy-checked number next to raw.

Usage:
    python analysis/scoreboard.py --model qwen38-27b
    python analysis/scoreboard.py --model qwen38-27b --subset_file configs/representative_subset.json
    python analysis/scoreboard.py --model qwen38-27b --verified
    python analysis/scoreboard.py --all            # every model in configs/models.txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from newton_common import (  # noqa: E402
    update_results, filter_to_subset, clean_rmsle_outliers, detect_outliers_modified_zscore_column,
    calculate_trial_stats, load_results_by_trial, load_verified_labels, read_models_from_file,
    resolve_result_dir, results_by_trial_csv,
    DIFFICULTIES, SYSTEMS, MODULE_ORDER,
)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell Modified-Z-Score RMSLE outlier masking, same as the old
    summarize_results.aggregate_results."""
    out = []
    for _, g in df.groupby(["module", "equation_difficulty", "model_system", "agent_backend"]):
        out.append(detect_outliers_modified_zscore_column(g.copy(), "rmsle"))
    return pd.concat(out).reset_index(drop=True) if out else df


def _cell_grid(df: pd.DataFrame, title: str):
    print(f"\n--- {title}: SA% by (system x difficulty) ---")
    for backend, bdf in df.groupby("agent_backend"):
        print(f"\n  {backend}")
        header = f"  {'system':<16}" + "".join(f"{d:>10}" for d in DIFFICULTIES)
        print(header)
        for system in SYSTEMS:
            row = f"  {system:<16}"
            for diff in DIFFICULTIES:
                cell = bdf[(bdf["equation_difficulty"] == diff) & (bdf["model_system"] == system)]
                acc, _, _, _ = calculate_trial_stats(cell)
                row += f"{(acc*100):>9.1f}" + " " if pd.notna(acc) else f"{'-':>10}"
            print(row)
        acc, std, rmsle, _ = calculate_trial_stats(bdf)
        tok = bdf["total_tokens"].mean()
        print(f"  {'OVERALL':<16}{(acc*100):>9.1f}   (rmsle {rmsle:.3f}, {tok:,.0f} tok/trial)"
              if pd.notna(acc) else f"  {'OVERALL':<16}{'-':>9}")


def _per_module(df: pd.DataFrame):
    print("\n--- SA% by module x agent ---")
    print(f"  {'module':<24}" + "".join(f"{b[:14]:>16}" for b in sorted(df['agent_backend'].unique())))
    backends = sorted(df["agent_backend"].unique())
    for module in [m for m in MODULE_ORDER if m in df["module"].unique()]:
        row = f"  {module:<24}"
        for b in backends:
            cell = df[(df["module"] == module) & (df["agent_backend"] == b)]
            acc, _, _, _ = calculate_trial_stats(cell)
            row += f"{(acc*100):>15.1f} " if pd.notna(acc) else f"{'-':>16}"
        print(row)


def _agents_ab(df: pd.DataFrame):
    backends = sorted(df["agent_backend"].unique())
    if len(backends) < 2:
        return
    print("\n--- agent A/B (overall verified? no -- raw SA) ---")
    for b in backends:
        bdf = df[df["agent_backend"] == b]
        acc, std, _, _ = calculate_trial_stats(bdf)
        print(f"  {b:<22} {acc*100:>5.1f}%  (n_trials={len(bdf)}, "
              f"{bdf['rounds'].mean():.1f} rounds, {bdf['experiments'].mean():.0f} exp, "
              f"{bdf['total_tokens'].mean():,.0f} tok)")


def _coverage(df: pd.DataFrame):
    print("\n--- coverage (trials on disk per module x agent) ---")
    ct = df.pivot_table(index="module", columns="agent_backend", values="trial_id",
                        aggfunc="count", fill_value=0)
    print(ct.to_string())
    missing = sorted(set(MODULE_ORDER) - set(df["module"].unique()))
    if missing:
        print(f"  modules with NO trials: {missing}")


def _resource_quicklook(df: pd.DataFrame):
    d = df.dropna(subset=["total_tokens", "exact_accuracy"]).copy()
    if d.empty:
        return
    print("\n--- SA% vs. total_tokens quartile (raw) ---")
    try:
        q = d["total_tokens"].quantile([0, .25, .5, .75, 1.0]).tolist()
        if len(set(q)) == 5:
            d["_q"] = pd.cut(d["total_tokens"], q, labels=["Q1(few)", "Q2", "Q3", "Q4(most)"],
                             include_lowest=True)
            g = d.groupby("_q", observed=True)["exact_accuracy"].agg(
                n="size", sa_pct=lambda x: round(100 * x.mean(), 1))
            print(g.to_string())
    except (ValueError, IndexError):
        pass


def _budget_report(df: pd.DataFrame):
    """Grant economics -- only prints when the run was budgeted (utils/budget.py)."""
    if "budget_spent" not in df.columns:
        return
    d = df.dropna(subset=["budget_spent"]).copy()
    if d.empty:
        return
    cur = "$"
    money = lambda x: f"{cur}{x:,.0f}"
    start = d["starting_funds"].dropna()
    if len(start) and start.nunique() == 1:
        start_txt = money(start.iloc[0])
    elif len(start):
        start_txt = f"{money(start.min())}-{money(start.max())}, varies by module"
    else:
        start_txt = "(unknown)"

    print(f"\n--- budget: spend per trial (starting funds {start_txt}) ---")
    print(f"  {'agent':<22}{'spent':>12}{'remaining':>12}{'overspent':>11}{'billed reqs':>13}")

    def _spend_row(name, g):
        over = 100 * g["budget_overspent"].fillna(False).mean()
        print(f"  {name:<22}{money(g['budget_spent'].mean()):>12}{money(g['funds_remaining'].mean()):>12}"
              f"{over:>10.1f}%{g['num_billed_requests'].mean():>13.1f}")

    for b, g in d.groupby("agent_backend"):
        _spend_row(b, g)
    _spend_row("OVERALL", d)

    print("\n--- budget: SA% by module x agent (mean spend in parens) ---")
    backends = sorted(d["agent_backend"].unique())
    print(f"  {'module':<24}" + "".join(f"{b[:20]:>22}" for b in backends))
    for module in [m for m in MODULE_ORDER if m in d["module"].unique()]:
        row = f"  {module:<24}"
        for b in backends:
            cell = d[(d["module"] == module) & (d["agent_backend"] == b)]
            acc, _, _, _ = calculate_trial_stats(cell)
            row += (f"{acc*100:>13.1f} ({money(cell['budget_spent'].mean()):>6})"
                    if pd.notna(acc) else f"{'-':>22}")
        print(row)

    dd = d.dropna(subset=["exact_accuracy"])
    if not dd.empty:
        print("\n--- SA% vs. budget-spent quartile (raw) ---")
        try:
            q = dd["budget_spent"].quantile([0, .25, .5, .75, 1.0]).tolist()
            if len(set(q)) == 5:
                dd = dd.assign(_q=pd.cut(dd["budget_spent"], q,
                                         labels=["Q1(cheap)", "Q2", "Q3", "Q4(spendy)"],
                                         include_lowest=True))
                g = dd.groupby("_q", observed=True)["exact_accuracy"].agg(
                    n="size", sa_pct=lambda x: round(100 * x.mean(), 1))
                print(g.to_string())
        except (ValueError, IndexError):
            pass


def _verified_line(df: pd.DataFrame, model: str, budget: bool = False):
    labels = load_verified_labels(model, budget)
    if labels is None:
        sfx = "_budget" if budget else ""
        print(f"\n(no analysis/verdicts_{model}{sfx}.csv -- run `diagnostics.py verdicts --model {model}"
              f"{' --budget' if budget else ''}` for the sympy-checked number; SA above is raw judge "
              f"accuracy and may be inflated)")
        return

    lab = labels[["path", "verified_success", "raw_success"]]
    # Keep the verified stats consistent with the currently displayed df (e.g. subset_file filtering).
    shown = set(df["path"].astype(str)) if "path" in df.columns else None
    if shown is not None:
        lab = lab[lab["path"].astype(str).isin(shown)].reset_index(drop=True)

    print(f"\n--- raw vs. verified SA (from verdicts_{model}{'_budget' if budget else ''}.csv, "
          f"n={len(lab)}) ---")
    if lab.empty:
        print("  (no overlapping trials between verdicts CSV and the currently displayed rows)")
        return
    print(f"  raw judge SA      : {100*lab['raw_success'].mean():.1f}%")
    print(f"  sympy-verified SA : {100*lab['verified_success'].mean():.1f}%")
    flipped = int((lab["raw_success"] != lab["verified_success"]).sum())
    print(f"  {flipped} trials differ (judge-lenient credit / judge-strict misses)")

def scoreboard(model: str, result_dir: str, subset_file: str, show_verified: bool, refresh: bool,
               budget: bool = False):
    csv_path = results_by_trial_csv(budget)
    if refresh:
        update_results(model, result_dir, csv_path=csv_path)
    df = load_results_by_trial(model, csv_path=csv_path)
    if df.empty:
        raise SystemExit(f"No rows for {model} in {Path(csv_path).name}")
    df = df.replace([np.inf, -np.inf], np.nan)
    if subset_file:
        df = filter_to_subset(df, subset_file)
    df = _clean(df)

    print(f"\n{'='*70}\nScoreboard: {model}   (n={len(df)} trials"
          + (", budgeted" if budget else "")
          + (f", subset={Path(subset_file).name}" if subset_file else "") + f")\n{'='*70}")

    _cell_grid(df, model)
    _per_module(df)
    _agents_ab(df)
    _coverage(df)
    _resource_quicklook(df)
    if budget:
        _budget_report(df)
    if show_verified:
        _verified_line(df, model, budget)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--all", action="store_true", help="every model in --models_file")
    ap.add_argument("--models_file", default="configs/models.txt")
    ap.add_argument("--result_dir", default=None,
                    help="default: evaluation_results, or the budgeted tree when --budget is set")
    ap.add_argument("--budget", action="store_true",
                    help="score a budgeted run: read from the budget results tree, use a separate "
                         "results_by_trial CSV, and print the grant-economics section")
    ap.add_argument("--subset_file", default=None)
    ap.add_argument("--verified", action="store_true",
                    help="also show the sympy-verified SA (needs diagnostics.py verdicts to have run)")
    ap.add_argument("--no-refresh", action="store_true",
                    help="don't rescan trial JSONs, use results_by_trial.csv as-is")
    args = ap.parse_args()

    if args.all:
        models = read_models_from_file(Path(args.models_file))
        if not models:
            raise SystemExit(f"No models in {args.models_file}")
    elif args.model:
        models = [args.model]
    else:
        raise SystemExit("pass --model <name> or --all")

    result_dir = resolve_result_dir(args.result_dir, args.budget)
    for m in models:
        scoreboard(m, result_dir, args.subset_file, args.verified, not args.no_refresh, args.budget)


if __name__ == "__main__":
    main()
