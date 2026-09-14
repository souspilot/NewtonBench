"""
NewtonBench headline numbers -- fast. Refreshes analysis/results_by_trial.csv
from the trial JSONs on disk, then prints symbolic accuracy by cell, by
module, and by agent, plus coverage and a resource quick-look.

No sympy, no transcript parsing. For "where is the model going wrong",
"is that 90% judge-inflated", mistake taxonomy, etc. -> diagnostics.py.

Raw exact-accuracy is provisional stored-judge scoring. For publication,
run `diagnostics.py verdicts --model X`, explicitly adjudicate every unresolved
row, then pass --verified. Publication mode fails closed if any label is absent.

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
    resolve_result_dir, results_by_trial_csv, load_trial_failures, verdicts_csv_path,
    DIFFICULTIES, SYSTEMS, MODULE_ORDER,
)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell Modified-Z-Score RMSLE outlier masking, same as the old
    summarize_results.aggregate_results."""
    out = []
    for _, g in df.groupby(["module", "equation_difficulty", "model_system", "agent_backend"]):
        out.append(detect_outliers_modified_zscore_column(g.copy(), "rmsle"))
    return pd.concat(out).reset_index(drop=True) if out else df


def _apply_verified_labels(df: pd.DataFrame, model: str, budget: bool,
                           result_dir: str) -> pd.DataFrame:
    """Replace raw judge accuracy with fully resolved publication labels.

    Fails closed when the verdict cache is absent, stale, or unresolved. This
    prevents a point estimate conditional on only SymPy-checkable examples.
    """
    labels = load_verified_labels(model, budget, result_dir)
    if labels is None:
        raise SystemExit(
            f"No verdict cache for {model}. Run `python analysis/diagnostics.py verdicts "
            f"--model {model}{' --budget' if budget else ''}` first."
        )
    if not {"path", "file_sha256"}.issubset(df.columns):
        raise SystemExit("The results CSV predates content-keyed scoring; rerun without --no-refresh.")
    required = {"path", "file_sha256", "verified_success"}
    if not required.issubset(labels.columns):
        raise SystemExit("Verdict cache uses an obsolete schema; rerun diagnostics.py verdicts.")
    lab = labels[["path", "file_sha256", "verified_success"]].copy()
    lab = lab.rename(columns={"file_sha256": "verdict_sha256"})
    lab["verified_success"] = lab["verified_success"].astype(str).str.lower().map(
        {"true": True, "false": False, "1.0": True, "0.0": False,
         "1": True, "0": False})
    merged = df.merge(lab, on="path", how="left", validate="one_to_one")
    stale = merged["verdict_sha256"].notna() & (
        merged["file_sha256"].astype(str) != merged["verdict_sha256"].astype(str))
    unresolved = merged["verified_success"].isna() | stale
    if unresolved.any():
        examples = merged.loc[unresolved, "path"].head(3).tolist()
        raise SystemExit(
            f"Publication scoring refused: {int(unresolved.sum())}/{len(merged)} displayed "
            f"trials lack resolved symbolic labels. Adjudicate the unresolved CSV and rerun "
            f"diagnostics.py verdicts --adjudications <file>. Examples: {examples}"
        )
    merged["original_judge_accuracy"] = merged["exact_accuracy"]
    merged["exact_accuracy"] = merged["verified_success"].astype(float)
    return merged.drop(columns=["verified_success", "verdict_sha256"])


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
    print("\n--- agent A/B (same scoring mode as scoreboard header) ---")
    for b in backends:
        bdf = df[df["agent_backend"] == b]
        acc, std, _, _ = calculate_trial_stats(bdf)
        print(f"  {b:<22} {acc*100:>5.1f}%  (n_trials={len(bdf)}, "
              f"{bdf['rounds'].mean():.1f} rounds, {bdf['experiments'].mean():.0f} exp, "
              f"{bdf['total_tokens'].mean():,.0f} tok)")


def _coverage(df: pd.DataFrame, failures: pd.DataFrame = None):
    print("\n--- coverage (trials on disk per module x agent) ---")
    ct = df.pivot_table(index="module", columns="agent_backend", values="trial_id",
                        aggfunc="count", fill_value=0)
    print(ct.to_string())
    missing = sorted(set(MODULE_ORDER) - set(df["module"].unique()))
    if missing:
        print(f"  modules with NO trials: {missing}")
    if failures is not None and not failures.empty:
        print("\n--- operational failures after runner retries (excluded from SA) ---")
        failed = failures.groupby("agent_backend").size()
        for agent in sorted(failed.index):
            n_fail = int(failed.get(agent, 0))
            print(f"  {agent:<22} {n_fail:>4} failure record(s) on disk")


def _resource_quicklook(df: pd.DataFrame):
    d = df.dropna(subset=["total_tokens", "exact_accuracy"]).copy()
    if d.empty:
        return
    print("\n--- SA% vs. total_tokens quartile ---")
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
        print("\n--- SA% vs. budget-spent quartile ---")
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


def _verified_line(df: pd.DataFrame, model: str, budget: bool = False,
                   result_dir: str = None):
    labels = load_verified_labels(model, budget, result_dir)
    if labels is None:
        expected = Path(verdicts_csv_path(model, budget, result_dir)).name
        print(f"\n(no analysis/{expected} -- run `diagnostics.py verdicts --model {model}"
              f"{' --budget' if budget else ''}` for the sympy-checked number; SA above is raw judge "
              f"accuracy and may be inflated)")
        return

    needed = ["path", "verified_success", "raw_success"]
    if "verification_resolved" in labels.columns:
        needed.append("verification_resolved")
    lab = labels[needed].copy()
    # Keep the verified stats consistent with the currently displayed df (e.g. subset_file filtering).
    shown = set(df["path"].astype(str)) if "path" in df.columns else None
    if shown is not None:
        lab = lab[lab["path"].astype(str).isin(shown)].reset_index(drop=True)

    verdict_name = Path(verdicts_csv_path(model, budget, result_dir)).name
    print(f"\n--- raw vs. verified SA (from {verdict_name}, "
          f"n={len(lab)}) ---")
    if lab.empty:
        print("  (no overlapping trials between verdicts CSV and the currently displayed rows)")
        return
    raw = lab["raw_success"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}).fillna(False)
    verified = lab["verified_success"].astype(str).str.lower().map(
        {"true": True, "false": False, "1.0": True, "0.0": False,
         "1": True, "0": False})
    resolved = verified.notna()
    n = len(lab)
    successes = int(verified.fillna(False).sum())
    unresolved = int((~resolved).sum())
    print(f"  stored judge SA   : {100*raw.mean():.1f}%")
    print(f"  resolved labels   : {int(resolved.sum())}/{n} ({100*resolved.mean():.1f}%)")
    if resolved.any():
        print(f"  SA among resolved : {100*verified[resolved].mean():.1f}%")
    print(f"  all-trial SA range: {100*successes/n:.1f}-{100*(successes+unresolved)/n:.1f}%")
    flipped = int((raw[resolved] != verified[resolved]).sum())
    print(f"  {flipped} resolved trials differ from the stored judge")

def scoreboard(model: str, result_dir: str, subset_file: str, show_verified: bool, refresh: bool,
               budget: bool = False):
    csv_path = results_by_trial_csv(budget, result_dir)
    if refresh:
        update_results(model, result_dir, csv_path=csv_path)
    df = load_results_by_trial(model, csv_path=csv_path)
    if df.empty:
        raise SystemExit(f"No rows for {model} in {Path(csv_path).name}")
    df = df.replace([np.inf, -np.inf], np.nan)
    if subset_file:
        df = filter_to_subset(df, subset_file)
    if show_verified:
        df = _apply_verified_labels(df, model, budget, result_dir)
    df = _clean(df)
    failures = load_trial_failures(result_dir, model)
    if subset_file and not failures.empty:
        failures = filter_to_subset(failures, subset_file)

    score_label = "VERIFIED CASCADE" if show_verified else "PROVISIONAL STORED-JUDGE"
    print(f"\n{'='*70}\nScoreboard: {model}   [{score_label}]   (n={len(df)} trials"
          + (", budgeted" if budget else "")
          + (f", subset={Path(subset_file).name}" if subset_file else "") + f")\n{'='*70}")

    _cell_grid(df, model)
    _per_module(df)
    _agents_ab(df)
    _coverage(df, failures)
    _resource_quicklook(df)
    if budget:
        _budget_report(df)
    if show_verified:
        _verified_line(df, model, budget, result_dir)


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
    ap.add_argument("--budget-config", default=None,
                    help="Budget run JSON whose results_dir should be analysed; also enables --budget")
    ap.add_argument("--subset_file", default=None)
    ap.add_argument("--verified", action="store_true",
                    help="publication mode: score with resolved symbolic/adjudicated labels and "
                         "refuse output if any displayed trial remains unresolved")
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

    args.budget = args.budget or bool(args.budget_config)
    result_dir = resolve_result_dir(args.result_dir, args.budget, args.budget_config)
    for m in models:
        scoreboard(m, result_dir, args.subset_file, args.verified, not args.no_refresh, args.budget)


if __name__ == "__main__":
    main()
