"""
Shared plumbing for the NewtonBench analysis entry points.

Two entry points sit on top of this module:

  * scoreboard.py  -- fast, headline numbers. Refreshes results_by_trial.csv
                      and prints per-cell / per-module / per-agent symbolic
                      accuracy. No sympy, no transcript parsing.
  * diagnostics.py -- slow, "where is the model going wrong" numbers, with
                      example trajectories. Judge-vs-sympy agreement buckets,
                      mistake taxonomy, chat-history trace mining, agent
                      divergence.

Everything both need -- trial loading with re-run-directory dedup, the
representative-subset filter, RMSLE outlier cleaning, the deterministic
verdict/agreement-bucket logic, and the results_by_trial.csv builder -- lives
here so neither entry point imports the other.

`compute_verdicts` imports structural_equivalence lazily, so importing this
module does NOT pull in sympy: scoreboard.py stays sympy-free.
"""
import json
import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

DEFAULT_RMSLE_THRESHOLD = 1e-3
MAX_TURNS = 10  # every module's system prompt states "up to 10 rounds"

DIFFICULTIES = ["easy", "medium", "hard"]
SYSTEMS = ["vanilla_equation", "simple_system", "complex_system"]
SYS_SHORT = {"vanilla_equation": "VanEq", "simple_system": "SimpS", "complex_system": "CompS"}
AGENT_SHORT = {"vanilla_agent": "Vanilla", "code_assisted_agent": "CodeAst", "planned_agent": "Planned"}
MODULE_ORDER = [
    "m0_gravity", "m1_coulomb_force", "m2_magnetic_force", "m3_fourier_law",
    "m4_snell_law", "m5_radioactive_decay", "m6_underdamped_harmonic",
    "m7_malus_law", "m8_sound_speed", "m9_hooke_law",
    "m10_be_distribution", "m11_heat_transfer",
]
MODULE_SHORT = {
    "m0_gravity": "Gravity", "m1_coulomb_force": "Coulomb", "m2_magnetic_force": "Magnetic",
    "m3_fourier_law": "Fourier", "m4_snell_law": "Snell", "m5_radioactive_decay": "Radioact",
    "m6_underdamped_harmonic": "Harmonic", "m7_malus_law": "Malus", "m8_sound_speed": "Sound",
    "m9_hooke_law": "Hooke", "m10_be_distribution": "BoseEin", "m11_heat_transfer": "HeatTr",
}

ANALYSIS_DIR = Path(__file__).resolve().parent  # so paths work regardless of cwd
RESULTS_BY_TRIAL_CSV = str(ANALYSIS_DIR / "results_by_trial.csv")
AGGREGATED_SUMMARY_CSV = str(ANALYSIS_DIR / "aggregated_trial_summary.csv")


def analysis_path(name: str) -> str:
    """A path inside analysis/ (verdicts_<model>.csv, trace CSVs, ...), cwd-independent."""
    return str(ANALYSIS_DIR / name)


# ---------------------------------------------------------------------------
# Trial loading (deduped, one row per logical config) -- used by diagnostics.py
# ---------------------------------------------------------------------------

def _path_version(trial_path: Path) -> int:
    """Trailing _vN of a config directory (.../vanilla_equation_noise0_0_v2 -> 2)."""
    config_dir = trial_path.parent.parent  # up from trials/
    m = re.search(r"v(\d+)$", str(config_dir).rstrip("/"))
    return int(m.group(1)) if m else 0


def load_trials(result_dir: str, model: str, include_fails: bool = False) -> pd.DataFrame:
    """One row per real trial JSON under evaluation_results/<model>/.

    include_fails=True keeps *_fail.json (round-budget-exhausted trials, stub
    nan submission) with is_fail=True and exact_accuracy defaulted to 0.0 -- a
    trial that burned its budget and failed IS the data point a "did hitting
    the round limit correlate with failure" question needs.

    De-dupes to one row per (module, difficulty, system, law_version,
    agent_backend, trial_id), keeping the highest _vN config directory, so
    stale re-run directories don't get counted as independent trials.
    """
    model_dir = Path(result_dir) / model
    if not model_dir.is_dir():
        raise SystemExit(f"No such directory: {model_dir}")

    rows = []
    for trials_dir in model_dir.rglob("trials"):
        for trial_path in sorted(trials_dir.glob("trial*.json")):
            is_fail = trial_path.name.endswith("_fail.json")
            if is_fail and not include_fails:
                continue
            try:
                with open(trial_path) as f:
                    data = json.load(f)
            except Exception as e:  # noqa: BLE001
                print(f"Skipping unreadable {trial_path}: {e}")
                continue

            ev = data.get("evaluation", {}) or {}
            acc = ev.get("exact_accuracy")
            if acc is None and is_fail:
                acc = 0.0
            rows.append(dict(
                path=str(trial_path),
                path_version=_path_version(trial_path),
                is_fail=is_fail,
                status=data.get("status"),
                trial_id=data.get("trial_id"),
                module=data.get("module_name"),
                equation_difficulty=data.get("equation_difficulty"),
                model_system=data.get("model_system"),
                law_version=data.get("law_version"),
                agent_backend=data.get("agent_backend"),
                rmsle=ev.get("rmsle"),
                exact_accuracy=acc,
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

    identity_cols = ["module", "equation_difficulty", "model_system", "law_version",
                     "agent_backend", "trial_id"]
    before = len(df)
    df = (df.sort_values("path_version")
            .drop_duplicates(subset=identity_cols, keep="last")
            .reset_index(drop=True))
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} stale duplicate trial(s) from older re-run directories "
              f"(kept the highest _vN version per logical config).")
    return df


# ---------------------------------------------------------------------------
# Representative-subset filter (works on trial-level or CSV-level frames:
# only needs module / equation_difficulty / model_system columns)
# ---------------------------------------------------------------------------

def load_subset_cells(subset_file: Optional[str]):
    """{module: {(difficulty, system), ...}} from a representative_subset.json,
    or None when not given. Same JSON shape as run_all_evaluations.py."""
    if not subset_file:
        return None
    p = Path(subset_file)
    if not p.exists():
        raise SystemExit(f"--subset_file not found: {subset_file}")
    with open(p) as f:
        raw = json.load(f)
    return {m: {(c["difficulty"], c["system"]) for c in cells} for m, cells in raw.items()}


def filter_to_subset(df: pd.DataFrame, subset_file: Optional[str]) -> pd.DataFrame:
    """Keep only rows whose (module, equation_difficulty, model_system) is a
    whitelisted cell in subset_file. No-op when subset_file is None.

    Needed because trial loading reads EVERY trial JSON on disk: if a run's
    config changed mid-flight (representative_subset_big.json ->
    representative_subset.json), the model directory mixes cell coverage and
    per-cell aggregates silently pool trials from configs never meant to be
    compared.
    """
    cells = load_subset_cells(subset_file)
    if cells is None:
        return df
    allowed = {(module, difficulty, system)
               for module, module_cells in cells.items()
               for (difficulty, system) in module_cells}
    idx = pd.MultiIndex.from_frame(df[["module", "equation_difficulty", "model_system"]])
    mask = idx.isin(allowed)
    dropped = int((~mask).sum())
    if dropped:
        print(f"Subset filter ({subset_file}): kept {int(mask.sum())}/{len(df)} rows in "
              f"whitelisted (module, difficulty, system) cells, dropped {dropped} out-of-subset.")
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# RMSLE outlier cleaning (Modified Z-Score, per cell) -- shared by both
# ---------------------------------------------------------------------------

def detect_outliers_modified_zscore_column(df, column_name, threshold=3.5):
    """Mask Modified-Z-Score outliers (and non-finite values) in one column as NaN."""
    data = df[column_name].values
    if len(data) == 0:
        return df
    data_for_stats = np.where(np.isinf(data), np.nan, data)
    median = np.nanmedian(data_for_stats)
    mad = np.nanmedian(np.abs(data_for_stats - median))
    if mad == 0:
        outlier_mask = ~np.isfinite(data)
    else:
        modified_z_scores = 0.6745 * (data - median) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        outlier_mask |= ~np.isfinite(modified_z_scores)
    df.loc[outlier_mask, column_name] = np.nan
    return df


def clean_rmsle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rmsle_cleaned"] = df["rmsle"]
    cleaned = []
    for _, g in df.groupby(["module", "equation_difficulty", "model_system", "agent_backend"]):
        g2 = g.copy()
        g2["rmsle_cleaned"] = g2["rmsle"]
        g2 = detect_outliers_modified_zscore_column(g2, "rmsle_cleaned")
        cleaned.append(g2)
    return pd.concat(cleaned).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Deterministic verdicts / agreement buckets -- diagnostics.py only
# (lazy sympy import keeps scoreboard.py sympy-free)
# ---------------------------------------------------------------------------

def compute_verdicts(df: pd.DataFrame, rmsle_threshold: float) -> pd.DataFrame:
    """Add judge_verdict / rmsle_verdict / structural_verdict / agreement_bucket.

    agreement_bucket trusts the deterministic sympy structural check whenever
    it reached a verdict; the RMSLE threshold only decides not_checkable rows.
    Buckets: consistent_pass, consistent_fail, judge_lenient (judge says
    equivalent, sympy disagrees), judge_strict (judge says wrong, fit is exact).
    """
    from structural_equivalence import check_constant_equivalence  # lazy: pulls sympy

    df = df.copy()
    df["judge_verdict"] = df["symbolic_equivalent"].fillna(False).astype(bool)
    df["rmsle_verdict"] = df["rmsle"] < rmsle_threshold

    structural = []
    for _, row in df.iterrows():
        if not isinstance(row["submitted_law"], str) or not isinstance(row["ground_truth_law"], str):
            structural.append("not_checkable")
            continue
        structural.append(check_constant_equivalence(row["submitted_law"], row["ground_truth_law"]))
    df["structural_verdict"] = structural

    def bucket(row):
        j = row["judge_verdict"]
        sv = row["structural_verdict"]
        if sv == "constant_equivalent":
            return "consistent_pass" if j else "judge_strict"
        if sv == "structurally_different":
            return "judge_lenient" if j else "consistent_fail"
        r = row["rmsle_verdict"]
        if j and r:
            return "consistent_pass"
        if not j and not r:
            return "consistent_fail"
        if j and not r:
            return "judge_lenient"
        return "judge_strict"

    df["agreement_bucket"] = df.apply(bucket, axis=1)
    df["raw_success"] = df["exact_accuracy"].fillna(0.0) >= 0.5
    df["verified_success"] = df["agreement_bucket"] == "consistent_pass"
    return df


# ---------------------------------------------------------------------------
# verified_success label passing: diagnostics writes it, scoreboard reads it
# ---------------------------------------------------------------------------

def verdicts_csv_path(model: str) -> str:
    return analysis_path(f"verdicts_{model}.csv")


def load_verified_labels(model: str) -> Optional[pd.DataFrame]:
    """The per-trial verified_success table diagnostics.py's `verdicts`
    subcommand writes. None if it was never run for this model."""
    p = verdicts_csv_path(model)
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# results_by_trial.csv builder + per-cell aggregation -- scoreboard.py
# ---------------------------------------------------------------------------

def extract_version_from_path(results_dir: str) -> str:
    m = re.search(r"v(\d+)$", results_dir.rstrip("/"))
    return f"v{m.group(1)}" if m else "v_unknown"


def read_models_from_file(models_file: Path) -> List[str]:
    if not models_file.exists():
        return []
    out = []
    with models_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def update_results(model_name: str, result_dir: str, csv_path: str = RESULTS_BY_TRIAL_CSV):
    """Compile every non-fail trial JSON for one model into results_by_trial.csv,
    upserting on the logical-config key (so re-running is idempotent)."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=[
            "trial_id", "module", "model_name", "noise_level", "equation_difficulty", "model_system",
            "law_version", "agent_backend", "rmsle", "exact_accuracy", "rounds",
            "experiments", "total_tokens", "file_version",
        ])

    model_dir = os.path.join(result_dir, model_name)
    if not os.path.isdir(model_dir):
        print(f"Directory not found for model: {model_name}")
        return

    for module in os.listdir(model_dir):
        module_path = os.path.join(model_dir, module)
        if not os.path.isdir(module_path):
            continue
        for root, dirs, _files in os.walk(module_path):
            if "trials" not in dirs:
                continue
            trials_dir = os.path.join(root, "trials")
            for file in os.listdir(trials_dir):
                if not file.endswith(".json") or "fail" in file:
                    continue
                m = re.search(r"trial(\d+)", file)
                if not m:
                    continue
                with open(os.path.join(trials_dir, file)) as f:
                    data = json.load(f)
                new_row = {
                    "trial_id": int(m.group(1)),
                    "module": data.get("module_name"),
                    "model_name": data.get("model_name"),
                    "noise_level": data.get("noise_level"),
                    "equation_difficulty": data.get("equation_difficulty"),
                    "model_system": data.get("model_system"),
                    "law_version": data.get("law_version"),
                    "agent_backend": data.get("agent_backend"),
                    "rmsle": data.get("evaluation", {}).get("rmsle"),
                    "exact_accuracy": data.get("evaluation", {}).get("exact_accuracy"),
                    "rounds": data.get("rounds"),
                    "experiments": data.get("num_experiments"),
                    "total_tokens": data.get("total_tokens"),
                    "file_version": extract_version_from_path(root),
                }
                mask = (
                    (df["trial_id"] == new_row["trial_id"])
                    & (df["module"] == new_row["module"])
                    & (df["model_name"] == new_row["model_name"])
                    & (df["noise_level"] == new_row["noise_level"])
                    & (df["equation_difficulty"] == new_row["equation_difficulty"])
                    & (df["model_system"] == new_row["model_system"])
                    & (df["law_version"] == new_row["law_version"])
                    & (df["agent_backend"] == new_row["agent_backend"])
                )
                if mask.any():
                    df.loc[mask, list(new_row.keys())] = list(new_row.values())
                else:
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Results updated in {csv_path}")


def calculate_trial_stats(df):
    """Mean/std of per-trial-id mean accuracy & rmsle (the paper's trial-then-mean)."""
    if df.empty:
        return np.nan, np.nan, np.nan, np.nan
    trial_means = df.groupby("trial_id").agg(
        mean_accuracy=("exact_accuracy", "mean"),
        mean_rmsle=("rmsle", "mean"),
    ).dropna()
    if trial_means.empty:
        return np.nan, np.nan, np.nan, np.nan
    ma = trial_means["mean_accuracy"].mean()
    sa = trial_means["mean_accuracy"].std()
    mr = trial_means["mean_rmsle"].mean()
    sr = trial_means["mean_rmsle"].std()
    return ma, (0 if np.isnan(sa) else sa), mr, (0 if np.isnan(sr) else sr)


def load_results_by_trial(model: Optional[str] = None,
                          csv_path: str = RESULTS_BY_TRIAL_CSV) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise SystemExit(f"{csv_path} not found -- run scoreboard.py (it refreshes it) first.")
    df = pd.read_csv(csv_path).replace([np.inf, -np.inf], np.nan)
    if model:
        df = df[df["model_name"] == model].reset_index(drop=True)
    return df
