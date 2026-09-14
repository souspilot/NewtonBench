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
import ast
import hashlib
import json
import os
import re
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


@contextmanager
def time_limit(seconds: float):
    """Abort the wrapped block after `seconds` (raises TimeoutError). No-op when
    SIGALRM is unavailable (non-Unix) or seconds <= 0. Guards the sympy calls in
    compute_verdicts / classify_mismatch, some of which -- sp.simplify on nested
    exp/log/power expressions -- can otherwise run effectively forever."""
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(_signum, _frame):
        raise TimeoutError(f"exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


DEFAULT_SYMPY_TIMEOUT = 20.0  # seconds per unique law pair

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
REPO_ROOT = ANALYSIS_DIR.parent
RESULTS_BY_TRIAL_CSV = str(ANALYSIS_DIR / "results_by_trial.csv")
RESULTS_BY_TRIAL_BUDGET_CSV = str(ANALYSIS_DIR / "results_by_trial_budget.csv")
AGGREGATED_SUMMARY_CSV = str(ANALYSIS_DIR / "aggregated_trial_summary.csv")

# Per-trial budget fields (present only in budgeted runs; see utils/budget.py).
BUDGET_COLS = ["budget_spent", "funds_remaining", "budget_overdraft", "budget_overspent",
               "num_billed_requests", "starting_funds"]

# The runner's sampling protocol keeps the latest four completed trajectories
# per task configuration. Infrastructure failures are attempts, not replacement
# scientific trials, and are reported separately.
TARGET_TRIALS_PER_CONFIG = 4
TRIAL_IDENTITY_COLS = [
    "module", "noise_level", "equation_difficulty", "model_system",
    "law_version", "agent_backend", "trial_id",
]
TRIAL_CONFIG_COLS = [c for c in TRIAL_IDENTITY_COLS if c != "trial_id"]


def analysis_path(name: str) -> str:
    """A path inside analysis/ (verdicts_<model>.csv, trace CSVs, ...), cwd-independent."""
    return str(ANALYSIS_DIR / name)


def budget_result_dir(config_path: Optional[str] = None) -> str:
    """Directory tree budgeted runs write to (configs/budget/budget.json -> results_dir);
    falls back to the documented default if the config can't be read."""
    try:
        import sys as _sys
        if str(REPO_ROOT) not in _sys.path:
            _sys.path.insert(0, str(REPO_ROOT))
        from utils.budget import load_budget_config
        return load_budget_config(config_path).results_dir
    except Exception:
        if config_path:
            raise
        return "budget_evaluation_results"


def resolve_result_dir(result_dir: Optional[str], budget: bool,
                       budget_config: Optional[str] = None) -> str:
    """`--result_dir` wins if given; otherwise pick the tree for the mode."""
    if result_dir:
        return result_dir
    return budget_result_dir(budget_config) if (budget or budget_config) else "evaluation_results"


def result_run_suffix(budget: bool, result_dir: Optional[str] = None) -> str:
    """Filesystem-safe analysis suffix identifying the exact result tree."""
    if not budget:
        return ""
    if not result_dir:
        return "_budget"
    name = Path(result_dir).resolve().name
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_.-") or "budget"
    return f"_{slug}"


def results_by_trial_csv(budget: bool, result_dir: Optional[str] = None) -> str:
    """Return a run-specific CSV path so different budgets cannot collide."""
    if not budget:
        return RESULTS_BY_TRIAL_CSV
    if not result_dir:
        return RESULTS_BY_TRIAL_BUDGET_CSV
    return analysis_path(f"results_by_trial{result_run_suffix(True, result_dir)}.csv")


def _budget_from_trial(data: dict) -> dict:
    """Pull the flat budget columns out of a trial JSON's `budget` block."""
    b = data.get("budget") or {}
    return dict(
        budget_spent=b.get("total_spent"),
        funds_remaining=b.get("funds_remaining"),
        budget_overdraft=b.get("overdraft"),
        budget_overspent=b.get("overspent"),
        num_billed_requests=b.get("num_billed_requests"),
        starting_funds=b.get("starting_funds"),
    )


# ---------------------------------------------------------------------------
# Trial loading (deduped, one row per logical config) -- used by diagnostics.py
# ---------------------------------------------------------------------------

def _path_version(trial_path: Path) -> int:
    """Trailing _vN of a config directory (.../vanilla_equation_noise0_0_v2 -> 2)."""
    config_dir = trial_path.parent.parent  # up from trials/
    m = re.search(r"v(\d+)$", str(config_dir).rstrip("/"))
    return int(m.group(1)) if m else 0


def load_trials(result_dir: str, model: str, include_fails: bool = False,
                max_trials_per_config: Optional[int] = TARGET_TRIALS_PER_CONFIG) -> pd.DataFrame:
    """Load a reproducible set of trial attempts from ``result_dir/model``.

    Completed ``trialN.json`` files are scientific trials. A completed trial
    with missing/malformed evaluation output is retained and assigned accuracy
    0; missing RMSLE never removes it from the accuracy denominator.

    ``trialN_fail.json`` files are runner/infrastructure failures after retries.
    They are excluded from scientific accuracy by default. Set
    ``include_fails=True`` for operational audits; a completed file always wins
    over a fail file with the same logical identity.

    Re-run directories are de-duplicated by logical identity. The latest
    ``_vN`` completed record is kept, then only the highest trial IDs up to
    ``max_trials_per_config`` are retained per task configuration. Pass None
    to disable that final cap (useful for failure-rate audits).
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
                raw_bytes = trial_path.read_bytes()
                data = json.loads(raw_bytes)
            except Exception as e:  # noqa: BLE001
                print(f"Skipping unreadable {trial_path}: {e}")
                continue

            ev = data.get("evaluation", {}) or {}
            raw_acc = ev.get("exact_accuracy")
            try:
                acc = float(raw_acc)
                if not np.isfinite(acc) or acc not in (0.0, 1.0):
                    raise ValueError
            except (TypeError, ValueError):
                # A completed but malformed submission is a scientific failure,
                # not missing data. This is the key denominator invariant.
                acc = 0.0
            m = re.search(r"trial(\d+)", trial_path.name)
            trial_id = data.get("trial_id")
            if trial_id is None and m:
                trial_id = int(m.group(1))
            rows.append(dict(
                path=str(trial_path),
                file_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                path_version=_path_version(trial_path),
                path_mtime=trial_path.stat().st_mtime,
                is_fail=is_fail,
                status=data.get("status"),
                trial_id=trial_id,
                module=data.get("module_name"),
                model_name=data.get("model_name", model),
                evaluated_model=data.get("evaluated_model", data.get("model_name", model)),
                judge_model=data.get("LLM judge"),
                noise_level=data.get("noise_level"),
                equation_difficulty=data.get("equation_difficulty"),
                model_system=data.get("model_system"),
                law_version=data.get("law_version"),
                agent_backend=data.get("agent_backend"),
                rmsle=ev.get("rmsle"),
                exact_accuracy=acc,
                symbolic_equivalent=ev.get("symbolic_equivalent"),
                symbolic_msg=ev.get("symbolic_msg"),
                evaluation_error=ev.get("error"),
                submitted_law=data.get("submitted_law"),
                ground_truth_law=ev.get("ground_truth_law"),
                rounds=data.get("rounds"),
                num_experiments=data.get("num_experiments"),
                total_tokens=data.get("total_tokens"),
                **_budget_from_trial(data),
            ))
    if not rows:
        raise SystemExit(f"No trial files found under {model_dir}")
    df = pd.DataFrame(rows)
    df["trial_id"] = pd.to_numeric(df["trial_id"], errors="coerce")
    df["rmsle"] = df["rmsle"].replace([np.inf, -np.inf], np.nan)

    # Prefer a completed trajectory to a fail stub for the same logical trial,
    # even if stale directories contain both. Within that class, latest _vN
    # and then latest mtime win deterministically.
    df["_completed_priority"] = (~df["is_fail"]).astype(int)
    before = len(df)
    df = (df.sort_values(["_completed_priority", "path_version", "path_mtime", "path"])
            .drop_duplicates(subset=TRIAL_IDENTITY_COLS, keep="last")
            .reset_index(drop=True))
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} stale duplicate trial(s) from older re-run directories "
              f"(preferred completed records, then the highest _vN version).")

    if not include_fails:
        df = df[~df["is_fail"]].copy()

    if max_trials_per_config is not None:
        if max_trials_per_config <= 0:
            raise ValueError("max_trials_per_config must be positive or None")
        # Trial IDs increase on every resumed run, so the highest IDs are the
        # same latest completed trajectories the runner retains.
        df = (df.sort_values("trial_id", ascending=False, na_position="last")
                .groupby(TRIAL_CONFIG_COLS, dropna=False, group_keys=False)
                .head(max_trials_per_config))

    return (df.drop(columns=["_completed_priority"])
              .sort_values(TRIAL_CONFIG_COLS + ["trial_id"], na_position="last")
              .reset_index(drop=True))


def load_trial_failures(result_dir: str, model: str) -> pd.DataFrame:
    """Return de-duplicated runner failures for operational reporting only."""
    attempts = load_trials(result_dir, model, include_fails=True,
                           max_trials_per_config=None)
    return attempts[attempts["is_fail"]].reset_index(drop=True)


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
    if np.all(np.isnan(data_for_stats)):
        return df
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

def _module_parameter_names(module_name: str) -> Optional[List[str]]:
    """Expected positional parameters from a benchmark module's public signature."""
    try:
        # Read prompts.py rather than importing the package. Importing a physics
        # module pulls in API clients that a standalone scoring machine should
        # not need merely to recover the public function signature.
        prompts_path = REPO_ROOT / "modules" / module_name / "prompts.py"
        tree = ast.parse(prompts_path.read_text(encoding="utf-8"))
        signature = None
        for stmt in tree.body:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                if any(isinstance(t, ast.Name) and t.id == "FUNCTION_SIGNATURE" for t in targets):
                    value = stmt.value
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        signature = value.value
                        break
        if signature is None:
            return None
        node = ast.parse(signature + "\n    pass").body[0]
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        return [a.arg for a in node.args.args]
    except Exception:  # noqa: BLE001
        return None


def load_adjudications(path: Optional[str]) -> Optional[pd.DataFrame]:
    """Load explicit human/independent-local-judge labels keyed by trial path.

    Required columns are ``path`` and ``adjudicated_success`` (0/1 or
    true/false). Optional ``adjudicator`` and ``notes`` columns are preserved.
    Duplicate paths or invalid labels fail loudly.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--adjudications not found: {path}")
    adj = pd.read_csv(p)
    required = {"path", "adjudicated_success"}
    missing = required - set(adj.columns)
    if missing:
        raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")
    if adj["path"].duplicated().any():
        dupes = adj.loc[adj["path"].duplicated(keep=False), "path"].tolist()
        raise SystemExit(f"{path} has duplicate path labels: {dupes[:5]}")

    def parse_label(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "pass", "yes"}:
            return True
        if text in {"false", "fail", "no"}:
            return False
        raise ValueError(f"invalid adjudicated_success label: {value!r}")

    try:
        adj["adjudicated_success"] = adj["adjudicated_success"].map(parse_label)
    except ValueError as exc:
        raise SystemExit(f"Invalid adjudication file {path}: {exc}") from exc
    if "adjudicator" not in adj.columns:
        adj["adjudicator"] = "unspecified"
    return adj


def compute_verdicts(df: pd.DataFrame, rmsle_threshold: float,
                     sympy_timeout: float = DEFAULT_SYMPY_TIMEOUT,
                     adjudications: Optional[pd.DataFrame] = None,
                     independent_judge: Optional[str] = None) -> pd.DataFrame:
    """Add judge_verdict / rmsle_verdict / structural_verdict / agreement_bucket.

    The deterministic structural result is the primary label whenever it can
    decide. ``not_checkable`` remains unresolved unless the caller explicitly
    names the independent judge stored in the trial tree. An explicit
    adjudications table may resolve any row and records its source.

    The structural check is run once per UNIQUE (submitted_law, ground_truth_law)
    pair (trials re-run the same config, so pairs repeat a lot) and each call is
    wrapped in time_limit(sympy_timeout) -> 'not_checkable' on a hang.
    """
    from structural_equivalence import check_constant_equivalence  # lazy: pulls sympy

    df = df.copy().reset_index(drop=True)
    df["judge_verdict"] = df["symbolic_equivalent"].fillna(False).astype(bool)
    df["rmsle_verdict"] = pd.to_numeric(df["rmsle"], errors="coerce") < rmsle_threshold

    pairs = {}
    uniq = df[["module", "submitted_law", "ground_truth_law"]].drop_duplicates()
    n_timeout = 0
    param_cache = {}
    for module, sub, gt in uniq.itertuples(index=False):
        key = (module, sub, gt)
        if not isinstance(sub, str) or not isinstance(gt, str):
            pairs[key] = "not_checkable"
            continue
        if module not in param_cache:
            param_cache[module] = _module_parameter_names(module)
        try:
            with time_limit(sympy_timeout):
                pairs[key] = check_constant_equivalence(
                    sub, gt, expected_param_names=param_cache[module])
        except Exception:  # noqa: BLE001  (TimeoutError included)
            pairs[key] = "not_checkable"
            n_timeout += 1
    if n_timeout:
        print(f"  ({n_timeout} law pair(s) hit the {sympy_timeout}s sympy timeout -> not_checkable)")
    df["structural_verdict"] = [pairs.get((m, s, g), "not_checkable")
                                for m, s, g in zip(df["module"], df["submitted_law"],
                                                   df["ground_truth_law"])]

    def bucket(row):
        j = row["judge_verdict"]
        sv = row["structural_verdict"]
        if sv == "constant_equivalent":
            return "consistent_pass" if j else "judge_strict"
        if sv == "structurally_different":
            return "judge_lenient" if j else "consistent_fail"
        return "unresolved"

    df["agreement_bucket"] = df.apply(bucket, axis=1)
    df["raw_success"] = df["exact_accuracy"].fillna(0.0) >= 0.5
    verified = pd.Series(pd.NA, index=df.index, dtype="boolean")
    source = pd.Series("unresolved", index=df.index, dtype="string")
    passed = df["structural_verdict"] == "constant_equivalent"
    failed = df["structural_verdict"] == "structurally_different"
    verified.loc[passed] = True
    verified.loc[failed] = False
    source.loc[passed | failed] = "deterministic_symbolic"

    if independent_judge:
        if "judge_model" not in df.columns:
            raise SystemExit("Independent-judge fallback requested, but trial rows do not record a judge model.")
        observed = df["judge_model"].dropna().astype(str)
        mismatched = observed.ne(independent_judge)
        missing = df["judge_model"].isna()
        if missing.any() or mismatched.any():
            found = sorted(observed.unique().tolist())
            raise SystemExit(
                f"Independent-judge fallback refused: expected every row to be judged by "
                f"{independent_judge!r}, found {found} and {int(missing.sum())} missing value(s)."
            )
        if "evaluated_model" not in df.columns or df["evaluated_model"].isna().any():
            raise SystemExit(
                "Independent-judge fallback requested, but one or more rows do not identify "
                "the evaluated model."
            )
        self_judged = df["evaluated_model"].astype(str).eq(independent_judge)
        if self_judged.any():
            raise SystemExit(
                f"Independent-judge fallback refused: {int(self_judged.sum())} row(s) were "
                f"evaluated by the same model, {independent_judge!r}."
            )
        unresolved = verified.isna()
        verified.loc[unresolved] = df.loc[unresolved, "judge_verdict"].astype(bool).to_numpy()
        source.loc[unresolved] = f"independent_judge:{independent_judge}"

    if adjudications is not None and not adjudications.empty:
        cols = ["path", "adjudicated_success", "adjudicator"]
        labels = adjudications[cols].rename(columns={"adjudicator": "_adjudicator"})
        df = df.merge(labels, on="path", how="left")
        mask = df["adjudicated_success"].notna()
        verified.loc[mask] = df.loc[mask, "adjudicated_success"].astype(bool).to_numpy()
        source.loc[mask] = ("adjudication:" +
                            df.loc[mask, "_adjudicator"].fillna("unspecified").astype(str))
        df = df.drop(columns=["_adjudicator"])

    df["verified_success"] = verified
    df["verification_source"] = source
    df["verification_resolved"] = verified.notna()
    return df


# ---------------------------------------------------------------------------
# verified_success label passing: diagnostics writes it, scoreboard reads it
# ---------------------------------------------------------------------------

def verdicts_csv_path(model: str, budget: bool = False,
                      result_dir: Optional[str] = None) -> str:
    return analysis_path(f"verdicts_{model}{result_run_suffix(budget, result_dir)}.csv")


def load_verified_labels(model: str, budget: bool = False,
                         result_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
    """The per-trial verified_success table diagnostics.py's `verdicts`
    subcommand writes. None if it was never run for this model."""
    p = verdicts_csv_path(model, budget, result_dir)
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
    """Compile the protocol-defined completed trials into results_by_trial.csv."""
    base_cols = [
        "path", "file_sha256", "trial_id", "module", "model_name", "noise_level", "equation_difficulty", "model_system",
        "law_version", "agent_backend", "rmsle", "exact_accuracy", "rounds",
        "experiments", "total_tokens", "file_version", "status", "evaluation_error",
    ] + BUDGET_COLS
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for c in base_cols:  # keep older CSVs forward-compatible
            if c not in df.columns:
                df[c] = np.nan
        # A model may be evaluated under several named run configs. Refresh means
        # "rebuild this model from result_dir", not "merge with whichever run was
        # analysed previously"; otherwise equal logical trial keys overwrite while
        # missing keys from the old run linger in the scoreboard.
        if "model_name" in df.columns:
            df = df[df["model_name"] != model_name].copy()
    else:
        df = pd.DataFrame(columns=base_cols)

    model_dir = os.path.join(result_dir, model_name)
    if not os.path.isdir(model_dir):
        print(f"Directory not found for model: {model_name}")
        return

    trials = load_trials(result_dir, model_name, include_fails=False,
                         max_trials_per_config=TARGET_TRIALS_PER_CONFIG)
    fresh = pd.DataFrame({
        "path": trials["path"],
        "file_sha256": trials["file_sha256"],
        "trial_id": trials["trial_id"],
        "module": trials["module"],
        "model_name": trials["model_name"].fillna(model_name),
        "noise_level": trials["noise_level"],
        "equation_difficulty": trials["equation_difficulty"],
        "model_system": trials["model_system"],
        "law_version": trials["law_version"],
        "agent_backend": trials["agent_backend"],
        "rmsle": trials["rmsle"],
        "exact_accuracy": trials["exact_accuracy"].fillna(0.0),
        "rounds": trials["rounds"],
        "experiments": trials["num_experiments"],
        "total_tokens": trials["total_tokens"],
        "file_version": trials["path_version"].map(lambda v: f"v{int(v)}"),
        "status": trials["status"],
        "evaluation_error": trials["evaluation_error"],
        **{c: trials[c] for c in BUDGET_COLS},
    })
    df = pd.concat([df, fresh], ignore_index=True)

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Results updated in {csv_path}")


def calculate_trial_stats(df):
    """Accuracy and RMSLE summaries with independent missing-data policies.

    Missing accuracy is a failed completed trial and therefore 0. Missing RMSLE
    is unavailable only for the RMSLE summary; it never changes the accuracy
    denominator.
    """
    if df.empty:
        return np.nan, np.nan, np.nan, np.nan
    group_cols = [c for c in ["module", "equation_difficulty", "model_system",
                              "law_version", "agent_backend", "trial_id"]
                  if c in df.columns]
    if not group_cols:
        group_cols = ["trial_id"]
    work = df.copy()
    work["exact_accuracy"] = pd.to_numeric(work["exact_accuracy"], errors="coerce").fillna(0.0)
    work["rmsle"] = pd.to_numeric(work["rmsle"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan)
    acc_means = work.groupby(group_cols, dropna=False)["exact_accuracy"].mean()
    rmsle_means = work.groupby(group_cols, dropna=False)["rmsle"].mean().dropna()
    if acc_means.empty:
        return np.nan, np.nan, np.nan, np.nan
    ma = acc_means.mean()
    sa = acc_means.std()
    mr = rmsle_means.mean() if not rmsle_means.empty else np.nan
    sr = rmsle_means.std() if not rmsle_means.empty else np.nan
    return ma, (0 if np.isnan(sa) else sa), mr, (0 if np.isnan(sr) else sr)


def load_results_by_trial(model: Optional[str] = None,
                          csv_path: str = RESULTS_BY_TRIAL_CSV) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise SystemExit(f"{csv_path} not found -- run scoreboard.py (it refreshes it) first.")
    df = pd.read_csv(csv_path).replace([np.inf, -np.inf], np.nan)
    if model:
        df = df[df["model_name"] == model].reset_index(drop=True)
    return df
