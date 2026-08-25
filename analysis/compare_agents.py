#!/usr/bin/env python3
"""
Agent Comparison Script for NewtonBench
========================================
Compares vanilla_agent, code_assisted_agent, and planned_agent side by
side across all modules, difficulties, versions, and system types.

Produces:
  1. Summary table: SA% per (module, system_type, difficulty) for each agent
  2. Divergence report: trials where agents disagree (one succeeds, other fails)
  3. Optional: extract chat logs for divergent cases

Usage:
    python compare_agents.py
    python compare_agents.py --model qwq-32b --dump-divergent
    python compare_agents.py --model qwq-32b --module m0_gravity
"""

import argparse
import json
import os
import glob
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AGENTS = ["vanilla_agent", "code_assisted_agent", "planned_agent"]
AGENT_SHORT = {"vanilla_agent": "Vanilla", "code_assisted_agent": "CodeAst", "planned_agent": "Planned"}

ALL_MODULES = [
    "m0_gravity", "m1_coulomb_force", "m2_magnetic_force", "m3_fourier_law",
    "m4_snell_law", "m5_radioactive_decay", "m6_underdamped_harmonic",
    "m7_malus_law", "m8_sound_speed", "m9_hooke_law",
    "m10_be_distribution", "m11_heat_transfer",
]
MODULE_SHORT = {
    "m0_gravity": "Gravity", "m1_coulomb_force": "Coulomb",
    "m2_magnetic_force": "Magnetic", "m3_fourier_law": "Fourier",
    "m4_snell_law": "Snell", "m5_radioactive_decay": "Radioact",
    "m6_underdamped_harmonic": "Harmonic", "m7_malus_law": "Malus",
    "m8_sound_speed": "Sound", "m9_hooke_law": "Hooke",
    "m10_be_distribution": "BoseEin", "m11_heat_transfer": "HeatTr",
}
DIFFICULTIES = ["easy", "medium", "hard"]
VERSIONS = ["v0", "v1", "v2"]
SYSTEM_TYPES = ["vanilla_equation", "simple_system", "complex_system"]
SYS_SHORT = {"vanilla_equation": "VanEq", "simple_system": "SimpS", "complex_system": "CompS"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_trials(base_dir: str, model: str) -> Dict:
    """
    Load all trial-level results.
    Returns: dict keyed by (agent, module, difficulty, version, system_type)
             → list of trial dicts
    """
    trials = defaultdict(list)
    model_dir = Path(base_dir) / model

    if not model_dir.exists():
        print(f"ERROR: {model_dir} not found")
        return trials

    for module_dir in sorted(model_dir.iterdir()):
        if not module_dir.is_dir():
            continue
        module = module_dir.name

        for agent_dir in module_dir.iterdir():
            if not agent_dir.is_dir() or agent_dir.name not in AGENTS:
                continue
            agent = agent_dir.name

            for diff_dir in agent_dir.iterdir():
                if not diff_dir.is_dir():
                    continue
                difficulty = diff_dir.name

                for ver_dir in diff_dir.iterdir():
                    if not ver_dir.is_dir():
                        continue
                    version = ver_dir.name

                    for sys_dir in ver_dir.iterdir():
                        if not sys_dir.is_dir():
                            continue
                        sys_name = sys_dir.name.split("_noise")[0]
                        trials_dir = sys_dir / "trials"
                        if not trials_dir.exists():
                            continue

                        for tf in sorted(trials_dir.glob("trial*.json")):
                            if "_chat_history" in tf.name:
                                continue
                            try:
                                with open(tf) as f:
                                    data = json.load(f)
                                data["_source_file"] = str(tf)
                                data["_chat_log"] = str(tf).replace(".json", "_chat_history.log")
                                data["_agent"] = agent
                                data["_module"] = module
                                data["_difficulty"] = difficulty
                                data["_version"] = version
                                data["_system_type"] = sys_name
                                key = (agent, module, difficulty, version, sys_name)
                                trials[key].append(data)
                            except Exception:
                                pass
    return trials


def _get_accuracy(trial: Dict) -> float:
    """Extract exact_accuracy from a trial, handling nesting."""
    ev = trial.get("evaluation", {})
    if isinstance(ev, dict):
        return float(ev.get("exact_accuracy", 0.0))
    return 0.0


def _get_rmsle(trial: Dict) -> float:
    ev = trial.get("evaluation", {})
    if isinstance(ev, dict):
        return float(ev.get("rmsle", float("nan")))
    return float("nan")


def _get_law(trial: Dict) -> str:
    return trial.get("submitted_law", "N/A")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(trials: Dict, modules: List[str]):
    print("\n" + "=" * 120)
    print("AGENT COMPARISON: Symbolic Accuracy (%) by Configuration")
    print("=" * 120)

    # Aggregate: for each (agent, module, system_type, difficulty),
    # average SA% across versions and trials
    agg = defaultdict(list)
    for (agent, module, diff, ver, sys), trial_list in trials.items():
        for t in trial_list:
            agg[(agent, module, sys, diff)].append(_get_accuracy(t))

    for module in modules:
        mod_has_data = any(
            k[1] == module for k in agg
        )
        if not mod_has_data:
            continue

        print(f"\n{'─' * 120}")
        print(f"  {MODULE_SHORT.get(module, module)} ({module})")
        print(f"{'─' * 120}")

        # Header
        header = f"  {'System':<8} {'Diff':<7}"
        for agent in AGENTS:
            short = AGENT_SHORT[agent]
            header += f" │ {short + ' SA%':>10} {'tok':>6} {'exp':>5}"
        header += " │ Best"
        print(header)
        print(f"  {'':─<8} {'':─<7}" + (" ┼ " + "─" * 10 + " " + "─" * 6 + " " + "─" * 5) * len(AGENTS) + " ┼ " + "─" * 8)

        for sys_type in SYSTEM_TYPES:
            for diff in DIFFICULTIES:
                row = f"  {SYS_SHORT[sys_type]:<8} {diff:<7}"
                sa_values = {}
                for agent in AGENTS:
                    key = (agent, module, sys_type, diff)
                    vals = agg.get(key, [])
                    if vals:
                        sa = statistics.mean(vals) * 100
                        sa_values[agent] = sa
                        # Compute avg tokens and experiments
                        tkeys = [(agent, module, diff, v, sys_type) for v in VERSIONS]
                        all_t = []
                        for tk in tkeys:
                            all_t.extend(trials.get(tk, []))
                        avg_tok = statistics.mean([t.get("total_tokens", 0) for t in all_t]) / 1000 if all_t else 0
                        avg_exp = statistics.mean([t.get("num_experiments", 0) for t in all_t]) if all_t else 0
                        row += f" │ {sa:>10.1f} {avg_tok:>5.1f}k {avg_exp:>5.0f}"
                    else:
                        row += f" │ {'—':>10} {'—':>6} {'—':>5}"

                # Highlight best
                if sa_values:
                    best_sa = max(sa_values.values())
                    best_agents = [AGENT_SHORT[a] for a, v in sa_values.items() if v == best_sa]
                    row += f" │ {','.join(best_agents)}"
                else:
                    row += f" │ —"
                print(row)

    # Overall averages
    print(f"\n{'═' * 120}")
    print("OVERALL AVERAGES (across all available modules)")
    print(f"{'═' * 120}")
    header = f"  {'System':<8} {'Diff':<7}"
    for agent in AGENTS:
        header += f" │ {AGENT_SHORT[agent] + ' SA%':>10}"
    print(header)
    print(f"  {'':─<8} {'':─<7}" + (" ┼ " + "─" * 10) * len(AGENTS))

    for sys_type in SYSTEM_TYPES:
        for diff in DIFFICULTIES:
            row = f"  {SYS_SHORT[sys_type]:<8} {diff:<7}"
            for agent in AGENTS:
                vals = []
                for module in modules:
                    key = (agent, module, sys_type, diff)
                    vals.extend(agg.get(key, []))
                if vals:
                    row += f" │ {statistics.mean(vals) * 100:>10.1f}"
                else:
                    row += f" │ {'—':>10}"
            print(row)

    # Grand average
    print(f"  {'':─<8} {'':─<7}" + (" ┼ " + "─" * 10) * len(AGENTS))
    row = f"  {'ALL':<8} {'ALL':<7}"
    for agent in AGENTS:
        vals = []
        for key, v in agg.items():
            if key[0] == agent:
                vals.extend(v)
        if vals:
            row += f" │ {statistics.mean(vals) * 100:>10.1f}"
        else:
            row += f" │ {'—':>10}"
    print(row)


# ---------------------------------------------------------------------------
# Divergence analysis
# ---------------------------------------------------------------------------

def find_divergent_trials(trials: Dict) -> List[Dict]:
    """
    Find configurations where agents disagree: one gets 100% accuracy
    on a trial set while another gets <100%, or vice versa.

    Returns a list of divergence records with details.
    """
    # Group by (module, difficulty, version, system_type)
    configs = defaultdict(dict)
    for (agent, module, diff, ver, sys), trial_list in trials.items():
        config_key = (module, diff, ver, sys)
        configs[config_key][agent] = trial_list

    divergences = []
    for config_key, agent_trials in configs.items():
        module, diff, ver, sys = config_key
        if len(agent_trials) < 2:
            continue

        # Compute per-agent accuracy
        agent_sa = {}
        for agent, tlist in agent_trials.items():
            accs = [_get_accuracy(t) for t in tlist]
            agent_sa[agent] = statistics.mean(accs) if accs else 0.0

        # Find pairs where one succeeds and the other fails
        agents_list = list(agent_sa.keys())
        for i in range(len(agents_list)):
            for j in range(i + 1, len(agents_list)):
                a1, a2 = agents_list[i], agents_list[j]
                sa1, sa2 = agent_sa[a1], agent_sa[a2]
                if abs(sa1 - sa2) > 0.01:  # they disagree
                    winner = a1 if sa1 > sa2 else a2
                    loser = a2 if sa1 > sa2 else a1
                    divergences.append({
                        "module": module,
                        "difficulty": diff,
                        "version": ver,
                        "system_type": sys,
                        "winner": winner,
                        "loser": loser,
                        "winner_sa": max(sa1, sa2),
                        "loser_sa": min(sa1, sa2),
                        "winner_trials": agent_trials[winner],
                        "loser_trials": agent_trials[loser],
                    })

    # Sort by magnitude of divergence
    divergences.sort(key=lambda d: d["winner_sa"] - d["loser_sa"], reverse=True)
    return divergences


def print_divergence_report(divergences: List[Dict]):
    print("\n\n" + "=" * 120)
    print("DIVERGENCE REPORT: Cases Where Agents Disagree")
    print("=" * 120)

    if not divergences:
        print("  No divergences found (all agents agree on all configurations).")
        return

    # Summary: count wins per agent pair
    pair_wins = defaultdict(lambda: defaultdict(int))
    for d in divergences:
        pair_wins[d["winner"]][d["loser"]] += 1

    print("\n  Win counts (row beats column):")
    header = f"  {'':>12}"
    for a in AGENTS:
        header += f" {AGENT_SHORT[a]:>10}"
    print(header)
    for a1 in AGENTS:
        row = f"  {AGENT_SHORT[a1]:>12}"
        for a2 in AGENTS:
            if a1 == a2:
                row += f" {'—':>10}"
            else:
                row += f" {pair_wins[a1][a2]:>10}"
        print(row)

    # Top divergences table
    print(f"\n  Top divergences (biggest SA% gaps):")
    print(f"  {'Module':<12} {'Sys':<8} {'Diff':<7} {'Ver':<4} │ {'Winner':<10} {'SA%':>5} {'Loser':<10} {'SA%':>5} │ {'Gap':>5}")
    print(f"  {'':─<12} {'':─<8} {'':─<7} {'':─<4} ┼ {'':─<10} {'':─>5} {'':─<10} {'':─>5} ┼ {'':─>5}")

    for d in divergences[:50]:
        gap = (d["winner_sa"] - d["loser_sa"]) * 100
        print(
            f"  {MODULE_SHORT.get(d['module'], d['module']):<12} "
            f"{SYS_SHORT.get(d['system_type'], d['system_type']):<8} "
            f"{d['difficulty']:<7} {d['version']:<4} │ "
            f"{AGENT_SHORT[d['winner']]:<10} {d['winner_sa']*100:>5.1f} "
            f"{AGENT_SHORT[d['loser']]:<10} {d['loser_sa']*100:>5.1f} │ "
            f"{gap:>+5.1f}"
        )


# ---------------------------------------------------------------------------
# Divergent trial dumping
# ---------------------------------------------------------------------------

def dump_divergent_examples(divergences: List[Dict], output_dir: str, max_examples: int = 20):
    """
    Write detailed comparison files for the top divergent cases.
    Each file contains both agents' submitted laws and optionally chat logs.
    """
    os.makedirs(output_dir, exist_ok=True)

    manifest = []
    for idx, d in enumerate(divergences[:max_examples]):
        fname = (
            f"{idx:02d}_{d['module']}_{d['system_type']}_"
            f"{d['difficulty']}_{d['version']}_"
            f"{AGENT_SHORT[d['winner']]}vs{AGENT_SHORT[d['loser']]}.txt"
        )
        fpath = os.path.join(output_dir, fname)

        with open(fpath, "w") as f:
            f.write(f"DIVERGENCE CASE #{idx + 1}\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Module:      {d['module']}\n")
            f.write(f"System:      {d['system_type']}\n")
            f.write(f"Difficulty:  {d['difficulty']}\n")
            f.write(f"Version:     {d['version']}\n")
            f.write(f"Winner:      {d['winner']} (SA={d['winner_sa']*100:.1f}%)\n")
            f.write(f"Loser:       {d['loser']} (SA={d['loser_sa']*100:.1f}%)\n")
            f.write(f"\n{'=' * 80}\n")

            # Winner trials
            f.write(f"\n--- {AGENT_SHORT[d['winner']]} TRIALS ---\n")
            for t in d["winner_trials"]:
                tid = t.get("trial_id", "?")
                acc = _get_accuracy(t)
                rmsle = _get_rmsle(t)
                law = _get_law(t)
                f.write(f"\n  Trial {tid}: accuracy={acc}, rmsle={rmsle:.6f}\n")
                f.write(f"  Submitted law:\n    {law}\n")

            # Loser trials
            f.write(f"\n--- {AGENT_SHORT[d['loser']]} TRIALS ---\n")
            for t in d["loser_trials"]:
                tid = t.get("trial_id", "?")
                acc = _get_accuracy(t)
                rmsle = _get_rmsle(t)
                law = _get_law(t)
                f.write(f"\n  Trial {tid}: accuracy={acc}, rmsle={rmsle:.6f}\n")
                f.write(f"  Submitted law:\n    {law}\n")

            # Append chat logs if available
            f.write(f"\n{'=' * 80}\n")
            f.write(f"CHAT LOGS\n")
            f.write(f"{'=' * 80}\n")

            # Loser chat log (more interesting for failure analysis)
            for t in d["loser_trials"]:
                chat_path = t.get("_chat_log", "")
                if chat_path and os.path.exists(chat_path):
                    f.write(f"\n--- {AGENT_SHORT[d['loser']]} Trial {t.get('trial_id', '?')} Chat ---\n")
                    with open(chat_path) as cf:
                        content = cf.read()
                        # Truncate very long logs
                        if len(content) > 20000:
                            f.write(content[:10000])
                            f.write(f"\n\n... [TRUNCATED {len(content) - 20000} chars] ...\n\n")
                            f.write(content[-10000:])
                        else:
                            f.write(content)
                    break  # just first trial's chat

        manifest.append({
            "file": fname,
            "module": d["module"],
            "system": d["system_type"],
            "difficulty": d["difficulty"],
            "version": d["version"],
            "winner": AGENT_SHORT[d["winner"]],
            "loser": AGENT_SHORT[d["loser"]],
            "gap": f"{(d['winner_sa'] - d['loser_sa']) * 100:+.1f}%",
        })

    # Write manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Wrote {len(manifest)} divergence case files to {output_dir}/")
    print(f"  Manifest: {manifest_path}")


# ---------------------------------------------------------------------------
# Planned agent improvement summary
# ---------------------------------------------------------------------------

def print_planned_vs_others(trials: Dict, modules: List[str]):
    """
    Focused summary: where does planned_agent improve over the other two?
    Groups by system_type to show the decomposition effect.
    """
    print("\n\n" + "=" * 120)
    print("PLANNED AGENT IMPROVEMENT SUMMARY")
    print("=" * 120)

    agg = defaultdict(list)
    for (agent, module, diff, ver, sys), trial_list in trials.items():
        for t in trial_list:
            agg[(agent, sys, diff)].append(_get_accuracy(t))

    print(f"\n  {'System':<8} {'Diff':<7} │ {'Vanilla':>8} {'CodeAst':>8} {'Planned':>8} │ {'P-V':>6} {'P-C':>6}")
    print(f"  {'':─<8} {'':─<7} ┼ {'':─>8} {'':─>8} {'':─>8} ┼ {'':─>6} {'':─>6}")

    for sys_type in SYSTEM_TYPES:
        for diff in DIFFICULTIES:
            v_vals = agg.get(("vanilla_agent", sys_type, diff), [])
            c_vals = agg.get(("code_assisted_agent", sys_type, diff), [])
            p_vals = agg.get(("planned_agent", sys_type, diff), [])

            v_sa = statistics.mean(v_vals) * 100 if v_vals else None
            c_sa = statistics.mean(c_vals) * 100 if c_vals else None
            p_sa = statistics.mean(p_vals) * 100 if p_vals else None

            v_str = f"{v_sa:>8.1f}" if v_sa is not None else f"{'—':>8}"
            c_str = f"{c_sa:>8.1f}" if c_sa is not None else f"{'—':>8}"
            p_str = f"{p_sa:>8.1f}" if p_sa is not None else f"{'—':>8}"

            pv = f"{p_sa - v_sa:>+6.1f}" if (p_sa is not None and v_sa is not None) else f"{'—':>6}"
            pc = f"{p_sa - c_sa:>+6.1f}" if (p_sa is not None and c_sa is not None) else f"{'—':>6}"

            print(f"  {SYS_SHORT[sys_type]:<8} {diff:<7} │ {v_str} {c_str} {p_str} │ {pv} {pc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare NewtonBench agents")
    parser.add_argument("--base-dir", default="evaluation_results")
    parser.add_argument("--model", default="qwq-32b")
    parser.add_argument("--module", default=None,
                        help="Restrict to one module (e.g. m0_gravity)")
    parser.add_argument("--dump-divergent", action="store_true",
                        help="Write detailed comparison files for divergent cases")
    parser.add_argument("--dump-dir", default="analysis/divergent_cases",
                        help="Output dir for divergent case files")
    parser.add_argument("--max-examples", type=int, default=20,
                        help="Max divergent examples to dump")
    args = parser.parse_args()

    print(f"Agent Comparison: {args.model}")
    print(f"  Base dir: {args.base_dir}")

    # Determine which modules to analyze
    modules = [args.module] if args.module else ALL_MODULES

    # Load all trial-level data
    trials = load_all_trials(args.base_dir, args.model)
    n_trials = sum(len(v) for v in trials.values())
    agents_found = set(k[0] for k in trials.keys())
    modules_found = sorted(set(k[1] for k in trials.keys()))

    print(f"  Loaded {n_trials} trials across {len(agents_found)} agents, "
          f"{len(modules_found)} modules")
    print(f"  Agents: {[AGENT_SHORT.get(a, a) for a in sorted(agents_found)]}")
    print(f"  Modules: {modules_found}")

    if args.module:
        # Filter to requested module
        trials = {k: v for k, v in trials.items() if k[1] == args.module}
        modules_found = [m for m in modules_found if m == args.module]

    # 1. Summary table
    print_summary_table(trials, modules_found)

    # 2. Planned agent improvement summary
    print_planned_vs_others(trials, modules_found)

    # 3. Divergence analysis
    divergences = find_divergent_trials(trials)
    print_divergence_report(divergences)

    # 4. Optionally dump detailed examples
    if args.dump_divergent:
        dump_dir = os.path.join(args.dump_dir, args.model)
        dump_divergent_examples(divergences, dump_dir, args.max_examples)


if __name__ == "__main__":
    main()