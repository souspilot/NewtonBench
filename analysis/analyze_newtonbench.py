#!/usr/bin/env python3
"""
NewtonBench Results Analysis Script
====================================
Compares local QwQ-32B evaluation results against the paper's Table 2
(arXiv:2510.07172v3, ICLR 2026).

Directory structure expected:
  evaluation_results/{model}/{module}/{agent_type}/{difficulty}/{version}/{system_type}/
    aggregated_results.json

Paper's Table 2 reports Symbolic Accuracy (%) aggregated over 12 domains × 3 versions,
with mean ± std from 4 trials per configuration.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
import statistics

# ──────────────────────────────────────────────────────────────────────────────
# Paper-reported QwQ-32B results from Table 2 (arXiv:2510.07172v3)
# Format: (system_type, difficulty) → SA%
# ──────────────────────────────────────────────────────────────────────────────

PAPER_RESULTS = {
    "vanilla_agent": {
        ("vanilla_equation", "easy"):   74.3,
        ("vanilla_equation", "medium"): 55.6,
        ("vanilla_equation", "hard"):   20.8,
        ("simple_system",    "easy"):   47.2,
        ("simple_system",    "medium"): 25.7,
        ("simple_system",    "hard"):    0.7,
        ("complex_system",   "easy"):   20.8,
        ("complex_system",   "medium"): 11.1,
        ("complex_system",   "hard"):    0.0,
        # Aggregate metrics
        "_avg_sa": 28.5,
        "_avg_rmsle": 2.4270,
        "_avg_tokens_k": 14.49,
    },
    "code_assisted_agent": {
        ("vanilla_equation", "easy"):   71.5,
        ("vanilla_equation", "medium"): 59.0,
        ("vanilla_equation", "hard"):   25.7,
        ("simple_system",    "easy"):   52.8,
        ("simple_system",    "medium"): 32.6,
        ("simple_system",    "hard"):    2.1,
        ("complex_system",   "easy"):   27.1,
        ("complex_system",   "medium"): 11.1,
        ("complex_system",   "hard"):    0.7,
        "_avg_sa": 31.4,
        "_avg_rmsle": 2.0599,
        "_avg_tokens_k": 14.62,
    }
}

# Paper standard deviations (from Table 2) for reference
PAPER_STD = {
    "vanilla_agent": {
        ("vanilla_equation", "easy"):   5.258,
        ("vanilla_equation", "medium"): 9.886,
        ("vanilla_equation", "hard"):   3.586,
        ("simple_system",    "easy"):   7.522,
        ("simple_system",    "medium"): 8.295,
        ("simple_system",    "hard"):   1.389,
        ("complex_system",   "easy"):   5.782,
        ("complex_system",   "medium"): 5.072,
        ("complex_system",   "hard"):   0.000,
    },
    "code_assisted_agent": {
        ("vanilla_equation", "easy"):   5.727,
        ("vanilla_equation", "medium"): 4.744,
        ("vanilla_equation", "hard"):   2.660,
        ("simple_system",    "easy"):   7.172,
        ("simple_system",    "medium"): 5.727,
        ("simple_system",    "hard"):   1.389,
        ("complex_system",   "easy"):   2.660,
        ("complex_system",   "medium"): 3.928,
        ("complex_system",   "hard"):   1.389,
    }
}

ALL_MODULES = [
    "m0_gravity", "m1_coulomb_force", "m2_magnetic_force", "m3_fourier_law",
    "m4_snell_law", "m5_radioactive_decay", "m6_underdamped_harmonic",
    "m7_malus_law", "m8_sound_speed", "m9_hooke_law",
    "m10_be_distribution", "m11_heat_transfer"
]

MODULE_NAMES = {
    "m0_gravity": "Gravity",
    "m1_coulomb_force": "Coulomb",
    "m2_magnetic_force": "Magnetic",
    "m3_fourier_law": "Fourier",
    "m4_snell_law": "Snell",
    "m5_radioactive_decay": "Radioactive",
    "m6_underdamped_harmonic": "Harmonic",
    "m7_malus_law": "Malus",
    "m8_sound_speed": "Sound",
    "m9_hooke_law": "Hooke",
    "m10_be_distribution": "Bose-Einstein",
    "m11_heat_transfer": "Heat Transfer",
}

DIFFICULTIES = ["easy", "medium", "hard"]
VERSIONS = ["v0", "v1", "v2"]
SYSTEM_TYPES = ["vanilla_equation", "simple_system", "complex_system"]
AGENT_TYPES = ["vanilla_agent", "code_assisted_agent"]


def load_results(base_dir: str, model: str = "qwq-32b"):
    """
    Walk the results directory and load all aggregated_results.json files.
    Returns a list of dicts with parsed metadata and metrics.
    """
    results = []
    model_dir = Path(base_dir) / model

    if not model_dir.exists():
        print(f"ERROR: Directory not found: {model_dir}")
        sys.exit(1)

    for module in sorted(model_dir.iterdir()):
        if not module.is_dir():
            continue
        for agent_type in module.iterdir():
            if not agent_type.is_dir():
                continue
            for difficulty in agent_type.iterdir():
                if not difficulty.is_dir():
                    continue
                for version in difficulty.iterdir():
                    if not version.is_dir():
                        continue
                    for system_type in version.iterdir():
                        if not system_type.is_dir():
                            continue
                        agg_file = system_type / "aggregated_results.json"
                        if agg_file.exists():
                            with open(agg_file) as f:
                                data = json.load(f)
                            results.append({
                                "module": module.name,
                                "agent_type": agent_type.name,
                                "difficulty": difficulty.name,
                                "version": version.name,
                                "system_type": system_type.name.split("_noise")[0],  # strip noise suffix
                                "noise_level": data["config"].get("noise_level", 0.0),
                                "sa": data["aggregate"]["all_trials"]["average_exact_accuracy"],
                                "rmsle": data["aggregate"]["all_trials"]["average_rmsle"],
                                "rounds": data["aggregate"]["all_trials"]["average_rounds"],
                                "experiments": data["aggregate"]["all_trials"]["average_experiments"],
                                "tokens": data["aggregate"]["all_trials"]["average_total_tokens"],
                                "num_trials": data["aggregate"]["all_trials"]["num_total_trials"],
                                "retries": data["aggregate"]["retry_statistics"]["total_retry_attempts"],
                                "raw_system_type_dir": system_type.name,
                            })
    return results


def compute_paper_style_table(results, noise_level=0.0):
    """
    Aggregate results the same way the paper does:
    - Group by (agent_type, system_type, difficulty)
    - Each group has up to 12 modules × 3 versions = 36 data points
    - Each data point's SA is the average_exact_accuracy (avg of 4 trials)
    - Report mean and std across these data points, converted to %

    The paper does 4 independent runs and reports mean±std of the 4 run-level
    aggregates. Since we have 4 trials within a single run, we compute:
    - Per-run SA = mean of all (module × version) SA values in that run
    - The paper repeats the entire experiment 4 times; we have 1 run with 4 trials.

    For comparison, we'll report the single-run aggregate (matching trial structure).
    """
    # Filter to noise=0 only
    filtered = [r for r in results if abs(r["noise_level"] - noise_level) < 1e-6]

    # Group by (agent_type, system_type, difficulty)
    groups = defaultdict(list)
    for r in filtered:
        key = (r["agent_type"], r["system_type"], r["difficulty"])
        groups[key].append(r)

    return groups


def print_comparison(groups):
    """Print a comparison table: local results vs paper."""
    print("\n" + "=" * 100)
    print("COMPARISON: Local Results vs Paper (QwQ-32B, noise=0.0)")
    print("=" * 100)

    for agent_type in AGENT_TYPES:
        agent_label = "Vanilla Agent" if agent_type == "vanilla_agent" else "Code-Assisted Agent"
        print(f"\n{'─' * 100}")
        print(f"  {agent_label}")
        print(f"{'─' * 100}")
        print(f"  {'System Type':<20} {'Diff':<8} │ {'Local SA%':>10} {'Paper SA%':>10} {'Δ':>8} │ "
              f"{'Local RMSLE':>12} {'#Modules':>10} {'#Versions':>10}")
        print(f"  {'':─<20} {'':─<8} ┼ {'':─>10} {'':─>10} {'':─>8} ┼ {'':─>12} {'':─>10} {'':─>10}")

        for sys_type in SYSTEM_TYPES:
            for diff in DIFFICULTIES:
                key = (agent_type, sys_type, diff)
                data_points = groups.get(key, [])

                if not data_points:
                    print(f"  {sys_type:<20} {diff:<8} │ {'N/A':>10} ", end="")
                    paper_val = PAPER_RESULTS.get(agent_type, {}).get((sys_type, diff), None)
                    if paper_val is not None:
                        print(f"{paper_val:>10.1f} {'N/A':>8} │ {'N/A':>12} {'0':>10} {'0':>10}")
                    else:
                        print(f"{'N/A':>10} {'N/A':>8} │ {'N/A':>12} {'0':>10} {'0':>10}")
                    continue

                # Compute mean SA% across all data points (modules × versions)
                sa_values = [r["sa"] * 100 for r in data_points]
                rmsle_values = [r["rmsle"] for r in data_points]
                local_sa = statistics.mean(sa_values)
                local_rmsle = statistics.mean(rmsle_values)
                n_modules = len(set(r["module"] for r in data_points))
                n_versions = len(set(r["version"] for r in data_points))

                paper_val = PAPER_RESULTS.get(agent_type, {}).get((sys_type, diff), None)
                delta = ""
                if paper_val is not None:
                    delta = f"{local_sa - paper_val:+.1f}"

                print(f"  {sys_type:<20} {diff:<8} │ {local_sa:>10.1f} "
                      f"{paper_val if paper_val is not None else 'N/A':>10} "
                      f"{delta:>8} │ {local_rmsle:>12.4f} "
                      f"{n_modules:>10} {n_versions:>10}")


def print_per_module_breakdown(groups):
    """Print per-module breakdown to see which domains are contributing."""
    print("\n\n" + "=" * 100)
    print("PER-MODULE BREAKDOWN (Symbolic Accuracy %)")
    print("=" * 100)

    for agent_type in AGENT_TYPES:
        agent_label = "Vanilla Agent" if agent_type == "vanilla_agent" else "Code-Assisted Agent"
        print(f"\n{'─' * 100}")
        print(f"  {agent_label}")
        print(f"{'─' * 100}")

        # Collect all data for this agent
        all_data = defaultdict(lambda: defaultdict(list))
        for (at, st, diff), data_points in groups.items():
            if at != agent_type:
                continue
            for r in data_points:
                all_data[r["module"]][(st, diff)].append(r["sa"] * 100)

        if not all_data:
            print("  No data available.")
            continue

        # Header
        header = f"  {'Module':<18}"
        for sys_type in SYSTEM_TYPES:
            short_name = {"vanilla_equation": "VanEq",
                         "simple_system": "SimpS",
                         "complex_system": "CompS"}[sys_type]
            for diff in DIFFICULTIES:
                header += f" {short_name}_{diff[:1].upper():>7}"
        header += f" {'Avg':>8}"
        print(header)
        print(f"  {'':─<18}" + "─" * (len(header) - 20))

        module_avgs = {}
        for module in ALL_MODULES:
            if module not in all_data:
                continue
            row = f"  {MODULE_NAMES.get(module, module):<18}"
            all_vals = []
            for sys_type in SYSTEM_TYPES:
                for diff in DIFFICULTIES:
                    vals = all_data[module].get((sys_type, diff), [])
                    if vals:
                        mean_val = statistics.mean(vals)
                        row += f" {mean_val:>7.1f}"
                        all_vals.append(mean_val)
                    else:
                        row += f" {'—':>7}"
            if all_vals:
                avg = statistics.mean(all_vals)
                row += f" {avg:>8.1f}"
                module_avgs[module] = avg
            print(row)

        if module_avgs:
            overall = statistics.mean(module_avgs.values())
            print(f"  {'OVERALL':<18}" + " " * (len(header) - 28) + f" {overall:>8.1f}")


def print_trial_level_analysis(results, noise_level=0.0):
    """Analyze trial-level consistency (since the paper reports ±std from 4 runs)."""
    print("\n\n" + "=" * 100)
    print("TRIAL-LEVEL STATISTICS")
    print("=" * 100)

    filtered = [r for r in results if abs(r["noise_level"] - noise_level) < 1e-6]

    for agent_type in AGENT_TYPES:
        agent_label = "Vanilla Agent" if agent_type == "vanilla_agent" else "Code-Assisted Agent"
        print(f"\n  {agent_label}:")
        agent_data = [r for r in filtered if r["agent_type"] == agent_type]
        if not agent_data:
            print("    No data.")
            continue

        total_configs = len(agent_data)
        total_trials = sum(r["num_trials"] for r in agent_data)
        total_retries = sum(r["retries"] for r in agent_data)
        avg_sa = statistics.mean([r["sa"] for r in agent_data]) * 100
        avg_rmsle = statistics.mean([r["rmsle"] for r in agent_data])
        avg_tokens = statistics.mean([r["tokens"] for r in agent_data])
        avg_rounds = statistics.mean([r["rounds"] for r in agent_data])
        avg_experiments = statistics.mean([r["experiments"] for r in agent_data])

        print(f"    Configurations loaded: {total_configs}")
        print(f"    Total trials:          {total_trials}")
        print(f"    Total retries:         {total_retries}")
        print(f"    Mean SA%:              {avg_sa:.1f}")
        print(f"    Mean RMSLE:            {avg_rmsle:.4f}")
        print(f"    Mean tokens:           {avg_tokens:.0f} ({avg_tokens/1000:.2f}k)")
        print(f"    Mean rounds:           {avg_rounds:.1f}")
        print(f"    Mean experiments:      {avg_experiments:.1f}")


def print_coverage_report(results, noise_level=0.0):
    """Show which modules/configs are present vs missing."""
    print("\n\n" + "=" * 100)
    print("COVERAGE REPORT")
    print("=" * 100)

    filtered = [r for r in results if abs(r["noise_level"] - noise_level) < 1e-6]

    for agent_type in AGENT_TYPES:
        agent_label = "Vanilla Agent" if agent_type == "vanilla_agent" else "Code-Assisted Agent"
        print(f"\n  {agent_label}:")

        agent_data = [r for r in filtered if r["agent_type"] == agent_type]
        present_modules = set(r["module"] for r in agent_data)
        missing_modules = set(ALL_MODULES) - present_modules

        # Expected configs per module: 3 difficulties × 3 versions × 3 system_types = 27
        expected_per_module = len(DIFFICULTIES) * len(VERSIONS) * len(SYSTEM_TYPES)
        total_expected = expected_per_module * len(ALL_MODULES)

        print(f"    Modules present:  {sorted(present_modules)}")
        print(f"    Modules missing:  {sorted(missing_modules)}")
        print(f"    Configs found:    {len(agent_data)} / {total_expected} "
              f"({len(agent_data)/total_expected*100:.0f}%)")

        # Per-module detail
        for module in ALL_MODULES:
            mod_data = [r for r in agent_data if r["module"] == module]
            if mod_data:
                configs = set((r["difficulty"], r["version"], r["system_type"]) for r in mod_data)
                print(f"      {module:<30} {len(mod_data):>3}/{expected_per_module} configs")
            else:
                print(f"      {module:<30}   0/{expected_per_module} configs  ← MISSING")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze NewtonBench results vs paper")
    parser.add_argument("--base-dir", default="evaluation_results",
                        help="Base directory containing model results")
    parser.add_argument("--model", default="qwq-32b",
                        help="Model name (directory name)")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Noise level to analyze (default: 0.0)")
    args = parser.parse_args()

    print("NewtonBench Results Analysis")
    print(f"  Base dir: {args.base_dir}")
    print(f"  Model:    {args.model}")
    print(f"  Noise:    {args.noise}")

    # Load all results
    results = load_results(args.base_dir, args.model)
    print(f"\n  Loaded {len(results)} result configurations total.")

    # Coverage report
    print_coverage_report(results, args.noise)

    # Compute paper-style aggregation
    groups = compute_paper_style_table(results, args.noise)

    # Print comparison
    print_comparison(groups)

    # Per-module breakdown
    print_per_module_breakdown(groups)

    # Trial-level stats
    print_trial_level_analysis(results, args.noise)

    print("\n\n" + "=" * 100)
    print("NOTES")
    print("=" * 100)
    print("""
  1. The paper aggregates over ALL 12 modules × 3 versions = 36 data points per cell.
     If you only have a subset of modules, your numbers may differ from the paper
     simply due to domain-specific difficulty (e.g., Bose-Einstein is hardest at ~18% avg).

  2. The paper reports mean ± std from 4 INDEPENDENT RUNS of the full benchmark.
     Your results are from a single run with 4 trials. The trial-level variance
     within a single run may differ from inter-run variance.

  3. The paper uses specific API-based models. If you're running local inference
     with a different quantization or configuration, results may diverge.

  4. The LLM judge for symbolic accuracy matters. The paper uses their own judge;
     differences in judge implementation can affect SA scores.

  5. Small differences (±5%) are expected due to stochasticity. Larger systematic
     deviations may indicate configuration differences.
""")


if __name__ == "__main__":
    main()