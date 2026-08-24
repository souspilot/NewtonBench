#!/usr/bin/env python3
"""
Test the ExperimentPlanner against all NewtonBench modules.

No LLM required — validates prompt parsing, variable extraction,
experiment generation, and execution for all 12 × 3 configurations.

Usage (from the NewtonBench root):
    python test_planner.py
"""

import importlib
import traceback

from planned_agent.planner import ExperimentPlanner, sanitize_experiment

MODULES = [
    "m0_gravity", "m1_coulomb_force", "m2_magnetic_force", "m3_fourier_law",
    "m4_snell_law", "m5_radioactive_decay", "m6_underdamped_harmonic",
    "m7_malus_law", "m8_sound_speed", "m9_hooke_law",
    "m10_be_distribution", "m11_heat_transfer",
]

SYSTEMS = ["vanilla_equation", "simple_system", "complex_system"]


def test_module(module_name: str):
    print(f"\n{'=' * 60}")
    print(f"  {module_name}")
    print(f"{'=' * 60}")

    try:
        mod = importlib.import_module(f"modules.{module_name}")
    except Exception as e:
        print(f"  SKIP: could not import ({e})")
        return

    planner = ExperimentPlanner(points_per_sweep=5, noise_check_reps=2)
    versions = (
        mod.get_available_law_versions("easy")
        if hasattr(mod, "get_available_law_versions") else [None]
    )
    law_version = versions[0] if versions else None

    for system in SYSTEMS:
        print(f"\n  --- {system} ---")
        try:
            prompt = mod.get_task_prompt(system, noise_level=0.0)
        except Exception as e:
            print(f"    SKIP get_task_prompt: {e}")
            continue

        plan = planner.plan(prompt, system_type_override=system)
        print(f"    Variables:    {[v.name for v in plan.variables]}")
        print(f"    Noise check: {len(plan.noise_check)} experiments")
        print(f"    Sweeps:      {len(plan.sweeps)} experiments")
        print(f"    Total:       {plan.total_experiments}")
        hint_preview = plan.preprocessing_hint[:80].replace("\n", " ")
        print(f"    Hint:        {hint_preview}...")

        if not plan.sweeps and not plan.noise_check:
            print("    WARN: empty plan!")
            continue

        test_exps = [
            sanitize_experiment(e)
            for e in (plan.noise_check + plan.sweeps)[:3]
        ]
        ok, fail = 0, 0
        for exp in test_exps:
            try:
                mod.run_experiment_for_module(
                    **exp, noise_level=0.0, difficulty="easy",
                    system=system, law_version=law_version,
                )
                ok += 1
            except Exception as e:
                fail += 1
                print(f"    FAIL experiment {exp}: {e}")

        print(f"    Execution:   {ok}/{ok + fail} succeeded")


def main():
    print("ExperimentPlanner Test Suite")
    print("Testing prompt parsing + experiment execution for all modules\n")

    for name in MODULES:
        try:
            test_module(name)
        except Exception:
            print(f"  ERROR in {name}:")
            traceback.print_exc()

    print("\n\nDone.")


if __name__ == "__main__":
    main()