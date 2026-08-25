import os
import subprocess
import argparse
import importlib
import json
import glob
import time
from typing import Tuple, List, Dict, Optional
from collections import defaultdict

def get_module_folders():
    """Scan the 'modules' directory for all module folders (e.g., m0_gravity)."""
    module_dir = 'modules'
    if not os.path.isdir(module_dir):
        print(f"Error: Directory '{module_dir}' not found.")
        return []
    
    module_folders = [d for d in os.listdir(module_dir) if os.path.isdir(os.path.join(module_dir, d)) and d.startswith('m')]
    return sorted(module_folders)

def get_law_versions_for_difficulty(module_name, difficulty):
    """Dynamically import a module and get the number of law versions for a given difficulty."""
    try:
        module = importlib.import_module(f"modules.{module_name}.laws")
        if hasattr(module, 'get_available_law_versions'):
            return module.get_available_law_versions(difficulty)
        else:
            print(f"Warning: Module {module_name} does not have 'get_available_law_versions'. Assuming 1 law version.")
            return [None] # Assume one version if function not found
    except (ImportError, ValueError) as e:
        print(f"Could not get law versions for {module_name} ({difficulty}): {e}")
        return []

# Default location of the representative subset (see configs/representative_subset.json,
# built by configs/generate_subset.py).
#
# Each cell of the paper's Table 2 / Appendix B.1 is a mean over 12 runs (3 law
# versions x 4 trials) for one (module, difficulty, system) combination -- so a
# cell's reported value is always a multiple of 1/12 (e.g. 91.7% = 11/12). Running
# fewer than 3 versions for a cell can never reproduce that resolution, so this
# subset does NOT thin versions or trials within a cell it runs. Instead it thins
# which CELLS get run: for each of the 12 modules, only 2 of its 9 (difficulty,
# system) cells are run, but those 2 are run at FULL fidelity (all 3 versions x 4
# trials), making their numbers directly comparable to the corresponding entries
# in Appendix B.1. trials_per_law is left at the paper's default (4) regardless of
# subsetting -- deliberately NOT reduced, so each configuration keeps full retry
# slack in case a trial fails.
#
# 12 modules x 2 cells x 3 versions = 72 configurations, x 4 trials/law = 288
# total trials per (model, agent_backend), vs. 324 configs / 1,296 trials full.
DEFAULT_SUBSET_FILE = os.path.join('configs', 'representative_subset.json')
DEFAULT_TRIALS = 4

# A subset file is {module: [{"difficulty": d, "system": s}, ...]} -- the explicit
# whitelist of (difficulty, system) cells to run for that module, each at full
# law-version fidelity. SubsetCells is that whitelist, as a Dict[module] -> Set[(difficulty, system)].
SubsetCells = Dict[str, set]


def load_subset_cells(subset_file: str) -> SubsetCells:
    """Load a {module: [{"difficulty": d, "system": s}, ...]} cell whitelist from JSON.

    Returns an empty dict (meaning "no restriction") if the file doesn't exist so
    that this feature degrades gracefully rather than hard failing.
    """
    if not subset_file or not os.path.exists(subset_file):
        return {}
    with open(subset_file, 'r') as f:
        raw = json.load(f)
    return {
        module_name: {(cell["difficulty"], cell["system"]) for cell in cells}
        for module_name, cells in raw.items()
    }

def get_experiment_path(model_name: str, module: str, agent_backend: str, difficulty: str, 
                       law_version: str, system: str, noise_level: float) -> str:
    """Generate standardized experiment directory path."""
    noise_str = str(noise_level).replace('.', '_')
    law_version_str = law_version if law_version is not None else "random"
         
    # Standard behavior: Find the latest version number for this configuration
    base_pattern = os.path.join(
        "evaluation_results", model_name, module, agent_backend, difficulty, law_version_str, 
        f"{system}_noise{noise_str}_v*"
    )
    
    existing_dirs = glob.glob(base_pattern)
    if existing_dirs:
        # Find the highest version number
        version_nums = []
        for path in existing_dirs:
            try:
                version_part = path.split('_v')[-1]
                version_nums.append(int(version_part))
            except (ValueError, IndexError):
                continue
        latest_version = max(version_nums) if version_nums else 0
        return os.path.join(
            "evaluation_results", model_name, module, agent_backend, difficulty, law_version_str,
            f"{system}_noise{noise_str}_v{latest_version}"
        )
    else:
        # No existing directory, will be created as v1
        return os.path.join(
            "evaluation_results", model_name, module, agent_backend, difficulty, law_version_str,
            f"{system}_noise{noise_str}_v1"
        )

def check_experiment_completion(experiment_path: str, expected_trials: int = 4, model_name: str = None, agent_backend: str = None) -> Tuple[bool, int, int]:
    """Check if an experiment configuration is complete.
    
    Args:
        experiment_path: Path to experiment directory
        expected_trials: Expected number of trials (default: 4)
        model_name: Model name for special handling (e.g., gpt5mini)
        agent_backend: Agent backend type for special handling (e.g., code_assisted_agent)
    
    Returns:
        tuple: (is_complete: bool, completed_trials: int, total_expected: int)
    """
    if not os.path.exists(experiment_path):
        return False, 0, expected_trials

    # Check for aggregated results
    aggregated_path = os.path.join(experiment_path, "aggregated_results.json")
    trials_dir = os.path.join(experiment_path, "trials")
    
    if not os.path.exists(aggregated_path) or not os.path.exists(trials_dir):
        return False, 0, expected_trials
    
    # Read expected trials from aggregated results
    try:
        with open(aggregated_path, 'r') as f:
            config = json.load(f)
            expected_from_config = config.get('config', {}).get('trials', expected_trials)
    except (json.JSONDecodeError, FileNotFoundError):
        expected_from_config = expected_trials
    
    # Count actual trial files
    trial_json_files = glob.glob(os.path.join(trials_dir, "trial*.json"))
    # Filter out fail files
    valid_trial_files = [f for f in trial_json_files if not f.endswith('_fail.json')]
    completed_trials = len(valid_trial_files)
    
    # Validate trial files are not corrupted
    valid_trials = 0
    for trial_file in valid_trial_files:
        try:
            with open(trial_file, 'r') as f:
                trial_data = json.load(f)
                # Check if trial has essential fields
                if 'trial_id' in trial_data and 'evaluation' in trial_data:
                    valid_trials += 1
        except (json.JSONDecodeError, FileNotFoundError):
            continue
    
    is_complete = valid_trials >= expected_from_config
    return is_complete, valid_trials, expected_from_config

def cell_allowed(subset_cells: Dict[str, set], module_name: str, difficulty: str, system: str,
                  restrict_cells: bool) -> bool:
    """True if (module, difficulty, system) should run, given the subset whitelist."""
    if not restrict_cells:
        return True
    allowed = subset_cells.get(module_name)
    if not allowed:
        return False  # module has no whitelisted cells -> skip entirely
    return (difficulty, system) in allowed


def count_total_configurations(modules: List[str], difficulties: List[str], systems: List[str], 
                             law_versions_map: Dict[str, Dict[str, List[str]]], 
                             noise_levels: List[float], args,
                             subset_cells: Optional[Dict[str, set]] = None,
                             restrict_cells: bool = False) -> int:
    """Calculate total number of experiment configurations."""
    total = 0
    subset_cells = subset_cells or {}
    
    # Apply filters
    filtered_modules = [args.module] if args.module != "none" else modules
    filtered_difficulties = [args.equation_difficulty] if args.equation_difficulty != "none" else difficulties
    filtered_systems = [args.model_system] if args.model_system != "none" else systems
    
    for noise_level in noise_levels:
        for module_name in filtered_modules:
            for difficulty in filtered_difficulties:
                if module_name in law_versions_map and difficulty in law_versions_map[module_name]:
                    law_versions = law_versions_map[module_name][difficulty]
                    for system in filtered_systems:
                        if not cell_allowed(subset_cells, module_name, difficulty, system, restrict_cells):
                            continue
                        total += len(law_versions)
    
    return total

def generate_progress_report(completed: int, skipped: int, partial: int, failed: int, total: int) -> str:
    """Generate progress statistics report."""
    remaining = total - completed - skipped - partial - failed
    
    report = f"\n{'='*60}\n"
    report += "EXPERIMENT PROGRESS SUMMARY\n"
    report += f"{'='*60}\n"
    report += f"✓ Completed:     {completed:4d} configurations ({completed/total*100:5.1f}%)\n"
    report += f"⏭ Skipped:       {skipped:4d} configurations ({skipped/total*100:5.1f}%)\n"
    report += f"⚠ Partial:       {partial:4d} configurations ({partial/total*100:5.1f}%)\n"
    report += f"✗ Failed:        {failed:4d} configurations ({failed/total*100:5.1f}%)\n"
    report += f"⏳ Remaining:     {remaining:4d} configurations ({remaining/total*100:5.1f}%)\n"
    report += f"📊 Total:         {total:4d} configurations\n"
    report += f"{'='*60}\n"
    
    return report

def parse_noise_levels(noise_str: str) -> List[float]:
    """Parse comma-separated noise levels string."""
    try:
        return [float(x.strip()) for x in noise_str.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid noise levels: {noise_str}. Expected comma-separated floats.")

def get_configuration_name(module: str, difficulty: str, system: str, law_version: str, noise_level: float) -> str:
    """Generate human-readable configuration name."""
    law_str = law_version if law_version is not None else "random"
    return f"{module}/{difficulty}/{system}/{law_str}/noise{noise_level}"

def main():
    parser = argparse.ArgumentParser(description="Run all evaluations for all modules with noise level iteration and resume capability.")
    parser.add_argument("--model_name", type=str, default="gpt41mini", help="Name of the LLM to use.")
    parser.add_argument("--module", type=str, default="none", help="Name of the module to test (e.g., m0_gravity). Use 'none' for all modules.")
    parser.add_argument("-n", "--noise", type=float, default=0.0, help="Noise level for experiments (e.g., 0, 0.01, 0.1).")
    parser.add_argument("-t", "--trials_per_law", type=int, default=DEFAULT_TRIALS,
                      help=f"Number of trials to run for each law version. Defaults to {DEFAULT_TRIALS} "
                           f"(the paper's default) regardless of subsetting -- kept full intentionally so "
                           f"each configuration still has enough trials to absorb the occasional API failure.")
    parser.add_argument("-d", "--equation_difficulty", type=str, default="none", choices=["easy", "medium", "hard", "none"],
                      help="Difficulty level of the equation: easy, medium, or hard.")
    parser.add_argument("-m", "--model_system", type=str, default="none", choices=["vanilla_equation", "simple_system", "complex_system", "none"],
                      help="Model system selected to test the agent: vanilla_equation, simple_system, complex_system")
    parser.add_argument("-b", "--agent_backend", type=str, default="vanilla_agent", choices=["vanilla_agent", "code_assisted_agent"],
                      help="Agent backend to use for exploration. Default is vanilla_agent. When code_assisted_agent is selected, LLM is equipped with <python> tool use.")

    # Subsetting options (for fast iteration while staying comparable to specific
    # cells of the paper's Table 2 / Appendix B.1).
    parser.add_argument("--subset_file", type=str, default=DEFAULT_SUBSET_FILE,
                      help="Path to a JSON file of {module: [{\"difficulty\": d, \"system\": s}, ...]} "
                           "whitelisting which (difficulty, system) cells to run for each module. Defaults "
                           "to configs/representative_subset.json. Whitelisted cells are always run at FULL "
                           "fidelity (all law versions x the full trials_per_law), so their numbers land in "
                           "the same 1/(versions*trials) resolution as the paper and are directly comparable "
                           "to the corresponding Appendix B.1 entries. Cells not listed for a module are "
                           "skipped entirely -- there is no partial-fidelity option, since averaging over "
                           "fewer than 3 law versions can never reproduce the paper's per-cell resolution.")
    parser.add_argument("--full", action="store_true",
                      help="Disable subsetting and run the complete 324-task benchmark (all 12 modules, all "
                           "9 difficulty x system cells, all 3 law versions), matching the original paper's "
                           "full protocol exactly.")
    parser.add_argument("--full_module", action="store_true",
                      help="Only meaningful together with --module: ignore the subset whitelist for that one "
                           "module and run its complete 3x3 difficulty x system grid, even though it's covered "
                           "by the subset file. (Without this flag, --module together with an active subset "
                           "restricts to that module's whitelisted cells -- this is what lets run_master.py "
                           "orchestrate a subset run by invoking one module at a time.)")

    # Resume and control options
    parser.add_argument("--force_rerun", action="store_true", 
                      help="Force re-run even if experiments are already complete")
    parser.add_argument("--check_only", action="store_true", 
                      help="Only check completion status, don't run experiments")
    parser.add_argument("--dry_run", action="store_true", 
                      help="Show what would be executed without running anything")
    parser.add_argument("--no_prompt", action="store_true", 
                      help="Don't prompt for confirmation before starting")
    
    args = parser.parse_args()

    modules = get_module_folders()
    if not modules:
        print("No modules found. Exiting.")
        return

    difficulties = ["easy", "medium", "hard"]
    systems = ["vanilla_equation", "simple_system", "complex_system"]
    noise_levels = [args.noise]

    # Resolve subsetting: --full disables it outright. Otherwise load the cell
    # whitelist (if present) and decide whether to restrict:
    #   - args.module == "none": restrict globally across all modules (the normal
    #     multi-module run).
    #   - args.module given AND covered by the whitelist AND --full_module not set:
    #     restrict to just that module's whitelisted cells. This is what lets
    #     run_master.py orchestrate a full subset run by invoking run_all_evaluations.py
    #     once per module with an explicit --module -- if an explicit --module always
    #     bypassed the whitelist, that orchestration would silently balloon back up
    #     to the full 324-config benchmark.
    #   - args.module given but NOT covered by the whitelist, or --full_module set:
    #     run that module's full 3x3 grid (nothing to restrict to, or explicitly asked).
    subset_cells = {} if args.full else load_subset_cells(args.subset_file)
    subset_active = bool(subset_cells)
    if not subset_active or args.full_module:
        restrict_cells = False
    elif args.module == "none":
        restrict_cells = True
    else:
        restrict_cells = args.module in subset_cells

    if args.full:
        print("Running FULL benchmark protocol (324 tasks): all 12 modules, all 9 cells, all law versions.")
    elif subset_active:
        if restrict_cells and args.module == "none":
            total_cells = sum(len(v) for v in subset_cells.values())
            print(f"Using representative subset from '{args.subset_file}': {total_cells} (module, difficulty, "
                  f"system) cells across {len(subset_cells)} modules, each run at FULL fidelity (all law "
                  f"versions x {args.trials_per_law} trials) so results are directly comparable to the "
                  f"matching Appendix B.1 entries. Pass --full to run the complete 324-task benchmark instead.")
        elif restrict_cells:
            cells = sorted(subset_cells[args.module])
            print(f"Using representative subset from '{args.subset_file}' for module '{args.module}': "
                  f"restricting to its {len(cells)} whitelisted cell(s) {cells}, run at FULL fidelity. Pass "
                  f"--full_module to run this module's complete 3x3 grid instead.")
        elif args.full_module:
            print(f"--full_module given: running the complete 3x3 grid for module '{args.module}', "
                  f"ignoring the subset whitelist.")
        else:
            print(f"Module '{args.module}' has no entries in the subset whitelist -- running its complete "
                  f"3x3 grid.")
    else:
        print(f"Subset file '{args.subset_file}' not found -- running the full set of modules and cells "
              f"(pass --full to silence this message, or run configs/generate_subset.py).")

    # Pre-compute law versions for all modules and difficulties
    print("Scanning available law versions...")
    law_versions_map = defaultdict(dict)
    for module_name in modules:
        for difficulty in difficulties:
            law_versions = get_law_versions_for_difficulty(module_name, difficulty)
            if law_versions:
                law_versions_map[module_name][difficulty] = law_versions

    # Calculate total configurations
    total_configs = count_total_configurations(modules, difficulties, systems, law_versions_map, noise_levels,
                                                 args, subset_cells=subset_cells, restrict_cells=restrict_cells)
    
    print("\n" + "="*80)
    print("EXPERIMENT CONFIGURATION SUMMARY")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Agent Backend: {args.agent_backend}")
    print(f"Noise Levels: {noise_levels}")
    print(f"Trials per Configuration: {args.trials_per_law}")
    
    if args.module == "none":
        print(f"Modules: {len(modules)} modules ({', '.join(modules[:3])}{'...' if len(modules) > 3 else ''})")
    else:
        print(f"Module: {args.module}")
    
    if args.equation_difficulty == "none":
        print(f"Equation Difficulties: {difficulties}")
    else:
        print(f"Equation Difficulty: {args.equation_difficulty}")
    
    if args.model_system == "none":
        print(f"Model Systems: {systems}")
    else:
        print(f"Model System: {args.model_system}")
    
    print(f"Total Configurations: {total_configs}")
    print(f"Total Expected Trials: {total_configs * args.trials_per_law}")
    print("="*80)

    # Pre-flight completion check
    if not args.check_only and not args.dry_run:
        print("\nPerforming pre-flight completion check...")
        
    completed_count = 0
    partial_count = 0
    missing_count = 0
    skipped_count = 0
    failed_count = 0
    
    execution_plan = []
    
    # Apply filters for the main loop
    filtered_modules = [args.module] if args.module != "none" else modules
    filtered_difficulties = [args.equation_difficulty] if args.equation_difficulty != "none" else difficulties
    filtered_systems = [args.model_system] if args.model_system != "none" else systems
    
    # Check all configurations
    for noise_level in noise_levels:
        for module_name in filtered_modules:
            for difficulty in filtered_difficulties:
                    
                if module_name not in law_versions_map or difficulty not in law_versions_map[module_name]:
                        continue   
                    
                law_versions = law_versions_map[module_name][difficulty]
                
                for system in filtered_systems:   
                    if not cell_allowed(subset_cells, module_name, difficulty, system, restrict_cells):
                        continue
                    for law_version in law_versions:
                        config_name = get_configuration_name(module_name, difficulty, system, law_version, noise_level)
                        experiment_path = get_experiment_path(args.model_name, module_name, args.agent_backend, 
                                                           difficulty, law_version, system, noise_level)
                        
                        is_complete, completed_trials, expected_trials = check_experiment_completion(
                            experiment_path, args.trials_per_law, args.model_name, args.agent_backend)
                        
                        if is_complete and not args.force_rerun:
                            completed_count += 1
                            if args.check_only or args.dry_run:
                                print(f"✓ COMPLETE: {config_name} ({completed_trials}/{expected_trials} trials)")
                        elif completed_trials > 0 and completed_trials < expected_trials:
                            partial_count += 1
                            remaining_trials = expected_trials - completed_trials
                            if args.check_only or args.dry_run:
                                print(f"⚠ PARTIAL:  {config_name} ({completed_trials}/{expected_trials} trials, need {remaining_trials} more)")
                            if not args.check_only:
                                execution_plan.append({
                                    'config_name': config_name,
                                    'module': module_name,
                                    'equation_difficulty': difficulty,
                                    'model_system': system,
                                    'law_version': law_version,
                                    'noise_level': noise_level,
                                    'trials_needed': remaining_trials,
                                    'status': 'partial'
                                })
                        else:
                            missing_count += 1
                            if args.check_only or args.dry_run:
                                print(f"✗ MISSING:  {config_name} (0/{expected_trials} trials)")
                            if not args.check_only:
                                execution_plan.append({
                                    'config_name': config_name,
                                    'module': module_name,
                                    'equation_difficulty': difficulty,
                                    'model_system': system,
                                    'law_version': law_version,
                                    'noise_level': noise_level,
                                    'trials_needed': args.trials_per_law,
                                    'status': 'missing'
                                })
    
    # Generate progress report
    progress_report = generate_progress_report(completed_count, skipped_count, partial_count, failed_count, total_configs)
    print(progress_report)
    
    if args.check_only:
        print("\nCompletion check finished. Exiting.")
        return
    
    if args.dry_run:
        print(f"\nDRY RUN: Would execute {len(execution_plan)} configurations")
        total_trials_needed = sum(config['trials_needed'] for config in execution_plan)
        print(f"Total trials to execute: {total_trials_needed}")
        return
    
    if not execution_plan:
        print("\n🎉 All configurations are complete! Nothing to execute.")
        return
    
    # Prompt for confirmation
    if not args.no_prompt:
        print(f"\n📋 EXECUTION PLAN:")
        print(f"Will execute {len(execution_plan)} configurations")
        total_trials_needed = sum(config['trials_needed'] for config in execution_plan)
        print(f"Total trials to execute: {total_trials_needed}")
        
        response = input("\nProceed with execution? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("Execution cancelled.")
            return
    
    # Execute experiments
    print("\n" + "="*80)
    print("STARTING EXPERIMENT EXECUTION")
    print("="*80)
    
    executed_count = 0
    failed_executions = []
    start_time = time.time()
    
    for i, config in enumerate(execution_plan):
        print(f"\n[{i+1}/{len(execution_plan)}] Executing: {config['config_name']}")
        print(f"Status: {config['status'].upper()}, Trials needed: {config['trials_needed']}")
        
        command = [
            "python", "run_experiments.py",
            "--module", config['module'],
            "--equation_difficulty", config['equation_difficulty'],
            "--model_system", config['model_system'],
            "--law_version", config['law_version'] if config['law_version'] is not None else "None",
            "--trials", str(config['trials_needed']),
            "--model_name", args.model_name,
            "--agent_backend", args.agent_backend,
            "--noise", str(config['noise_level'])
        ]

        print(f"Command: {' '.join(command)}")
        
        try:
            subprocess.run(command, check=True)
            executed_count += 1
            print(f"✓ SUCCESS: {config['config_name']}")
        except subprocess.CalledProcessError as e:
            failed_executions.append({
                'config': config,
                'error': str(e),
                'return_code': e.returncode,
                'stdout': e.stdout,
                'stderr': e.stderr
            })
            print(f"✗ FAILED: {config['config_name']}")
            print(f"  Return code: {e.returncode}")
            print(f"  Error: {e.stderr[:200]}{'...' if len(e.stderr) > 200 else ''}")
            print("  Continuing with next configuration...")
        except KeyboardInterrupt:
            print(f"\n\n⚠ INTERRUPTED: Execution stopped by user")
            print(f"Progress: {executed_count}/{len(execution_plan)} configurations completed")
            break
        
        # Progress update
        elapsed = time.time() - start_time
        if i > 0:
            avg_time = elapsed / (i + 1)
            remaining_time = avg_time * (len(execution_plan) - i - 1)
            print(f"Progress: {i+1}/{len(execution_plan)} ({(i+1)/len(execution_plan)*100:.1f}%), "
                  f"ETA: {remaining_time/60:.1f} min")
    
    # Final summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"✓ Successful: {executed_count}/{len(execution_plan)} configurations")
    print(f"✗ Failed: {len(failed_executions)} configurations")
    print(f"⏱ Total time: {total_time/60:.1f} minutes")
    
    if failed_executions:
        print(f"\n❌ FAILED CONFIGURATIONS:")
        for failure in failed_executions:
            print(f"  - {failure['config']['config_name']}: {failure['error']}")
    
    print("\n🏁 Evaluation run finished.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Execution interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()