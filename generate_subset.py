"""
Generate configs/representative_subset.json.

NewtonBench's full protocol is a 324-task grid:
    12 modules  x  3 equation_difficulty (easy/medium/hard)
    x  3 law_version per difficulty (v0/v1/v2)
    x  3 model_system (vanilla_equation/simple_system/complex_system)

Each cell of the paper's Table 2 (aggregated) / Appendix B.1 (per-domain) is the
mean over 12 runs (3 law_versions x 4 trials) for one (module, difficulty, system)
combination -- e.g. Figure 10's values are all multiples of 1/12 (91.7% = 11/12,
16.7% = 2/12, etc.). Averaging over FEWER than 3 versions can never reproduce that
same 1/12 resolution, so there's no way to be directly comparable to a published
cell without running all 3 versions x 4 trials for it.

This subset therefore does NOT thin law versions or trials within a cell. Instead
it thins which CELLS get run at all:

    For each of the 12 modules, pick 2 of its 9 (difficulty, system) cells and run
    those at FULL fidelity -- all 3 law versions x 4 trials = 12 trials/cell,
    exactly matching the paper's protocol for that specific cell. The other 7
    cells for that module are skipped entirely (not run at reduced fidelity --
    just not run).

    12 modules x 2 cells x 3 versions = 72 configurations
    72 configs x 4 trials/law = 288 total trials per (model, agent_backend)
    (vs. 324 configs / 1,296 trials for the full benchmark)

Every module is covered (unlike an earlier version of this script that dropped 6
modules outright), and every one of the 9 (difficulty, system) cells is covered by
at least 2 different modules (see the coverage printout below), so you still get a
reasonable per-difficulty and per-system read across domains -- just not a dense
one. Whichever cells got selected, their numbers should land in the same 1/12-
increment format as the paper's tables and are directly comparable to the
corresponding entries in Appendix B.1.

Cell selection: modules are processed in sorted order (m0, m1, m10, m11, m2, ...);
each module's two cells are picked by walking a fixed rotation across the 9-cell
grid (index i%9 and (i+4)%9), which happens to also guarantee every module's two
cells differ in difficulty (an easier cell and a harder cell), and spreads the 24
total cell-picks almost perfectly evenly across the 9 possible cells (2-3 picks
each). Run this script again if you add/remove modules; it's deterministic.
"""
import glob
import json
import os

DIFFICULTIES = ["easy", "medium", "hard"]
SYSTEMS = ["vanilla_equation", "simple_system", "complex_system"]
CELLS = [(d, s) for d in DIFFICULTIES for s in SYSTEMS]  # 9 cells, index 0..8
CELL_OFFSET = 4  # coprime with 9 -> good spread; also keeps primary/secondary difficulty distinct

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "representative_subset.json")


def discover_modules() -> list:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mods = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(repo_root, "modules", "m*_*"))
    )
    if not mods:
        raise RuntimeError("No modules found under modules/ -- run this from within the repo.")
    return mods


def build_subset(modules: list) -> dict:
    subset = {}
    for i, module_name in enumerate(modules):
        primary = CELLS[i % len(CELLS)]
        secondary = CELLS[(i + CELL_OFFSET) % len(CELLS)]
        subset[module_name] = [
            {"difficulty": primary[0], "system": primary[1]},
            {"difficulty": secondary[0], "system": secondary[1]},
        ]
    return subset


def main():
    modules = discover_modules()
    subset = build_subset(modules)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(subset, f, indent=2, sort_keys=True)

    total_cells = sum(len(v) for v in subset.values())
    print(f"Wrote {OUTPUT_PATH}")
    print(f"{len(modules)} modules x 2 cells each = {total_cells} (module, difficulty, system) "
          f"selections, each at full fidelity (3 law versions x 4 trials)")
    print(f"x 3 versions = {total_cells * 3} task configurations "
          f"(vs. {len(modules) * 9 * 3} for the full benchmark)")
    print(f"x 4 trials/law = {total_cells * 3 * 4} total trials per (model, agent_backend)")

    from collections import Counter
    coverage = Counter()
    for cells in subset.values():
        for c in cells:
            coverage[(c["difficulty"], c["system"])] += 1
    print("\nCoverage per (difficulty, system) cell across all 12 modules:")
    for d, s in CELLS:
        print(f"  {d:6s} / {s:16s}: {coverage[(d, s)]} module(s)")


if __name__ == "__main__":
    main()