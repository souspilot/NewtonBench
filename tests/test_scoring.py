import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from newton_common import (  # noqa: E402
    _module_parameter_names,
    calculate_trial_stats,
    compute_verdicts,
    load_trial_failures,
    load_trials,
    results_by_trial_csv,
    update_results,
    verdicts_csv_path,
)
from structural_equivalence import check_constant_equivalence  # noqa: E402


class TrialProtocolTests(unittest.TestCase):
    def test_budget_analysis_artifacts_are_result_tree_specific(self):
        a = "/scratch/budget_evaluation_results_200_cap"
        b = "/scratch/budget_evaluation_results_400_cap"
        self.assertNotEqual(results_by_trial_csv(True, a), results_by_trial_csv(True, b))
        self.assertNotEqual(verdicts_csv_path("model", True, a),
                            verdicts_csv_path("model", True, b))

    def _write_trial(self, root, trial_id, *, version=1, accuracy=0.0,
                     rmsle=0.1, fail=False):
        trials = (Path(root) / "model" / "m0_gravity" / "vanilla_agent" /
                  "easy" / "v0" / f"vanilla_equation_noise0_0_v{version}" / "trials")
        trials.mkdir(parents=True, exist_ok=True)
        suffix = "_fail" if fail else ""
        path = trials / f"trial{trial_id}{suffix}.json"
        evaluation = {"rmsle": rmsle}
        if accuracy is not None:
            evaluation["exact_accuracy"] = accuracy
        data = {
            "trial_id": trial_id,
            "module_name": "m0_gravity",
            "model_name": "model",
            "noise_level": 0.0,
            "equation_difficulty": "easy",
            "model_system": "vanilla_equation",
            "law_version": "v0",
            "agent_backend": "vanilla_agent",
            "status": "failed" if fail else "completed",
            "rounds": 3,
            "num_experiments": 2,
            "total_tokens": 100,
            "submitted_law": "def discovered_law(mass1, mass2, distance):\n return 1",
            "evaluation": evaluation,
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_loader_caps_completed_trials_and_separates_runner_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_trial(tmp, 0, accuracy=1.0)
            self._write_trial(tmp, 1, accuracy=None, rmsle=float("nan"))
            self._write_trial(tmp, 2, accuracy=1.0)
            self._write_trial(tmp, 3, accuracy=0.0)
            self._write_trial(tmp, 3, version=2, accuracy=1.0)
            completed_four = self._write_trial(tmp, 4, accuracy=1.0)
            self._write_trial(tmp, 4, version=2, accuracy=0.0, fail=True)
            self._write_trial(tmp, 5, accuracy=0.0, fail=True)

            trials = load_trials(tmp, "model")
            self.assertEqual(trials["trial_id"].tolist(), [1, 2, 3, 4])
            self.assertEqual(trials["exact_accuracy"].tolist(), [0.0, 1.0, 1.0, 1.0])
            self.assertEqual(trials.loc[trials["trial_id"] == 4, "path"].item(),
                             str(completed_four))

            failures = load_trial_failures(tmp, "model")
            self.assertEqual(failures["trial_id"].tolist(), [5])

    def test_missing_rmsle_never_changes_accuracy_denominator(self):
        frame = pd.DataFrame({
            "trial_id": [0, 1, 2, 3],
            "exact_accuracy": [1.0, 0.0, 1.0, np.nan],
            "rmsle": [0.0, np.nan, 0.2, np.nan],
        })
        accuracy, _, rmsle, _ = calculate_trial_stats(frame)
        self.assertEqual(accuracy, 0.5)
        self.assertEqual(rmsle, 0.1)

    def test_results_csv_uses_the_same_trial_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            for trial_id in range(5):
                self._write_trial(
                    tmp, trial_id,
                    accuracy=None if trial_id == 4 else float(trial_id % 2),
                    rmsle=float("nan") if trial_id == 4 else 0.1,
                )
            self._write_trial(tmp, 5, accuracy=0.0, fail=True)
            csv_path = str(Path(tmp) / "results.csv")

            update_results("model", tmp, csv_path=csv_path)
            scored = pd.read_csv(csv_path)

            self.assertEqual(scored["trial_id"].tolist(), [1, 2, 3, 4])
            self.assertEqual(scored["exact_accuracy"].tolist(), [1.0, 0.0, 1.0, 0.0])


class SymbolicVerdictTests(unittest.TestCase):
    def test_expected_signature_loads_without_api_dependencies(self):
        self.assertEqual(_module_parameter_names("m0_gravity"),
                         ["mass1", "mass2", "distance"])

    def test_required_parameters_and_zero_law_cannot_pass(self):
        expected = ["mass1", "mass2", "distance"]
        ground_truth = "HIDDEN_CONSTANT * mass1 * mass2 / distance ** 2"
        omitted = "def discovered_law(mass1, distance):\n return mass1 / distance ** 2"
        zero = "def discovered_law(mass1, mass2, distance):\n return 0"
        self.assertEqual(check_constant_equivalence(omitted, ground_truth, expected),
                         "structurally_different")
        self.assertEqual(check_constant_equivalence(zero, ground_truth, expected),
                         "structurally_different")

    def test_renamed_parameters_are_compared_positionally(self):
        submitted = "def discovered_law(x, y, z):\n return 7 * x * y / z ** 2"
        ground_truth = "HIDDEN_CONSTANT * mass1 * mass2 / distance ** 2"
        self.assertEqual(
            check_constant_equivalence(
                submitted, ground_truth, ["mass1", "mass2", "distance"]),
            "constant_equivalent",
        )

    def test_verdicts_are_tri_state_and_adjudications_are_explicit(self):
        rows = pd.DataFrame([
            {
                "path": "proved-pass",
                "module": "m0_gravity",
                "submitted_law": (
                    "def discovered_law(mass1, mass2, distance):\n"
                    " return 7 * mass1 * mass2 / distance ** 1.5"
                ),
                "ground_truth_law": "HIDDEN_CONSTANT * mass1 * mass2 / distance ** 1.5",
                "symbolic_equivalent": False,
                "exact_accuracy": 0.0,
                "rmsle": 100.0,
            },
            {
                "path": "proved-fail",
                "module": "m0_gravity",
                "submitted_law": (
                    "def discovered_law(mass1, mass2, distance):\n"
                    " return mass1 / distance"
                ),
                "ground_truth_law": "HIDDEN_CONSTANT * mass1 * mass2 / distance ** 1.5",
                "symbolic_equivalent": True,
                "exact_accuracy": 1.0,
                "rmsle": 0.0,
            },
            {
                "path": "needs-review",
                "module": "m10_be_distribution",
                "submitted_law": (
                    "def discovered_law(omega, T):\n"
                    " return 1 / (math.exp(1e-14 * omega / T) + 1)"
                ),
                "ground_truth_law": "1 / (np.exp(HIDDEN_CONSTANT * omega / T) + 1)",
                "symbolic_equivalent": True,
                "exact_accuracy": 1.0,
                "rmsle": 0.0,
            },
        ])

        verdicts = compute_verdicts(rows, rmsle_threshold=1e-3)
        self.assertIs(verdicts.loc[0, "verified_success"], np.bool_(True))
        self.assertIs(verdicts.loc[1, "verified_success"], np.bool_(False))
        self.assertTrue(pd.isna(verdicts.loc[2, "verified_success"]))
        self.assertEqual(verdicts.loc[2, "agreement_bucket"], "unresolved")

        adjudications = pd.DataFrame({
            "path": ["needs-review"],
            "adjudicated_success": [True],
            "adjudicator": ["local-judge-plus-human"],
        })
        adjudicated = compute_verdicts(rows, rmsle_threshold=1e-3,
                                       adjudications=adjudications)
        self.assertTrue(adjudicated.loc[2, "verified_success"])
        self.assertEqual(adjudicated.loc[2, "verification_source"],
                         "adjudication:local-judge-plus-human")

        independently_judged = rows.assign(
            judge_model="gemma4-31b",
            evaluated_model="muse-glimmer-30b",
        )
        fallback = compute_verdicts(
            independently_judged,
            rmsle_threshold=1e-3,
            independent_judge="gemma4-31b",
        )
        self.assertTrue(fallback.loc[2, "verified_success"])
        self.assertEqual(
            fallback.loc[2, "verification_source"],
            "independent_judge:gemma4-31b",
        )
        # Decisive deterministic results remain primary even when the judge
        # disagrees with them.
        self.assertTrue(fallback.loc[0, "verified_success"])
        self.assertFalse(fallback.loc[1, "verified_success"])

        with self.assertRaises(SystemExit):
            compute_verdicts(
                independently_judged.assign(evaluated_model="gemma4-31b"),
                rmsle_threshold=1e-3,
                independent_judge="gemma4-31b",
            )


if __name__ == "__main__":
    unittest.main()
