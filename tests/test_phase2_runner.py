from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE2_PATH = ROOT / "kaggle" / "phase2" / "run_phase2.py"


def load_phase2_module():
    spec = importlib.util.spec_from_file_location("phase2_runner", PHASE2_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["phase2_runner"] = module
    spec.loader.exec_module(module)
    return module


class Phase2RunnerContractTest(unittest.TestCase):
    def test_stage_tokens_parse_to_ordered_int_tuple(self) -> None:
        phase2 = load_phase2_module()

        self.assertEqual(phase2.parse_stage_tokens("64,32,16"), (64, 32, 16))

    def test_forward_conditions_parse_and_reject_unknown_values(self) -> None:
        phase2 = load_phase2_module()

        self.assertEqual(
            phase2.parse_forward_conditions("average_pool, random_select"),
            ("average_pool", "random_select"),
        )
        with self.assertRaises(ValueError):
            phase2.parse_forward_conditions("average_pool,unknown")

    def test_strided_indices_use_exact_stride_when_divisible(self) -> None:
        phase2 = load_phase2_module()

        self.assertEqual(phase2.compute_strided_indices(8, 4), [0, 2, 4, 6])
        self.assertEqual(phase2.compute_strided_indices(5, 5), [0, 1, 2, 3, 4])

    def test_matched_sigmas_choose_nearest_initial_loss(self) -> None:
        phase2 = load_phase2_module()

        matched = phase2.choose_matched_sigmas(
            {"z1": 0.01, "z2": 0.04},
            {"sigma_0.05": 0.0025, "sigma_0.1": 0.01, "sigma_0.2": 0.04},
        )

        self.assertEqual(matched["z1"]["sigma_name"], "sigma_0.1")
        self.assertEqual(matched["z2"]["sigma_name"], "sigma_0.2")

    def test_gate_summary_passes_when_compression_beats_corruption(self) -> None:
        phase2 = load_phase2_module()

        def stage(mse: float, cosine: float, delta_nll: float) -> dict:
            return {
                "validation": {"final_loss": mse},
                "final": {
                    "mse": mse,
                    "cosine": cosine,
                    "latent_use": {
                        "relative_perturbation_sensitivity": 0.02,
                        "ablation_delta_mse": 0.002,
                        "swap_delta_mse": 0.002,
                    },
                    "generation_bridge": {"delta_token_nll_vs_h0": delta_nll},
                },
            }

        metrics = {
            "results": {
                "average_pool": {
                    "z1": stage(0.01, 0.9, 0.1),
                    "z2": stage(0.02, 0.8, 0.2),
                    "z3": stage(0.03, 0.7, 0.3),
                },
                "strided_select": {
                    "z1": stage(0.015, 0.85, 0.15),
                    "z2": stage(0.03, 0.75, 0.25),
                    "z3": stage(0.05, 0.65, 0.35),
                },
                "random_select": {
                    "z1": stage(0.04, 0.6, 0.5),
                    "z2": stage(0.05, 0.5, 0.6),
                    "z3": stage(0.06, 0.4, 0.7),
                },
                "gaussian_noise": {
                    "z1": stage(0.05, 0.5, 0.6),
                    "z2": stage(0.06, 0.4, 0.7),
                    "z3": stage(0.07, 0.3, 0.8),
                },
            }
        }

        gates = phase2.evaluate_phase2_gates(metrics)

        self.assertTrue(gates["overall_pass"])
        self.assertTrue(gates["P2-G1"]["pass"])
        self.assertTrue(gates["P2-G2"]["pass"])
        self.assertTrue(gates["P2-G6"]["pass"])


if __name__ == "__main__":
    unittest.main()
